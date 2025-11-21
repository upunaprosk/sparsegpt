import math
import os
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SGPT_PRECISION = torch.float32


class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros(
            (self.columns, self.columns), device=self.dev, dtype=SGPT_PRECISION
        )
        self.nsamples = 0
        # debias strength; set via env, e.g. os.environ["ALPHA"] = "1.0"
        self.alpha = float(os.environ.get("ALPHA", "0"))

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        # inp: [B, T, H] or [T, H]
        inp = inp.to(self.dev)
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # -> [1, T, H]

        batch_size = inp.shape[0]

        # optional debias term H_x01 (only when we really have a batch of size 2)
        H_x01 = None
        if (
            self.alpha != 0.0
            and batch_size == 2
            and isinstance(self.layer, (nn.Linear, transformers.Conv1D))
            and inp.dim() == 3
        ):
            # inp: [2, T, H]
            X0 = inp[0]  # [T, H]
            X1 = inp[1]  # [T, H]

            # treat features along columns -> [H, T]
            X0 = X0.to(dtype=SGPT_PRECISION).t()  # [H, T]
            X1 = X1.to(dtype=SGPT_PRECISION).t()  # [H, T]

            # number of "samples" used for scaling; follow same spirit as main H
            samples_Hx01 = (self.nsamples + batch_size) / 2.0 if self.nsamples > 0 else batch_size / 2.0
            if samples_Hx01 <= 0:
                samples_Hx01 = 1.0

            delta = math.sqrt(2.0 / samples_Hx01) * (X0 - X1)  # [H, T]
            # this is a proper [H, H] matrix, same shape as H
            H_x01 = delta.matmul(delta.t())  # [H, H]

        tmp = batch_size

        # standard SparseGPT input preprocessing for Linear / Conv1D
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])  # [B*T, H]
            inp = inp.t()  # [H, B*T]

        # standard Hessian accumulation
        if self.nsamples > 0:
            self.H *= self.nsamples / (self.nsamples + tmp)
        else:
            self.H.zero_()
        self.nsamples += tmp

        inp = inp.to(dtype=SGPT_PRECISION)
        inp = math.sqrt(2.0 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())  # [H, H]

        # add debias term if we actually computed it
        if H_x01 is not None:
            self.H += self.alpha * H_x01.to(self.H.device, dtype=SGPT_PRECISION)

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=0.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.to(dtype=SGPT_PRECISION)

        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev, dtype=SGPT_PRECISION)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i:(i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, "quantizer"):
                    q = quantize(
                        q.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
