import os
import time

import torch
import torch.nn as nn

from sparsegpt import *
from modelutils import *
from quant import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = 2048
    return model


@torch.no_grad__()
def llama_sequential(model, dataloader, dev):
    print("Starting...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    nsamples_total = 0
    for batch in dataloader:
        nsamples_total += batch[0].shape[0]
    inps = torch.zeros(
        (nsamples_total, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            bsz = inp.shape[0]
            start = cache["i"]
            end = start + bsz
            inps[start:end] = inp
            cache["i"] = end
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    print("Ready.")
    alpha = float(os.environ.get("ALPHA", "0"))
    pair_mode = (alpha != 0.0)
    if pair_mode and (nsamples_total % 2 != 0):
        print("WARNING: ALPHA != 0 but nsamples_total is odd; last sample will be ignored for pairing.")
    step = 2 if pair_mode else 1

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0], out)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(0, nsamples_total, step):
            x = inps[j:j+step]
            out = layer(x, attention_mask=attention_mask)[0]
            outs[j:j+step] = out

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning...")
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            gpts[name].free()

        for j in range(0, nsamples_total, step):
            x = inps[j:j+step]
            out = layer(x, attention_mask=attention_mask)[0]
            outs[j:j+step] = out

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad__()
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print("Evaluating ...")
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None}

    class CatcherEval(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = CatcherEval(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if getattr(model.model, "norm", None) is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})
    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="LLaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "stereoset"],
        help="Calibration dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--gmp", action="store_true")
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune_only", type=str, default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    args = parser.parse_args()
    if args.log_wandb:
        assert has_wandb, "wandb not installed"
        wandb.init(config=args)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_llama(args.model)
    model.eval()
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        llama_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if "down_proj" in n:
                break
        print(time.time() - tick)
    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, args.log_wandb)
    if args.save:
        model.save_pretrained(args.save)
