import os
import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    # use torch_dtype, not dtype
    model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype

    # ---- detect how many *token sequences* we have in total ----
    # dataloader is a list of (input_ids, targets)
    nsamples_total = 0
    for batch in dataloader:
        inp = batch[0]        # [B, T]
        nsamples_total += inp.shape[0]
    # ------------------------------------------------------------

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
            # inp: [B, T, H]
            bsz = inp.shape[0]
            start = cache["i"]
            end = start + bsz
            inps[start:end] = inp
            cache["i"] = end
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    # First pass: run full model once to fill `inps`
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    # ---- decide whether to group samples into pairs ----
    # If ALPHA != 0 we assume we are doing debiasing with paired batches.
    alpha = float(os.environ.get("ALPHA", "0"))
    pair_mode = (alpha != 0.0)
    if pair_mode and (nsamples_total % 2 != 0):
        print("WARNING: ALPHA != 0 but nsamples_total is odd; "
              "last sample will be ignored for pairing.")
    step = 2 if pair_mode else 1
    # ----------------------------------------------------

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
                # inp is a tuple; inp[0] is [B, T, H]
                gpts[name].add_batch(inp[0], out)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # ---- build Hessian: run layer on stored hidden states ----
        # if pair_mode: feed 2 samples together -> batch_size == 2 in SparseGPT
        for j in range(0, nsamples_total, step):
            x = inps[j:j+step]          # [step, T, H]
            out = layer(x, attention_mask=attention_mask)[0]
            outs[j:j+step] = out
        # ----------------------------------------------------------

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            gpts[name].free()

        # recompute outputs with pruned weights to propagate to next layer
        for j in range(0, nsamples_total, step):
            x = inps[j:j+step]
            out = layer(x, attention_mask=attention_mask)[0]
            outs[j:j+step] = out

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
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

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
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

    parser.add_argument(
        "model", type=str,
        help="OPT model to load; pass `facebook/opt-X`."
    )
    parser.add_argument(
        "dataset", type=str,
        choices=["wikitext2", "bbq", "ptb", "c4", "stereoset"],
        help="Where to extract calibration data from."
    )
    parser.add_argument(
        "--seed",
        type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128,
        help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp", type=float, default=.01,
        help="Percent of the average Hessian diagonal to use for dampening."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0,
        help="Target sparsity"
    )
    parser.add_argument(
        "--prunen", type=int, default=0,
        help="N for N:M pruning."
    )
    parser.add_argument(
        "--prunem", type=int, default=0,
        help="M for N:M pruning."
    )
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="Blocksize to use for adaptive mask selection."
    )
    parser.add_argument(
        "--gmp", action="store_true",
        help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16,
        help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1,
        help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000,
        help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only", type=str, default="",
        help="Prune only layers that contain this text."
    )
    parser.add_argument(
        "--invert", action="store_true",
        help="Invert subset."
    )
    parser.add_argument(
        "--save", type=str, default="",
        help="Path to saved model."
    )
    parser.add_argument(
        "--log_wandb", action="store_true",
        help="Whether to log to wandb."
    )

    args = parser.parse_args()

    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_opt(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if "fc2" in n:
                break
        print(time.time() - tick)

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)




