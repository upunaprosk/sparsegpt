import random
import json
import urllib.request

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer


def get_bbq(nsamples, seed, seqlen, model, tokenizer):
    """
    Load ambiguous BBQ examples, build batch-2 X0/X1 pairs,
    keep only equal-token-length sequences (pre-padding),
    and then pad/truncate to `seqlen` so the model always sees
    input_ids shape [2, seqlen].
    """

    ds = load_dataset("iproskurina/bbq-sampled-500-each", "All")

    # Merge splits
    records = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            records.extend(list(ds[split]))

    # Filter ambiguous only
    ambig = [ex for ex in records if ex["context_condition"] == "ambig"]

    equal_pairs = []
    random.seed(seed)

    for ex in ambig:
        ctx = ex["context"]
        q   = ex["question"]

        a0, a1, a2 = ex["ans0"], ex["ans1"], ex["ans2"]
        lab = ex["label"]  # index of the "UNDETERMINED" answer

        answers = [a0, a1, a2]

        # Keep the 2 answers that are NOT the “undetermined” one
        keep = [answers[i] for i in range(3) if i != lab]
        if len(keep) != 2:
            continue

        X0, X1 = keep

        full0 = f"{ctx}\nQ: {q}\nA: {X0}"
        full1 = f"{ctx}\nQ: {q}\nA: {X1}"

        # First pass: no padding, just truncation, to check equal token length
        raw0 = tokenizer(full0, truncation=True, max_length=seqlen, return_tensors="pt")
        raw1 = tokenizer(full1, truncation=True, max_length=seqlen, return_tensors="pt")

        if raw0.input_ids.shape[1] != raw1.input_ids.shape[1]:
            continue

        # Second pass: joint tokenization with fixed length seqlen
        t = tokenizer(
            [full0, full1],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seqlen,
        )
        inp = t.input_ids            # [2, seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100           # standard language modeling target mask

        equal_pairs.append((inp, tar))

        if len(equal_pairs) >= nsamples:
            break

    trainloader = equal_pairs

    # Build a long test sequence for eval (optional, simple concat)
    all_texts = []
    for inp, _ in trainloader:
        # decode both rows; note: this is just for eval, not used in pruning
        all_texts.append(tokenizer.decode(inp[0], skip_special_tokens=True))
        all_texts.append(tokenizer.decode(inp[1], skip_special_tokens=True))

    testenc = tokenizer(" ".join(all_texts), return_tensors="pt")

    class Wrap:
        def __init__(self, ids):
            self.input_ids = ids

    return trainloader, Wrap(testenc["input_ids"])


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class T:
        def __init__(self, ids):
            self.input_ids = ids
    return trainloader, T(valenc)


def get_stereoset(nsamples, seed, seqlen, model, tokenizer):
    url = "https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json"
    with urllib.request.urlopen(url) as f:
        data = json.load(f)

    pairs = []
    for entry in data["data"]["intrasentence"]:
        a = None
        b = None
        for s in entry["sentences"]:
            if s["gold_label"] == "anti-stereotype":
                a = s["sentence"]
            elif s["gold_label"] == "stereotype":
                b = s["sentence"]
        if a is not None and b is not None:
            pairs.append((a, b))

    if nsamples < len(pairs):
        pairs = pairs[:nsamples]

    trainloader = []
    for a, b in pairs:
        t = tokenizer(
            [a, b],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seqlen,
        )
        inp = t.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    flat = []
    for a, b in pairs:
        flat.append(a)
        flat.append(b)
    testenc = tokenizer(" ".join(flat), return_tensors="pt")

    class T:
        def __init__(self, ids):
            self.input_ids = ids
    return trainloader, T(testenc.input_ids)


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer(model)
    if "bbq" in name:
        return get_bbq(nsamples, seed, seqlen, model, tokenizer)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    if 'stereoset' in name.lower():
        return get_stereoset(nsamples, seed, seqlen, model, tokenizer)
