import argparse, os, math
from collections import Counter

SPECIALS = ["<unk>", "<s>", "</s>"]

def read_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]

def build_vocab(train_sents, min_freq):
    cnt = Counter(tok for sent in train_sents for tok in sent)
    vocab = {sp: i for i, sp in enumerate(SPECIALS)}
    for tok, c in cnt.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def map_unk(sents, vocab):
    return [[tok if tok in vocab else "<unk>" for tok in sent] for sent in sents]

def add_ngram_boundaries(sents, n=3):
    k = n - 1
    out = []
    for sent in sents:
        out.append((["<s>"] * k) + sent + ["</s>"])
    return out

def count_all(train_sents_with_bounds):
    """Return unigram, bigram, trigram counters and totals for denominators."""
    uni = Counter()
    bi  = Counter()
    tri = Counter()
    for sent in train_sents_with_bounds:
        for i, w in enumerate(sent):
            uni[w] += 1
            if i >= 1:
                bi[(sent[i-1], w)] += 1
            if i >= 2:
                tri[(sent[i-2], sent[i-1], w)] += 1
    total_tokens = sum(uni.values())
    return uni, bi, tri, total_tokens

def logprob_interp_trigram(sent, uni, bi, tri, uni_total, lam1, lam2, lam3):
    """Linear interpolation:
       p = λ1 * P_uni(w) + λ2 * P_bi(w|h1) + λ3 * P_tri(w|h2,h1)
       If a denominator is zero, that component contributes 0.
    """
    lp = 0.0
    for i in range(2, len(sent)):
        w1, w2, w3 = sent[i-2], sent[i-1], sent[i]

        # components
        p_uni = uni.get(w3, 0) / uni_total if uni_total > 0 else 0.0

        bi_denom = uni.get(w2, 0)
        p_bi = (bi.get((w2, w3), 0) / bi_denom) if bi_denom > 0 else 0.0

        tri_denom = bi.get((w1, w2), 0)
        p_tri = (tri.get((w1, w2, w3), 0) / tri_denom) if tri_denom > 0 else 0.0

        p = lam1 * p_uni + lam2 * p_bi + lam3 * p_tri
        if p <= 0.0:
            # should not happen if λ1>0 and uni_total>0
            return -math.inf
        lp += math.log(p)
    return lp

def corpus_perplexity_interp(sents, uni, bi, tri, uni_total, lam1, lam2, lam3):
    total_pred = 0
    total_logp = 0.0
    for sent in sents:
        n_pred = max(0, len(sent) - 2)
        total_pred += n_pred
        lp = logprob_interp_trigram(sent, uni, bi, tri, uni_total, lam1, lam2, lam3)
        if lp == -math.inf:
            return math.inf
        total_logp += lp
    if total_pred == 0:
        return math.inf
    return math.exp(- total_logp / total_pred)

def grid_lambdas(step=0.1):
    """Yield (λ1, λ2, λ3) with λ1+λ2+λ3=1."""
    vals = [i * step for i in range(int(1/step) + 1)]
    for l1 in vals:
        for l2 in vals:
            l3 = 1.0 - l1 - l2
            if l3 < -1e-12:
                continue
            if l3 < 0:  # numerical cleanup
                l3 = 0.0
            if abs(l1 + l2 + l3 - 1.0) <= 1e-9:
                yield (l1, l2, l3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train", type=str, default="ptb.train.txt")
    ap.add_argument("--valid", type=str, default="ptb.valid.txt")
    ap.add_argument("--test",  type=str, default="ptb.test.txt")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--eval_split", type=str, default="valid", choices=["valid","test"])
    ap.add_argument("--lambda1", type=float, default=None)  # unigram
    ap.add_argument("--lambda2", type=float, default=None)  # bigram
    ap.add_argument("--lambda3", type=float, default=None)  # trigram
    ap.add_argument("--tune", action="store_true", help="Grid-search lambdas on valid")
    ap.add_argument("--grid_step", type=float, default=0.1)
    args = ap.parse_args()

    # Load data
    trp = os.path.join(args.data_dir, args.train)
    vdp = os.path.join(args.data_dir, args.valid)
    tep = os.path.join(args.data_dir, args.test)

    train_raw = read_sentences(trp)
    valid_raw = read_sentences(vdp)
    test_raw  = read_sentences(tep)

    # Vocab from train only
    vocab = build_vocab(train_raw, args.min_freq)

    # Map OOV and add trigram boundaries (we evaluate as trigram predictor)
    train = add_ngram_boundaries(map_unk(train_raw, vocab), 3)
    valid = add_ngram_boundaries(map_unk(valid_raw, vocab), 3)
    test  = add_ngram_boundaries(map_unk(test_raw,  vocab), 3)

    # Counts
    uni, bi, tri, uni_total = count_all(train)

    if args.tune:
        best = (None, None, None, math.inf)
        tried = 0
        for l1, l2, l3 in grid_lambdas(step=args.grid_step):
            ppl = corpus_perplexity_interp(valid, uni, bi, tri, uni_total, l1, l2, l3)
            tried += 1
            if ppl < best[3]:
                best = (l1, l2, l3, ppl)
        l1, l2, l3, ppl = best
        print(f"TUNED (valid): λ1={l1:.2f}, λ2={l2:.2f}, λ3={l3:.2f} | Perplexity: {ppl:.4f} | tried={tried}")
        # also show a couple of neighbors if step allows (optional)
        return

    # Use provided lambdas (and normalize if needed)
    if args.lambda1 is None or args.lambda2 is None or args.lambda3 is None:
        # sensible default leaning on higher orders
        l1, l2, l3 = 0.1, 0.2, 0.7
    else:
        s = args.lambda1 + args.lambda2 + args.lambda3
        if s <= 0:
            l1, l2, l3 = 0.1, 0.2, 0.7
        else:
            l1, l2, l3 = args.lambda1 / s, args.lambda2 / s, args.lambda3 / s

    eval_sents = valid if args.eval_split == "valid" else test
    ppl = corpus_perplexity_interp(eval_sents, uni, bi, tri, uni_total, l1, l2, l3)

    print(f"INFO: split={args.eval_split}, |V|={len(vocab)}, min_freq={args.min_freq}")
    print(f"Lambdas: λ1={l1:.3f} (uni), λ2={l2:.3f} (bi), λ3={l3:.3f} (tri)")
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
