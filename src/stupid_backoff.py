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

def logprob_stupid_backoff(sent, uni, bi, tri, uni_total, alpha):
    """Brants et al. 'Stupid Backoff' (no normalization):
       if c(w1,w2,w3)>0: p = c(w1,w2,w3)/c(w1,w2)
       elif c(w2,w3)>0  : p = α * c(w2,w3)/c(w2)
       else              : p = α^2 * c(w3)/uni_total
    """
    lp = 0.0
    for i in range(2, len(sent)):
        w1, w2, w3 = sent[i-2], sent[i-1], sent[i]

        tri_denom = bi.get((w1, w2), 0)
        tri_num   = tri.get((w1, w2, w3), 0)
        if tri_num > 0 and tri_denom > 0:
            p = tri_num / tri_denom
        else:
            bi_denom = uni.get(w2, 0)
            bi_num   = bi.get((w2, w3), 0)
            if bi_num > 0 and bi_denom > 0:
                p = alpha * (bi_num / bi_denom)
            else:
                # unigram backoff; ensure > 0 thanks to <unk> mapping
                p = (alpha ** 2) * (uni.get(w3, 0) / uni_total if uni_total > 0 else 0.0)

        if p <= 0.0:
            # This shouldn't happen given <unk> and positive alpha.
            return -math.inf
        lp += math.log(p)
    return lp

def corpus_perplexity_stupid_backoff(sents, uni, bi, tri, uni_total, alpha):
    total_pred = 0
    total_logp = 0.0
    for sent in sents:
        n_pred = max(0, len(sent) - 2)
        total_pred += n_pred
        lp = logprob_stupid_backoff(sent, uni, bi, tri, uni_total, alpha)
        if lp == -math.inf:
            return math.inf
        total_logp += lp
    if total_pred == 0:
        return math.inf
    return math.exp(- total_logp / total_pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train", type=str, default="ptb.train.txt")
    ap.add_argument("--valid", type=str, default="ptb.valid.txt")
    ap.add_argument("--test",  type=str, default="ptb.test.txt")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--eval_split", type=str, default="valid", choices=["valid","test"])
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--tune", action="store_true", help="Grid-search alpha on valid")
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

    # Map OOV and add trigram boundaries
    train = add_ngram_boundaries(map_unk(train_raw, vocab), 3)
    valid = add_ngram_boundaries(map_unk(valid_raw, vocab), 3)
    test  = add_ngram_boundaries(map_unk(test_raw,  vocab), 3)

    # Counts
    uni, bi, tri, uni_total = count_all(train)

    if args.tune:
        best_alpha, best_ppl = None, math.inf
        tried = 0
        for a in [0.2, 0.3, 0.4, 0.5, 0.7, 0.8]:
            ppl = corpus_perplexity_stupid_backoff(valid, uni, bi, tri, uni_total, a)
            tried += 1
            if ppl < best_ppl:
                best_alpha, best_ppl = a, ppl
        print(f"TUNED (valid): alpha={best_alpha:.2f} | Perplexity: {best_ppl:.4f} | tried={tried}")
        return

    eval_sents = valid if args.eval_split == "valid" else test
    ppl = corpus_perplexity_stupid_backoff(eval_sents, uni, bi, tri, uni_total, args.alpha)

    print(f"INFO: model=stupid-backoff, split={args.eval_split}, alpha={args.alpha:.2f}, |V|={len(vocab)}")
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
