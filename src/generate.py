import argparse, os, math, random
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

def next_dist_interp(w1, w2, uni, bi, tri, uni_total, l1, l2, l3, vocab_words):
    probs = []
    for w in vocab_words:
        p_uni = uni.get(w, 0) / uni_total if uni_total > 0 else 0.0
        bi_denom = uni.get(w2, 0)
        p_bi = (bi.get((w2, w), 0) / bi_denom) if bi_denom > 0 else 0.0
        tri_denom = bi.get((w1, w2), 0)
        p_tri = (tri.get((w1, w2, w), 0) / tri_denom) if tri_denom > 0 else 0.0
        p = l1 * p_uni + l2 * p_bi + l3 * p_tri
        probs.append(p)
    s = sum(probs)
    if s <= 0:
        # fallback to unigram
        probs = [(uni.get(w, 0) / uni_total) for w in vocab_words]
        s = sum(probs)
    return [p / s for p in probs]

def next_dist_backoff(w1, w2, uni, bi, tri, uni_total, alpha, vocab_words):
    # Stupid backoff gives scores; we normalize them to sample.
    scores = []
    for w in vocab_words:
        tri_denom = bi.get((w1, w2), 0)
        tri_num   = tri.get((w1, w2, w), 0)
        if tri_num > 0 and tri_denom > 0:
            s = tri_num / tri_denom
        else:
            bi_denom = uni.get(w2, 0)
            bi_num   = bi.get((w2, w), 0)
            if bi_num > 0 and bi_denom > 0:
                s = alpha * (bi_num / bi_denom)
            else:
                s = (alpha ** 2) * (uni.get(w, 0) / uni_total if uni_total > 0 else 0.0)
        scores.append(s)
    ssum = sum(scores)
    if ssum <= 0:
        # fallback to unigram
        scores = [(uni.get(w, 0) / uni_total) for w in vocab_words]
        ssum = sum(scores)
    return [s / ssum for s in scores]

def sample_from(probs, words, rng):
    r = rng.random()
    acc = 0.0
    for p, w in zip(probs, words):
        acc += p
        if r <= acc:
            return w
    return words[-1]

def generate(num_sentences, max_len, model, lambdas, alpha, vocab, uni, bi, tri, uni_total, seed=42):
    rng = random.Random(seed)
    # We can sample any token except "<s>" mid-sentence; allow "</s>" to terminate.
    vocab_words = [w for w in vocab.keys() if w != "<s>"]

    outputs = []
    for _ in range(num_sentences):
        w1, w2 = "<s>", "<s>"
        sent = []
        for _t in range(max_len):
            if model == "interp":
                probs = next_dist_interp(w1, w2, uni, bi, tri, uni_total, *lambdas, vocab_words)
            else:
                probs = next_dist_backoff(w1, w2, uni, bi, tri, uni_total, alpha, vocab_words)
            w = sample_from(probs, vocab_words, rng)
            if w == "</s>":
                break
            if w == "<s>":
                # avoid inserting extra start tokens
                continue
            sent.append(w)
            w1, w2 = w2, w
        outputs.append(" ".join(sent))
    return outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train", type=str, default="ptb.train.txt")
    ap.add_argument("--valid", type=str, default="ptb.valid.txt")
    ap.add_argument("--test",  type=str, default="ptb.test.txt")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--model", type=str, choices=["interp", "backoff"], default="backoff")
    ap.add_argument("--lambda1", type=float, default=0.30)  # uni
    ap.add_argument("--lambda2", type=float, default=0.50)  # bi
    ap.add_argument("--lambda3", type=float, default=0.20)  # tri
    ap.add_argument("--alpha", type=float, default=0.80)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load + prep (train-only vocab)
    trp = os.path.join(args.data_dir, args.train)
    vdp = os.path.join(args.data_dir, args.valid)
    tep = os.path.join(args.data_dir, args.test)

    train_raw = read_sentences(trp)
    valid_raw = read_sentences(vdp)
    test_raw  = read_sentences(tep)

    vocab = build_vocab(train_raw, args.min_freq)
    train = add_ngram_boundaries(map_unk(train_raw, vocab), 3)

    uni, bi, tri, uni_total = count_all(train)

    # Generate
    lambdas = (args.lambda1, args.lambda2, args.lambda3)
    sents = generate(
        num_sentences=args.num,
        max_len=args.max_len,
        model=args.model,
        lambdas=lambdas,
        alpha=args.alpha,
        vocab=vocab,
        uni=uni, bi=bi, tri=tri, uni_total=uni_total,
        seed=args.seed
    )
    print("\n--- Generated Sentences ---")
    for i, s in enumerate(sents, 1):
        print(f"{i}. {s}")

if __name__ == "__main__":
    main()
