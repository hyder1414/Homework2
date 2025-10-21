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

def count_ngrams_trigram(sents):
    tri = Counter()
    bi  = Counter()
    for sent in sents:
        if len(sent) < 3:
            continue
        for i in range(2, len(sent)):
            w1, w2, w3 = sent[i-2], sent[i-1], sent[i]
            tri[(w1, w2, w3)] += 1
            bi[(w1, w2)] += 1
    return tri, bi

def sentence_logprob_add1_trigram(sent, tri, bi, V):
    """Laplace smoothing:
       P(w3 | w1,w2) = (count(w1,w2,w3) + 1) / (count(w1,w2) + V)
    """
    logp = 0.0
    for i in range(2, len(sent)):
        w1, w2, w3 = sent[i-2], sent[i-1], sent[i]
        num = tri.get((w1, w2, w3), 0) + 1
        den = bi.get((w1, w2), 0) + V
        logp += math.log(num / den)
    return logp

def corpus_perplexity_add1_trigram(sents, tri, bi, V):
    total_pred = 0  # number of predicted tokens = sum over sentences of (len - 2)
    total_logp = 0.0
    for sent in sents:
        n_pred = max(0, len(sent) - 2)
        total_pred += n_pred
        total_logp += sentence_logprob_add1_trigram(sent, tri, bi, V)
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
    args = ap.parse_args()

    # Load data
    train_path = os.path.join(args.data_dir, args.train)
    valid_path = os.path.join(args.data_dir, args.valid)
    test_path  = os.path.join(args.data_dir, args.test)

    train_raw = read_sentences(train_path)
    valid_raw = read_sentences(valid_path)
    test_raw  = read_sentences(test_path)

    # Vocab from train only
    vocab = build_vocab(train_raw, args.min_freq)
    V = len(vocab) - 1  # exclude <s>? Common choice is to include all word types.
    # We'll include ALL tokens in V, including specials, to keep it simple & consistent:
    V = len(vocab)

    # Map OOV and add trigram boundaries
    train = add_ngram_boundaries(map_unk(train_raw, vocab), 3)
    valid = add_ngram_boundaries(map_unk(valid_raw, vocab), 3)
    test  = add_ngram_boundaries(map_unk(test_raw,  vocab), 3)

    # Count trigrams & bigram contexts on TRAIN
    tri, bi = count_ngrams_trigram(train)

    # Evaluate
    eval_sents = valid if args.eval_split == "valid" else test
    ppl = corpus_perplexity_add1_trigram(eval_sents, tri, bi, V)

    print(f"INFO: model=trigram+add1, split={args.eval_split}, min_freq={args.min_freq}, |V|={V}")
    print(f"Counts: #trigrams={len(tri)}, #bigrams(contexts)={len(bi)}")
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
