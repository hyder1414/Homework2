import argparse, os, math
from collections import Counter, defaultdict

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

def add_ngram_boundaries(sents, n):
    k = n - 1
    out = []
    for sent in sents:
        out.append((["<s>"] * k) + sent + ["</s>"])
    return out

def count_ngrams(sents, n):
    """Return (ngram_counts, context_counts) for order n."""
    ngram_counts = Counter()
    context_counts = Counter()
    for sent in sents:
        if len(sent) < n:
            continue
        for i in range(n - 1, len(sent)):
            ngram = tuple(sent[i - n + 1 : i + 1])
            context = ngram[:-1]
            ngram_counts[ngram] += 1
            if n > 1:
                context_counts[context] += 1
    # For unigrams, context is empty; denom is total tokens
    if n == 1:
        total = sum(ngram_counts.values())
        context_counts[tuple()] = total
    return ngram_counts, context_counts

def sentence_logprob_mle(sent, n, ngram_counts, context_counts):
    """Return (logprob, zero_hit) using MLE; zero_hit=True if any prob=0 encountered."""
    logp = 0.0
    zero = False
    for i in range(n - 1, len(sent)):
        ngram = tuple(sent[i - n + 1 : i + 1])
        context = ngram[:-1]
        num = ngram_counts.get(ngram, 0)
        den = context_counts.get(context, 0)
        if den == 0 or num == 0:
            zero = True
            break
        logp += math.log(num / den)
    return logp, zero

def corpus_perplexity_mle(sents, n, ngram_counts, context_counts):
    """Return perplexity (float) or math.inf if any zero prob occurs."""
    total_tokens = 0
    total_logp = 0.0
    for sent in sents:
        total_tokens += max(0, len(sent) - (n - 1))  # number of predicted tokens
        lp, zero = sentence_logprob_mle(sent, n, ngram_counts, context_counts)
        if zero:
            return math.inf
        total_logp += lp
    if total_tokens == 0:
        return math.inf
    # perplexity uses natural log base (consistent across experiments)
    return math.exp(- total_logp / total_tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train", type=str, default="ptb.train.txt")
    ap.add_argument("--valid", type=str, default="ptb.valid.txt")
    ap.add_argument("--test",  type=str, default="ptb.test.txt")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--n", type=int, default=3, help="ngram order")
    ap.add_argument("--eval_split", type=str, default="test", choices=["valid","test"])
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, args.train)
    valid_path = os.path.join(args.data_dir, args.valid)
    test_path  = os.path.join(args.data_dir, args.test)

    train_raw = read_sentences(train_path)
    valid_raw = read_sentences(valid_path)
    test_raw  = read_sentences(test_path)

    # vocab from TRAIN only
    vocab = build_vocab(train_raw, args.min_freq)

    # map OOV to <unk>, then add boundaries for chosen n
    train = add_ngram_boundaries(map_unk(train_raw, vocab), args.n)
    valid = add_ngram_boundaries(map_unk(valid_raw, vocab), args.n)
    test  = add_ngram_boundaries(map_unk(test_raw,  vocab), args.n)

    # train counts
    ngram_counts, context_counts = count_ngrams(train, args.n)

    # eval
    eval_sents = valid if args.eval_split == "valid" else test
    ppl = corpus_perplexity_mle(eval_sents, args.n, ngram_counts, context_counts)

    print(f"INFO: n={args.n}, split={args.eval_split}, min_freq={args.min_freq}")
    print(f"Counts: #ngrams={len(ngram_counts)}, #contexts={len(context_counts)}")
    if math.isinf(ppl):
        print("Perplexity: INF (zero probability encountered)")
    else:
        print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
