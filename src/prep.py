import argparse, os
from collections import Counter

SPECIALS = ["<unk>", "<s>", "</s>"]

def read_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        # each line is already tokenized in PTB; split on whitespace
        return [line.strip().split() for line in f if line.strip()]

def build_vocab(train_sents, min_freq):
    cnt = Counter(tok for sent in train_sents for tok in sent)
    vocab = {sp: i for i, sp in enumerate(SPECIALS)}  # reserve specials
    for tok, c in cnt.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab, cnt

def map_unk(sents, vocab):
    unk = "<unk>"
    return [[tok if tok in vocab else unk for tok in sent] for sent in sents]

def add_ngram_boundaries(sents, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    k = n - 1
    out = []
    for sent in sents:
        out.append((["<s>"] * k) + sent + ["</s>"])
    return out

def oov_rate(sents, vocab):
    total = 0
    oov = 0
    for sent in sents:
        for tok in sent:
            total += 1
            if tok not in vocab:
                oov += 1
    return (oov / total) if total else 0.0, oov, total

def preview(name, sents, n=3, num=2):
    print(f"\n--- {name} PREVIEW (first {num} sentences) ---")
    for i, sent in enumerate(sents[:num]):
        print(f"[{i}] raw: {' '.join(sent[:40])}{' ...' if len(sent)>40 else ''}")
    sents_unk = map_unk(sents, vocab)  # uses outer-scope vocab (set later)
    sents_ng = add_ngram_boundaries(sents_unk, n)
    for i, sent in enumerate(sents_ng[:num]):
        print(f"[{i}] ngram({n}): {' '.join(sent[:40])}{' ...' if len(sent)>40 else ''}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder with ptb.train.txt / ptb.valid.txt / ptb.test.txt")
    ap.add_argument("--train", type=str, default="ptb.train.txt")
    ap.add_argument("--valid", type=str, default="ptb.valid.txt")
    ap.add_argument("--test",  type=str, default="ptb.test.txt")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--n", type=int, default=3, help="n-gram order for preview")
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, args.train)
    valid_path = os.path.join(args.data_dir, args.valid)
    test_path  = os.path.join(args.data_dir, args.test)

    for p in [train_path, valid_path, test_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing file: {p}")

    train_sents = read_sentences(train_path)
    valid_sents = read_sentences(valid_path)
    test_sents  = read_sentences(test_path)

    print(f"Loaded: train={len(train_sents)} sents, valid={len(valid_sents)} sents, test={len(test_sents)} sents")

    vocab, train_counts = build_vocab(train_sents, args.min_freq)
    print(f"Vocab size (incl specials): {len(vocab)} (min_freq={args.min_freq})")
    print("Top-10 tokens in train:", [w for w,_ in train_counts.most_common(10)])

    # OOV rates BEFORE mapping (diagnostic)
    valid_oov_rate, v_oov, v_tot = oov_rate(valid_sents, vocab)
    test_oov_rate,  t_oov, t_tot  = oov_rate(test_sents, vocab)
    print(f"OOV valid: {v_oov}/{v_tot} = {valid_oov_rate:.2%}")
    print(f"OOV test : {t_oov}/{t_tot} = {test_oov_rate:.2%}")

    # Make preview using current vocab and n
    # (vocab is referenced inside preview via closure—intentional)
    globals()['vocab'] = vocab
    preview("TRAIN", train_sents, n=args.n)
    preview("VALID", valid_sents, n=args.n)
