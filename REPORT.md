# N‑Gram Language Modeling and Evaluation — Penn Treebank (PTB)

**Course**: MSML 641 — Homework 2  
**Author**: Haider Khan  
**Repo / Code**: (https://github.com/hyder1414/Homework2)

---

## 1. Overview & Objectives
We implement and evaluate several N‑gram language models on the Penn Treebank (PTB) to illustrate the impact of N‑gram order, smoothing, and backoff on language modeling performance. We report **Perplexity (PPL)** on the held‑out **Test** set, using the **Dev/Validation** set to tune hyperparameters (e.g., interpolation weights and backoff discount).

Models covered:
- Maximum Likelihood Estimation (MLE) for N = 1, 2, 3, 4 (unsmoothed)
- **Add‑1 (Laplace)** smoothing for trigrams
- **Linear Interpolation** of unigram/bigram/trigram (λ₁+λ₂+λ₃=1; tuned on Dev)
- **Stupid Backoff** (α tuned on Dev)

---

## 2. Dataset
**Source:** Kaggle — Penn Treebank dataset (WSJ).  
**Splits:** `ptb.train.txt`, `ptb.valid.txt`, `ptb.test.txt`  
**Counts:** train **42,068** sents; valid **3,370** sents; test **3,761** sents.

Directory structure used in this project:
```
ptbdataset/
 ├── ptb.train.txt
 ├── ptb.valid.txt
 └── ptb.test.txt
```

---

## 3. Pre‑processing & Vocabulary
- Used PTB tokenization **as provided**; each line is one tokenized sentence.
- **Train‑only vocabulary** with a frequency cutoff of **min_freq = 2**.
- All tokens not in the train vocab are mapped to `<unk>` at eval time.
- Special tokens: `<unk>`, `<s>`, `</s>`.
- For N‑gram evaluation with order **n**, sentences are padded with **n−1** `<s>` at the start and a single `</s>` at the end (e.g., for trigrams: `<s> <s> ... </s>`).
- Final vocabulary size (including specials): **9,971**.
- **OOV rates before mapping** (measured w.r.t. train vocab): valid **0.37%**, test **0.02%**.
- Perplexity is computed with **natural log**; when any test event has zero probability under an **unsmoothed** model, PPL is reported as **INF**.

---

## 4. Modeling & Evaluation

### 4.1 MLE Baselines (N = 1, 2, 3, 4)
- Unigram yields finite PPL; higher‑order unsmoothed models suffer from data sparsity (zero probabilities → **INF PPL**).

| Model | Setting | Test PPL |
|---|---|---:|
| **MLE Unigram (N=1)** | — | **637.6990** |
| **MLE Bigram (N=2)** | — | **INF** |
| **MLE Trigram (N=3)** | — | **INF** |
| **MLE 4‑gram (N=4)** | — | **INF** |

### 4.2 Trigram + Add‑1 (Laplace)
Additive smoothing with V≈9,971 types:  
Test PPL is very high due to over‑smoothing (large additive mass per context).

| Split | PPL |
|---|---:|
| **Dev/Valid** | **3215.8482** |
| **Test** | **3295.0160** |

### 4.3 Linear Interpolation (uni/bi/tri)
We tuned λ on Dev with a simple grid (step=0.1), optimizing Dev perplexity.

- **Tuned weights (Dev):** λ₁=**0.30** (unigram), λ₂=**0.50** (bigram), λ₃=**0.20** (trigram)  
- **Dev PPL:** **199.4896**

Using the tuned weights on **Test**:
- **Test PPL:** **192.1177**

### 4.4 Stupid Backoff (tri → bi → uni)
We tuned the discount α on Dev over {0.2, 0.3, 0.4, 0.5, 0.7, 0.8}.

- **Tuned α (Dev):** **0.80**  
- **Dev PPL:** **115.9537**  
- **Test PPL:** **112.4923**  ← **Best overall**

---

## 5. Analysis & Discussion

### 5.1 Impact of N‑gram Order
- Under **unsmoothed MLE**, increasing N leads to **INF PPL** on Test because unseen n‑grams have zero probability, immediately exploding perplexity.
- Unigram avoids zeros but ignores context → high PPL (**~638**).
- Conclusion: **higher‑order models require smoothing/backoff** to be viable.

### 5.2 Why Add‑1 (Laplace) performs poorly
- Laplace adds 1 to **every** possible next token in a context, which with |V|≈10k substantially inflates denominators and over‑penalizes seen events. Result: **very large** PPL (3k+). Good pedagogy, poor LM.

### 5.3 Interpolation vs. Stupid Backoff
- **Interpolation** allocates probability mass to all orders **everywhere**, even when a strong trigram context exists. Our tuned λ emphasized **bigrams (0.50)** and **unigram (0.30)** over **trigram (0.20)**, reflecting real sparsity in trigram contexts. Test PPL **~192** is solid.
- **Stupid Backoff** trusts the highest‑order **when it has evidence**; otherwise it discounts (by α) and backs off without renormalization. This simple heuristic works extremely well on sparse data, yielding the **best** PPL (**~112**).

### 5.4 Qualitative Analysis — Generated Text
**Note:** PTB normalizes many numerals to the token `N`.

**Stupid Backoff (α=0.80):**
1. that more soviet in sales to N to N N  
2. the surplus totaled N percentage plan sad reality is fiat  
3. bids by the new plan being made by creditors to allow the industrial institute or assume about $ N billion sell rule of du pont  
4. within six years army the first admitted mr. bush N million face amount never thought possible in india 's campaign pitches southern comfort <unk> said  
5. fed jordan unions and businessmen see the credit of the

**Interpolation (λ=0.30/0.50/0.20):**
1. it is considering  
2. but in N in <unk> an <unk>  
3. no restrictions  
4. a subsidiary volume on the franchisees  
5. things more and N the francisco of federal regulators in one

**Commentary:** Outputs are locally plausible but often fragmentary due to the short Markov context (≤2 words). Backoff samples appear more “WSJ‑flavored”, while interpolation tends to shorten/hedge (heavier unigram/bigram influence).

---

## 6. Reproducibility — How to Run

### Environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Dataset placed under: ptbdataset/
```

### Scripts
- `src/prep.py` — Vocabulary & OOV preview
- `src/mle.py` — MLE PPL for N=1..4 (returns **INF** if any zero probability)
- `src/trigram_add1.py` — Trigram + Add‑1 (Laplace) PPL
- `src/interp.py` — Linear interpolation; `--tune` grid over λ
- `src/stupid_backoff.py` — Stupid Backoff; `--tune` grid over α
- `src/generate.py` — Sample sentences from Interp/Backoff

### Key Commands
```bash
# MLE (unsmoothed)
python3 src/mle.py --data_dir ptbdataset --n 1 --eval_split test
python3 src/mle.py --data_dir ptbdataset --n 2 --eval_split test
python3 src/mle.py --data_dir ptbdataset --n 3 --eval_split test
python3 src/mle.py --data_dir ptbdataset --n 4 --eval_split test

# Laplace (Trigram)
python3 src/trigram_add1.py --data_dir ptbdataset --eval_split valid
python3 src/trigram_add1.py --data_dir ptbdataset --eval_split test

# Interpolation (tune on Dev, then Test)
python3 src/interp.py --data_dir ptbdataset --tune --grid_step 0.1 --eval_split valid
python3 src/interp.py --data_dir ptbdataset --lambda1 0.30 --lambda2 0.50 --lambda3 0.20 --eval_split test

# Stupid Backoff (tune on Dev, then Test)
python3 src/stupid_backoff.py --data_dir ptbdataset --tune --eval_split valid
python3 src/stupid_backoff.py --data_dir ptbdataset --alpha 0.80 --eval_split test

# Generate sentences (best model: Backoff α=0.80)
python3 src/generate.py --data_dir ptbdataset --model backoff --alpha 0.80 --num 5 --max_len 25 --seed 7
```

---

## 7. Efficiency & Edge Cases
- All counters and lookups are dictionary‑based for O(1) expected access; counting is linear in corpus size.
- INF handling for MLE: if any zero‑prob test event is seen, we return **math.inf** immediately.
- `<unk>` mapping ensures backoff/unigram floors are non‑zero; Start/End tokens prevent degenerate contexts.
- For large |V|, Laplace is intentionally poor (pedagogical). Better smoothers (e.g., **Kneser–Ney**) are recommended for future work.

---

## 8. Conclusion
- **Uns­moothed** higher‑order LMs are unusable on sparse data (PPL=INF).
- **Add‑1** is simple but over‑smooths with large vocabularies (PPL>3k).
- **Interpolation** significantly improves over unigram (PPL~192).
- **Stupid Backoff** (α=0.80) performs best on PTB here (PPL~112), validating a strong backoff heuristic when data are sparse.

---

## 9. Appendix
- Logs reproduced using natural‑log perplexity.
- Numbers normalized as in PTB (`N` stands for most numeric tokens).
- Random seeds set for text generation.
