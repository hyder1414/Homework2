# N‑Gram Language Modeling (PTB) - Quick Start

## Project Layout
```
.
├── ptbdataset/
│   ├── ptb.train.txt
│   ├── ptb.valid.txt
│   └── ptb.test.txt
└── src/
    ├── prep.py
    ├── mle.py
    ├── trigram_add1.py
    ├── interp.py
    ├── stupid_backoff.py
    └── generate.py
```

## Environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# dataset: place the PTB text files under ptbdataset/
```

## Reproduce Key Results
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

# Generate 5 sentences (best model: Backoff α=0.80)
python3 src/generate.py --data_dir ptbdataset --model backoff --alpha 0.80 --num 5 --max_len 25 --seed 7
```

## Notes
- Vocab from **train only** with `min_freq=2`; specials: `<unk>`, `<s>`, `</s>`.
- Perplexity uses natural logs. Unsmoothed MLE returns **INF** if any test n‑gram is unseen.
- For reports: include OOV rates (valid≈0.37%, test≈0.02%), |V|≈9,971, and your tuned hyperparameters.
