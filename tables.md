# Results Table (Markdown-ready)

| Model | Key setting(s) | Valid PPL | Test PPL |
|---|---|---:|---:|
| **MLE Unigram (N=1)** | — | — | **637.6990** |
| **MLE Bigram (N=2)** | — | — | **INF** |
| **MLE Trigram (N=3)** | — | — | **INF** |
| **MLE 4-gram (N=4)** | — | — | **INF** |
| **Trigram + Add-1 (Laplace)** | \|V\|=9971 | **3215.8482** | **3295.0160** |
| **Linear Interp. (uni/bi/tri)** | λ=(0.30, 0.50, 0.20) | **199.4896** | **192.1177** |
| **Stupid Backoff** | α=0.80 | **115.9537** | **112.4923** |