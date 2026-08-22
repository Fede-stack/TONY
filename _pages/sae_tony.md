---
permalink: /sae/
title: "SAE"
author_profile: true
---

{% include toc %}

## TONY.SAE

Sparse Autoencoders trained on Qwen3 embeddings, used to decompose a text into a small set of **interpretable, human-labelled features**. Instead of a single opaque vector, each text is described by the few sparse latents that fire most strongly, each carrying a natural-language label — a transparent alternative to black-box classification.

> The pretrained SAEs are a work in progress; the models on the Hub are still being updated.

```python
from TONY.SAE import SAEInterpreter
```

Three SAEs of increasing sparsity budget are shipped together — `SAE16`, `SAE32`, `SAE64` (the number is the top-k of active latents) — and can be queried individually or all at once.

### `SAEInterpreter`

```python
SAEInterpreter(
    hf_repo_id="FritzStack/SAE-mental-health", hf_token=None,
    artifacts_dir="./sae_artifacts", openrouter_key=None,
    embedding_model="qwen/qwen3-embedding-4b", use_huggingface=False,
    max_freq=0.15, device=None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `hf_repo_id` | `FritzStack/SAE-mental-health` | HuggingFace repo holding weights, configs and feature labels. |
| `hf_token` | `None` | Required if the repo is private. |
| `artifacts_dir` | `./sae_artifacts` | Local cache; files already present are not re-downloaded. |
| `openrouter_key` | `None` | Required unless `use_huggingface=True`. |
| `embedding_model` | `qwen/qwen3-embedding-4b` | Encoder used to embed the texts. |
| `use_huggingface` | `False` | Embed locally (mean pooling, max 512 tokens) instead of via OpenRouter. |
| `max_freq` | `0.15` | Drops features firing on more than this fraction of the training data — filters out generic, non-informative latents. |
| `device` | `None` | `"cuda"`, `"mps"` or `"cpu"`; auto-detected. |

Artifacts are downloaded and all three SAEs loaded at construction time, so the first instantiation is slow and subsequent calls are not.

### Usage

The instance is callable:

```python
__call__(texts, top_k=3, sae="SAE64", batch_size=32) -> dict | list[dict]
```

- `texts` — a string (returns a `dict`) or a list of strings (returns a `list[dict]`).
- `top_k` — features returned per text per SAE, ranked by activation score.
- `sae` — `"SAE16"`, `"SAE32"`, `"SAE64"` or `"all"`.
- `batch_size` — batching for the embedding step. Embeddings are computed once and reused across SAEs.

```python
interpreter = SAEInterpreter(hf_token="hf_...", openrouter_key="sk-or-...")

interpreter("I haven't left my bed in three days")
# {'text': ..., 'sae': 'SAE64',
#  'features': [{'rank': 1, 'score': 2.41, 'feature_id': 812,
#                'label': 'social withdrawal / staying home', 'sae': 'SAE64'}, ...]}

interpreter("I haven't left my bed", sae="SAE64")   # features grouped by SAE
results = interpreter(["text 1", "text 2"], top_k=5)
```

With `sae="all"`, `features` is a dict keyed by SAE name rather than a flat list.

### Notes

- Only latents with a positive activation and passing the `max_freq` filter are returned, so a text may yield fewer than `top_k` features.
- OpenRouter calls retry up to five times with exponential backoff; a persistent failure raises `RuntimeError`.
- Requires `torch`, `numpy`, `huggingface_hub`, plus `transformers` when `use_huggingface=True`.
