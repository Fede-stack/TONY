---
permalink: /advMarkers/
title: "Advanced Markers"
author_profile: true
redirect_from:
  - /advmarkers.html
---

{% include toc %}

# TONY.AdvancedMarkers

Markers derived from the statistical and geometric properties of neural representations. So far two different markers have been implemented for this module. More are expected to be implemented in the future!

```python
from TONY.AdvancedMarkers import PerplexityExtractor, IntrinsicDimensionFinder
```

---

## `PerplexityExtractor`

Perplexity of a text under a causal language model — the exponential of the mean per-token cross-entropy. Low values mean the model finds the text predictable, high values atypical.

```python
PerplexityExtractor(model_name, device=None, backend="auto")
```

| Parameter | Default | Description |
|---|---|---|
| `model_name` | — | HuggingFace model name or local path. |
| `device` | `None` | `"cuda"` or `"cpu"`; auto-detected. Ignored on MLX. |
| `backend` | `"auto"` | `"mlx"`, `"hf"` (`"huggingface"` / `"transformers"`), or `"auto"`. |

With `"auto"`, MLX is selected when `model_name` contains `"mlx"`; otherwise HuggingFace. An explicit value always wins.

### Methods

- **`compute_perplexity(text) -> float`** — single text. Input is cast with `str()`. On the HF backend the text is truncated to the context window; on MLX, texts shorter than two tokens return `inf`.
- **`compute_perplexity_batch(texts, batch_size=8) -> list[float]`** — same order as input. `batch_size` only chunks the loop: texts are still scored one at a time.

```python
ppl = PerplexityExtractor("gpt2")
ppl.compute_perplexity("Some days I keep living, even though I feel completely alone")

ppl_mlx = PerplexityExtractor("mlx-community/Mistral-7B-Instruct-v0.3-4bit")  # MLX auto-detected
scores = ppl.compute_perplexity_batch(df["text"].tolist())
```

Requires `torch` + `transformers` (HF) or `mlx-lm` (MLX); imports are lazy, so only the stack in use needs installing.

---

## `IntrinsicDimensionFinder`

Intrinsic dimension (ID) of a set of representations — the effective number of degrees of freedom of the manifold they lie on, estimated with ABIDE (binomial estimator + kstar scale selection). Encoding is handled by the class.

```python
IntrinsicDimensionFinder(
    backend="sentence_transformers", model_name=None, api_key=None,
    n_iter=10, initial_id=None, Dthr=6.67, batch_size=64, verbose=False,
)
```

| Parameter | Default | Description |
|---|---|---|
| `backend` | `"sentence_transformers"` | Also `"openrouter"` or `"precomputed"`. Anything else raises `ValueError`. |
| `model_name` | backend-dependent | `all-MiniLM-L6-v2` (ST), `openai/text-embedding-3-small` (OpenRouter), empty (precomputed). |
| `api_key` | `None` | OpenRouter key; falls back to `OPENROUTER_API_KEY`. |
| `n_iter` | `10` | ABIDE iterations. |
| `initial_id` | `None` | Starting ID; `None` uses the 2NN estimator. |
| `Dthr` | `6.67` | Threshold for the kstar test. |
| `batch_size` | `64` | OpenRouter request batching only. |
| `verbose` | `False` | Print per-iteration ID. |

### Methods

- **`fit(representations, Dthr=None) -> IntrinsicDimensionResult`** — takes a list of strings, or an `(N, D)` array with `backend="precomputed"`. `Dthr` overrides the instance value for this call only. Passing an array under a non-precomputed backend warns and treats it as embeddings; passing strings under `"precomputed"` raises `TypeError`.
- **`fit_multiple(groups, Dthr=None) -> list[IntrinsicDimensionResult]`** — runs `fit` on each group independently (e.g. clinical vs. control).

### `IntrinsicDimensionResult`

| Attribute | Description |
|---|---|
| `id` | Final ID estimate. |
| `id_history` | ID at each iteration — check convergence here. |
| `kstars` | Per-point kstar (selected local neighbourhood scale) at the last iteration. |
| `embeddings` | Matrix fed to ABIDE. |
| `model_name`, `backend` | Encoder used. |
| `extra` | Metadata; currently the effective `Dthr`. |

```python
idf = IntrinsicDimensionFinder()
res = idf.fit(texts)
print(res.id, res.id_history)

idf = IntrinsicDimensionFinder(backend="precomputed")
res = idf.fit(embedding_matrix)
```

### Notes

- ID is a property of the *set*, not of a single document; estimates are unstable with only a few dozen points, and cost grows quickly with `N` (full distance matrix via `dadapy`).
- Lower `Dthr` selects smaller neighbourhoods, i.e. more local scales.
- `initial_id` is passed to `ABIDE.__init__` but never read by `return_ids_kstar_binomial`, so estimation always starts from 2NN.

Requires `numpy`, `requests`, `dadapy`, `scipy`; `sentence-transformers` only for the local backend.
