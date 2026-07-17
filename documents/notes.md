**`float32`** is IEEE 754 single precision: 1 sign bit + 8 exponent bits + 23 mantissa bits.

## Range (approximate)

| | float32 |
|---|---|
| Max finite | **~3.4 × 10³⁸** |
| Min normal positive | **~1.2 × 10⁻³⁸** |
| Min subnormal | **~1.4 × 10⁻⁴⁵** |
| Negative side | same magnitudes, flipped sign |

So roughly **±10⁻³⁸ to ±10³⁸**, plus `0`, `inf`, `-inf`, `NaN`.

## Precision (more important in ML)

float32 has ~**7 decimal digits** of precision.

- **Machine epsilon:** ~`1.19e-7` (smallest gap at 1.0)
- Integers **exactly** representable: up to **2²⁴ = 16,777,216**
- After that, `16777216 + 1 == 16777216` (rounding)

## Examples

```python
import numpy as np

np.finfo(np.float32).max    # ~3.4028235e+38
np.finfo(np.float32).min    # ~1.1754944e-38  (smallest normal)
np.finfo(np.float32).eps    # ~1.1920929e-07
```

## vs `float64` (Python default / NumPy default)

| | float32 | float64 |
|---|---|---|
| Bits | 32 | 64 |
| Decimal precision | ~7 digits | ~15 digits |
| Max | ~3.4e38 | ~1.8e308 |

## ML takeaway

For your notebook (TF-IDF weights ~0–1, logits, gradients): **float32 is fine**.

Problems show up with:
- very small/large numbers mixed in one computation
- long sums of tiny values
- needing exact integers > 16M

That is why training sometimes uses **fp64 for loss accumulation** or **fp16/bf16** on GPU for speed, with different tradeoffs.

**Short answer:** The `38` comes from converting powers of 2 (allowed by the 8 exponent bits) into decimal scientific notation.  
`2^127 ≈ 3.4×10^38` and `2^-126 ≈ 1.2×10^-38`.

---

## float32 bit layout (32 bits)

```
[ sign: 1 bit ][ exponent: 8 bits ][ mantissa: 23 bits ]
```

IEEE 754 formula for **normalized** numbers:

\[
\text{value} = (-1)^{sign} \times (1 + \frac{f}{2^{23}}) \times 2^{(e - 127)}
\]

- `e` = stored exponent (8 bits, 0–255)
- `f` = mantissa bits (0–2²³−1)
- **127** = bias (fixed for float32)

---

## Step 1: What exponents are allowed?

8 bits → 256 patterns, but not all mean normal numbers:

| Stored `e` | Meaning |
|---|---|
| 0 | zero / subnormal |
| 1–254 | **normal numbers** |
| 255 | inf / NaN |

For normals: actual power = `e - 127`

- Min normal: `e=1` → `2^(1-127) = 2^-126`
- Max normal: `e=254` → `2^(254-127) = 2^127`

So the **binary** range is roughly **2^-126 to 2^127** (before mantissa scaling).

---

## Step 2: Why that becomes 10^-38 to 10^38

Use: `log10(2) ≈ 0.301`

**Max (rough):**
```
2^127 → log10 = 127 × 0.301 ≈ 38.2  →  ~10^38
```
With max mantissa (~2): **~3.4 × 10^38**

**Min normal:**
```
2^-126 → log10 = -126 × 0.301 ≈ -37.9  →  ~10^-38
```
Exact: **1.175 × 10^-38**

That is where **e+38** and **e-38** come from — decimal form of those powers of 2.

---

## Step 3: Worked max example

Max normal float32:
- `e = 254` → exponent factor `2^127`
- mantissa all 1s → factor ≈ `2 - 2^-23`

```
(2 - 2^-23) × 2^127 ≈ 3.4028235 × 10^38
```

---

## Mental shortcut

| float32 | Exponent bits | Binary power range | Decimal order |
|---|---|---|---|
| | 8 (bias 127) | ~2^-126 … 2^127 | ~10^-38 … 10^38 |

Rule: **multiply exponent by ~0.3** to get the decimal order of magnitude.

```
127 × 0.3 ≈ 38   → 10^38
-126 × 0.3 ≈ -38 → 10^-38
```

Same idea for **float64** (11 exponent bits, bias 1023): range ~**10^-308 … 10^308**, because `2^1023 ≈ 10^308`.

---

**IEEE 754** is the standard for how computers store real numbers in binary. PyTorch `float32`, Python/NumPy `float64`, and GPU `float16`/`bfloat16` all follow it (with minor variants).

## Core idea

A float is **not** stored as exact decimal. It is stored as:

\[
\text{value} = (-1)^{sign} \times significand \times 2^{exponent}
\]

Like scientific notation, but **base 2**.

Example (conceptual):  
`−1 × 1.625 × 2^3 = −13.0`

---

## Bit layout (float32 = 32 bits)

```
| S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM |
 1     8 bits            23 bits
sign  exponent          mantissa (fraction)
```

| Part | Role |
|---|---|
| **Sign (S)** | 0 = positive, 1 = negative |
| **Exponent (E)** | stored with **bias** (127 for float32) |
| **Mantissa (M)** | fractional part; normalized numbers have implicit leading **1.** |

**Formula (normalized):**

\[
(-1)^S \times (1 + M/2^{23}) \times 2^{(E - 127)}
\]

---

## Why bias the exponent?

Exponent is stored as unsigned integer with offset so negative powers are easy to encode.

- Stored `E=127` → actual exponent `0` → × `2^0`
- Stored `E=128` → actual exponent `1` → × `2^1`
- Stored `E=126` → actual exponent `−1` → × `2^-1`

float32 bias = **127**  
float64 bias = **1023**

---

## Special exponent values

Not every bit pattern is a normal number.

| Stored E | Mantissa M | Meaning |
|---|---|---|
| 0 | 0 | **±0** |
| 0 | ≠0 | **subnormal** (very tiny, no implicit 1) |
| 1–254 | anything | **normal** numbers |
| 255 | 0 | **±infinity** |
| 255 | ≠0 | **NaN** (Not a Number) |

That is why you get `inf` from overflow and `nan` from `0/0`.

---

## Normal vs subnormal

**Normal:** mantissa is `1.xxxxx` (implicit 1)  
→ min ≈ `2^-126 ≈ 1.2e-38`

**Subnormal:** mantissa is `0.xxxxx` (no implicit 1)  
→ can go smaller, down to ~`1.4e-45`, but with less precision

---

## Precision, not just range

float32 has **23 mantissa bits** → ~**7 decimal digits** of precision.

- Numbers up to **2^24 = 16,777,216** are integers exactly representable
- Many decimals (e.g. `0.1`) are **not** exact in binary → rounding error

```python
0.1 + 0.2 == 0.3          # False in Python float (float64)
np.float32(0.1) + np.float32(0.2) == np.float32(0.3)  # also False
```

---

## Common formats (IEEE 754 family)

| Name | Bits | Exponent | Mantissa | Bias | Typical range |
|---|---|---|---|---|---|
| float16 | 16 | 5 | 10 | 15 | ~6e-8 … 6e4 |
| bfloat16 | 16 | 8 | 7 | 127 | ~1e-38 … 1e38 (same exp as fp32) |
| **float32** | **32** | **8** | **23** | **127** | **~1e-38 … 3e38** |
| float64 | 64 | 11 | 52 | 1023 | ~1e-308 … 1e308 |

---

## Tiny worked example (float32)

Store **−13.0**:

1. Binary: `−1101.0` = `−1.101 × 2^3`
2. Sign = 1
3. Exponent: `3 + 127 = 130` → `10000010`
4. Mantissa: `101` + zeros → `1010000...`

Bits (conceptually):  
`1 | 10000010 | 10100000000000000000000`

---

## ML-relevant implications

1. **Limited precision** → gradients can underflow/overflow
2. **Not associative:** `(a+b)+c ≠ a+(b+c)` in float math
3. **Mixed precision** (fp16/bf16 training) trades speed/memory for narrower mantissa
4. **Your notebook:** weights in fp32, TF-IDF in float32 — totally fine; no need for float64 here

---

## One-line summary

**IEEE 754 = sign + biased exponent + mantissa**, encoding `sign × 1.fraction × 2^power` in binary — fast on hardware, but approximate and range-limited.

---

**Interviewer-style answer:**

Floating point is stored as **sign + exponent + mantissa** because we need numbers that span a huge range **and** run fast on hardware — and fixed-width bits force a tradeoff.

## The problem we're solving

ML and scientific code need values like:
- `0.000001` (small gradient)
- `3.4` (weight)
- `1e9` (large accumulator)

You can't use **fixed-point** (one decimal/scaling position) — either small numbers vanish or large ones overflow.

You can't store **exact decimals** in 32 bits — most rationals need infinite digits in binary (`0.1` is the classic example).

So the standard picks: **approximate value + wide dynamic range**, in a fixed bit budget.

## Why sign / exponent / mantissa specifically

Think **scientific notation in base 2**:

\[
\text{number} = \text{sign} \times \text{significand} \times 2^{\text{exponent}}
\]

Each piece has a job:

**Sign (1 bit)**  
Cheapest way to handle negative weights, gradients, logits. Separate from magnitude.

**Exponent (8 bits in float32)**  
Controls **order of magnitude**. Moving the binary point left/right lets one format cover ~10⁻³⁸ to 10³⁸. That's why the same 32 bits can represent both tiny TF-IDF weights and large intermediate activations.

**Mantissa (23 bits)**  
Controls **precision within that magnitude**. More bits → finer resolution; fewer → coarser. This is where the ~7 decimal digits of float32 come from.

Splitting bits this way is deliberate: **exponent buys range, mantissa buys accuracy**.

## Why base 2 (not base 10)

Computers are binary. Multiplication/division by powers of 2 is a **bit shift** — essentially free on CPU/GPU. IEEE 754 maps cleanly to adders and multipliers in silicon, which is why every GPU is built around it.

## Why biased exponent

We need **negative** exponents (for small numbers) but want simple integer comparison hardware. Store `exponent + 127` so the bit pattern sorts roughly by magnitude. Engineering convenience, not math elegance.

## Why subnormals, inf, NaN

Real numerics hits edge cases:
- **Subnormals:** smooth the gap down to zero (gradients near underflow)
- **Inf:** overflow without crashing (`softmax`, `exp`)
- **NaN:** propagate invalid ops (`0/0`) instead of silent garbage

Interview line: *"The format isn't just for storing reals — it's for stable, predictable behavior at the edges."*

## The tradeoff you'd mention unprompted

IEEE 754 is a **compression scheme**:
- Fixed 32 bits per number
- Huge range
- Fast hardware
- **Cost:** rounding error, non-associativity, not all decimals exact

That's acceptable for neural nets because we already tolerate noise — SGD, dropout, approximate optimizers. We pick dtype (`fp32`, `fp16`, `bf16`) based on whether we need more **range** or more **precision** for that tensor.

## Tie to your notebook (if they push)

Your model weights (~2.6M × fp32 ≈ 9.8 MB) use this format because:
- weights don't need exact fractions
- we need a wide enough range for logits and gradients
- GPUs are optimized for fp32/fp16 linear algebra

**One-liner to close:**  
*"It's scientific notation in binary, packed into fixed bits so hardware can do fast approximate arithmetic across many orders of magnitude — which is exactly what deep learning needs."*

---

# TF-IDF

`TfidfVectorizer` turns each review into a fixed-length numeric vector so your MLP can consume it. It is a bag-of-words representation with term weighting, not a learned embedding.

## Pipeline (what actually happens)

1. **Tokenize + build vocabulary** (`fit` on train only)
   - Lowercase, split on whitespace/punctuation, drop very common/rare tokens (sklearn defaults).
   - Keep top **10,000** terms by document frequency (`max_features=10_000`).
   - Each dimension = one word/n-gram in that vocab.

2. **Term frequency (TF)** per document
   - Count how often each vocab term appears in the review.
   - With `sublinear_tf=True`: use `1 + log(tf)` instead of raw count, so repeating "good" 10x does not dominate as much.

3. **Inverse document frequency (IDF)** across corpus
   - Rare terms get higher weight; terms in almost every review get downweighted.
   - Standard sklearn form:  
     `idf(t) = log((1 + n_docs) / (1 + df(t))) + 1`
   - `df(t)` = number of docs containing term `t`.

4. **TF-IDF weight**
   - `weight(t, doc) = tf(t, doc) * idf(t)`
   - High when the term is frequent in this doc but uncommon globally.

5. **Output**
   - Sparse matrix `(n_docs, 10_000)`.
   - Your notebook densifies it: `.toarray().astype(np.float32)` → dense `(25000, 10000)`.

## Intuition

- **"brilliant"** in one positive review → high TF, moderate IDF → strong signal.
- **"the"** everywhere → high TF, very low IDF → near zero after weighting.
- You lose word order, negation structure, and semantics beyond co-occurrence stats.

## In your notebook specifically

```python
vectorizer = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
X_train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
X_test  = vectorizer.transform(test_texts).toarray().astype(np.float32)
```

- **`fit_transform` on train**: learns vocab + IDF from train only (correct).
- **`transform` on test**: maps test words to the same 10k features; OOV words are ignored.
- Result: each review is a 10k-dim feature vector fed to `ReviewDataset`.

## MLE-relevant tradeoffs

| Pros | Cons |
|---|---|
| Fast, interpretable baseline | No word order ("not good" ≈ "good not") |
| Works well for sentiment with linear-ish signal | Vocab fixed at train time |
| Cheap vs transformer tokenization | Dense `.toarray()` is ~1 GB RAM here |

This is a classic sparse text baseline. LLM pipelines replace this with subword token IDs + learned embeddings, but the training loop stays the same.

## Example

**Tiny corpus (3 reviews, 6 unique words after tokenization):**

```python
docs = [
    "good movie",           # doc 0, label positive
    "bad movie",            # doc 1, label negative
    "good good movie",      # doc 2, label positive
]
```

Assume vocab = `["bad", "good", "movie"]` → 3 features.

---

### Step 1: Raw counts (TF before weighting)

| doc | bad | good | movie |
|-----|-----|------|-------|
| 0   | 0   | 1    | 1     |
| 1   | 1   | 0    | 1     |
| 2   | 0   | 2    | 1     |

---

### Step 2: Document frequency (how many docs contain each word)

| term  | docs containing it | df |
|-------|--------------------|----|
| bad   | 1                  | 1  |
| good  | 2                  | 2  |
| movie | 3                  | 3  |

`movie` is in every doc → low discriminative power.  
`bad` appears in only one doc → high IDF.

---

### Step 3: IDF (sklearn-style, simplified)

Roughly:

- `idf(bad)`   ≈ high   (rare)
- `idf(good)`  ≈ medium
- `idf(movie)` ≈ low    (appears everywhere)

---

### Step 4: TF-IDF vectors (conceptual)

For doc 0 `"good movie"`:

- `good`: moderate (1× in doc, seen in 2/3 docs)
- `movie`: small (1× in doc, but in all docs)
- `bad`: 0

Vector ≈ `[0.0, 0.5, 0.2]`  (numbers illustrative)

For doc 1 `"bad movie"`:

- `bad` gets a strong weight (rare + present)
- `movie` still small

Vector ≈ `[0.8, 0.0, 0.2]`

For doc 2 `"good good movie"`:

- `good` boosted by count (or `1+log(2)` with `sublinear_tf=True`)
- `movie` still small

Vector ≈ `[0.0, 0.9, 0.2]`

---

### What the model sees

Each review → fixed 3-dim vector. The MLP learns weights like:

- high `bad` → negative
- high `good` → positive
- high `movie` alone → weak signal (common word)

---

### With sklearn (same tiny example)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

docs = ["good movie", "bad movie", "good good movie"]

vec = TfidfVectorizer()  # no max_features limit here
X = vec.fit_transform(docs).toarray()

print("vocab:", vec.get_feature_names_out())
print(pd.DataFrame(X, columns=vec.get_feature_names_out()))
```

Typical output shape:

```
vocab: ['bad' 'good' 'movie']

        bad      good     movie
0  0.000000  0.707107  0.707107   # "good movie"
1  0.894427  0.000000  0.447214   # "bad movie"
2  0.000000  0.894427  0.447214   # "good good movie"
```

Key takeaway: same vocab size for every doc; weights encode “important in this doc, rare in corpus.” Word order is ignored — `"good movie"` and `"movie good"` would be identical.

---

# LoRA

Here's where **887,042 trainable params** comes from — it splits into two buckets.

## 1. LoRA adapters (294,912 params)

DistilBERT has **6 layers**. You attach LoRA to **`q_lin`** and **`v_lin`** in each layer (`r=16`, hidden dim `768`).

Per attention module (q or v):

| Matrix | Shape | Params |
|--------|-------|--------|
| `lora_A` | `[16, 768]` | 12,288 |
| `lora_B` | `[768, 16]` | 12,288 |
| **Subtotal** | | **24,576** |

Per layer (q + v): `24,576 × 2 = 49,152`  
All 6 layers: `49,152 × 6 = **294,912**`

Formula per module: `r × d + d × r = 2rd = 2 × 16 × 768 = 24,576`

---

## 2. Classification head (592,130 params)

PEFT auto-trains the task head for `TaskType.SEQ_CLS` (`modules_to_save`):

| Layer | Shape | Params |
|-------|-------|--------|
| `pre_classifier.weight` | `[768, 768]` | 589,824 |
| `pre_classifier.bias` | `[768]` | 768 |
| `classifier.weight` | `[2, 768]` | 1,536 |
| `classifier.bias` | `[2]` | 2 |
| **Subtotal** | | **592,130** |

These were **randomly initialized** for SST-2 (not from pretrained weights), so they must be trained.

---

## Total

```
294,912  (LoRA)
+ 592,130  (classifier head)
= 887,042  trainable params
```

Trainable %: `887,042 / 67,842,052 = 1.31%` — matches your output.

---

## What is frozen?

Everything else in DistilBERT (~67M params): embeddings, all frozen `q_lin`/`v_lin`/`k_lin`/`o_lin` base weights, FFN layers, layer norms, etc.

---

## Config knobs in your setup

| Setting | Effect here |
|---------|-------------|
| `r=16` | LoRA rank → width of A/B |
| `lora_alpha=32` | scaling = `32/16 = 2.0` on the LoRA update |
| `target_modules=["q_lin", "v_lin"]` | 2 modules × 6 layers = 12 LoRA pairs |
| `bias="none"` | no bias training in LoRA layers |
| `TaskType.SEQ_CLS` | classifier head stays trainable |

---

## Memory impact (training, fp32 + Adam)

Only trainable params need optimizer state:

```
887,042 × 16 bytes ≈ 13.5 MB   (LoRA + head training overhead)
```

Base model ~68M params still sit in memory for forward pass (~260 MB fp32 weights), but you are **not** updating all of them — that is the LoRA win.

If you want, I can walk through what happens if you change `r` to 8 or add `k_lin`/`o_lin` to `target_modules`.

---

# Traing

## Base case for studying

**Goal:** learn the full pretraining loop (data → tokenizer → train → eval), not beat GPT-4.

### Recommended study ladder

| Stage | Model | Tokens | Why |
|---|---|---|---|
| **0. Pipeline check** | TinyStories **~33M** | 0.1–1B | Catch bugs cheaply |
| **1. Base case (do this)** | **SmolLM2-135M** or **Pythia-160M** | **2–10B** | Real modern small LLM, still affordable |
| **2. Stretch** | SmolLM2-360M / Pythia-410M | 10–20B | Only after stage 1 works |

**Base case = Stage 1:** ~150M params, ~5B tokens of FineWeb-Edu (or your own cleaned data).

That’s enough to see:
- loss going down
- coherent text generation
- scaling effects (batch size, LR, context length)
- checkpointing / resume / eval

---

## Cost to rent GPUs (ballpark, 2026 cloud rates)

Typical prices:
- **T4 / L4:** ~\$0.30–0.60 / hr  
- **A10 / RTX 4090:** ~\$0.40–0.80 / hr  
- **A100 40/80GB:** ~\$1.20–2.50 / hr  
- **H100:** ~\$2.50–4.50 / hr  

### Stage 0 — TinyStories 33M

| GPU | Time | Cost |
|---|---|---|
| 1× T4 / 4090 | 2–8 hours | **\$2–\$10** |

### Stage 1 — Base case (~135–160M, 5B tokens)

| GPU | Time | Cost |
|---|---|---|
| 1× A100 | ~1–2 days | **\$40–\$120** |
| 1× 4090 / A10 | ~2–4 days | **\$40–\$150** |
| 4× A100 | ~6–12 hours | **\$50–\$150** |

**Budget for the study base case: ~\$50–\$150.**

### Stage 2 — 360–410M, 10–20B tokens

| GPU | Time | Cost |
|---|---|---|
| 1× A100 | ~1–2 weeks | **\$200–\$800** |
| 4–8× A100 | ~2–4 days | **\$300–\$1,000** |

### What *not* to start with

| Target | Typical cost from scratch |
|---|---|
| GPT-2 1.5B | **\$5k–\$50k+** |
| 7B class | **\$50k–\$500k+** |

Too expensive and too slow for learning.

---

## Concrete “base case” I’d run

```text
Model:     SmolLM2-135M architecture (or Pythia-160M)
Data:      FineWeb-Edu sample, ~5B tokens
Context:   1024 or 2048
GPU:       1× A100 40/80GB on RunPod/Vast/Lambda
Time:      ~24–48 hours
Cost:      ~$50–$120
Success:   train loss steadily down; samples look like English
```

---

## Study checklist (what “done” means)

1. Train for at least 1–2B tokens without crashing  
2. Plot loss vs tokens  
3. Generate text from checkpoints (early vs late)  
4. Measure tokens/sec and \$/1B tokens  
5. Then decide whether to scale model or data

**Bottom line:** for studying, the base case is a **~150M modern LM on ~5B tokens for about \$50–\$150**. Start there; don’t rent a cluster for a 1B+ from-scratch run until that works.