# SEPIN discrepancy analysis

Observed: computed values are 3-10× lower than paper results.

---

## Cause 1 — Mini-batch NCE vs. full-dataset NCE (most likely)

### How it works now
`_batched_nce` splits N examples into chunks of `nce_batch_size=256`, computes SimCLR loss
independently per chunk, and averages. Each chunk has B=256 examples → 2B=512 vectors →
each example only sees 510 negatives.

### Why it shrinks SEPIN
The InfoNCE bound is `I(x;y) >= log(N) - L_NCE`. For a well-trained model:
- With B=256 negatives, the positive pair is trivially ranked #1 → L_full ≈ 0
- The LOO representation is also good enough → L_loo ≈ 0
- Delta = L_loo - L_full collapses near zero even though the true delta is large

With the full training set as negatives (N=50K for CIFAR-100, N=130K for ImageNet-100),
the task is genuinely hard — the positive pair must be identified among tens of thousands of
other examples — so L_full and L_loo are both larger and their difference is meaningful.

**Fix**: Use `_full_nce` (see [main_eval2.py](main_eval2.py)) which computes each example's
loss against ALL 2N examples using chunked matrix multiplication.

---

## Cause 2 — Normalization before vs. after slicing

### Current behaviour
Unnormalized z is sliced to (D-1) dims, then normalized to unit norm in (D-1)-dim space.

```python
z = F.normalize(z, dim=-1)   # normalizes the (D-1)-dim slice
```

### Paper's possible behaviour
Normalize the full D-dim vector first, then slice (without re-normalizing):

```python
z_norm = F.normalize(z, dim=-1)   # unit norm in D-dim space
z_loo  = z_norm[:, keep]          # NOT unit norm, has norm < 1
```

The geometric meaning differs:
- **Current**: remaining D-1 features are re-weighted relative to each other;
  feature i's magnitude contribution is redistributed among them.
- **Alternative**: each feature's weight in the full representation is preserved;
  removing feature i leaves a gap (||z_loo|| < 1), directly reducing alignment.

The alternative tends to produce larger deltas for informative features because the
"missing energy" from feature i is not compensated.

---

## Cause 3 — SEPIN@k = sum vs. mean

### Current behaviour
```python
results[f"sepin_{label}"] = float(deltas_ranked[:k].mean().item())
```

If the paper reports the **sum** of the top-k deltas, SEPIN@10 would be 10× larger and
SEPIN@100 would be 100× larger than the current values, while SEPIN@1 would match.

**Diagnostic**: Check if SEPIN@1 matches the paper. If it does, the issue is sum vs. mean.

---

## Cause 4 — Distributed gather in simclr_loss_func

`simclr_loss_func` calls `gather(z)` which in distributed mode collects from all GPUs.
If the paper was evaluated on 8 GPUs with nce_batch_size=256, the effective batch per NCE call
is 8×256=2048. Single-GPU evaluation with B=256 has 8× fewer negatives and systematically
smaller deltas.

---

## Cause 5 — Temperature sensitivity

SEPIN deltas scale approximately as 1/temperature. If the paper used a different evaluation
temperature than the training temperature, deltas shift proportionally.

The training temperature (stored in args.json as `method_kwargs.temperature`) is reused in
`main_disent.py`. Confirm the paper uses the same temperature for evaluation.

---

## Summary table

| Cause | Expected effect | Easy to test? |
|-------|----------------|---------------|
| Mini-batch NCE vs. full-dataset | 3–10× on all k | Yes — use `use_full_nce=True` |
| Normalization order | Hard to predict, varies by feature | Medium |
| Sum vs. mean SEPIN@k | k× on SEPIN@k, SEPIN@1 unchanged | Yes — compare SEPIN@1 with paper |
| Distributed gather | ~num_gpus× | Yes — check paper GPU count |
| Temperature mismatch | Uniform scaling | Yes — read paper appendix |
