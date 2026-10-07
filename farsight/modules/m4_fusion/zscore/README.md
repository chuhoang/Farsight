# zscore — fusion v1 (no training)
Per-modality cosine (NaN if probe or gallery lacks the modality) -> z-score with validation impostor
mean/std -> `s = Σ w_m q_m z_m / Σ w_m q_m` over available modalities (q = probe quality) -> ranked list,
`is_known = top1 >= tau`. All-missing probe -> NaN scores, `is_known=False`.

Calibrate on validation (S = `cosine_scores(stack(probes), stack(gallery))`, shape P x G x 3):
```python
f = registry.build("zscore"); f.fit(S, q_ids, g_ids)
f.grid_search(S, quality, q_ids, g_ids, metric="rank1")   # any key of eval.metrics.evaluate
f.set_tau(S, quality, q_ids, g_ids, fpir=0.01); f.save()  # -> weights/m4_fusion/zscore/stats.json
```
Without stats.json: mu=0, sd=1, tau=-inf (closed-set behaviour). Gallery stacking is cached per list object.
Search on a 10k gallery (512/256/2048-d) ≈ 3–25 ms CPU.
