# quality_gate (M2)

Decides per track whether to run the restorer: `median(quality per crop) < q0`.
Quality = KP-RPE face feature norm on the original (padded) crops, computed by M3's face encoder
and injected into `m2_restore/module.py` as `quality_fn`. Empty/NaN-only input -> no restore.
`q0` in `config.yaml` is a placeholder until tuned on degraded face crops.
