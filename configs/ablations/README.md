# GeoFormer-Phase++ Ablations

Run with:

```bash
python train.py --config configs/ablations/<file>.yaml
```

Files:

- `A0_baseline_geometry.yaml`: Baseline geometry decoder (no curriculum, no MulCA, no hybrid TS, 4 candidates).
- `A1_curriculum_robust.yaml`: A0 + robust validation PESQ settings + curriculum schedule.
- `A2_mulca.yaml`: A1 + MulCA block after dense encoder.
- `A3_hybrid_ts.yaml`: A1 + hybrid TS blocks.
- `A4_geom5_sparse.yaml`: A1 + 5 geometry candidates + sparse reliability regularization.
- `A5_full_stack.yaml`: Full GeoFormer-Phase++ stack (A1+A2+A3+A4).
- `A6_lite.yaml`: Parameter-reduced full stack (`dense_channel=48`, `num_tsblocks=3`, `dense_depth=3`).

Each file has its own `checkpoint_path` to avoid overwrite.
