# Cavity Flow Surrogate

By Yeongbeen Jang (장영빈)

A single-page demo of a neural-network surrogate for 2D lid-driven cavity flow.
The page evaluates a small MLP, (x, y, Re) → (u, v, p), directly in the browser at 50 × 50 cell centres,
so the flow field updates instantly as the Reynolds number changes. No CFD is solved in the browser and no server is needed.

**Live site:** https://yeongbeenjang.github.io/cavity-flow-surrogate/

## What the page shows

| Tab | Purpose |
|---|---|
| Any Re · NN | Network prediction for any Re from 20 to 1000; a warning appears above Re 300 (outside the training range). |
| Test cases · vs CFD | The 10 held-out cases inside the training range: CFD, network and error side by side. |
| Re 300–1000 · vs CFD | Extrapolation check: CFD reference and network prediction at the same Re (300, 350, …, 1000), same quantity, one shared colour scale, plus the error map. |

Below the instrument, six short sections explain the physics, what the network reproduces, where it fails
inside the range (lid corners, mass conservation), and how it behaves outside the range.

## Extrapolation check

- The trained model is used unchanged; it was trained only on Re 20–300.
- 15 extra CFD cases (Re 300–1000, step 50) were solved with the same solver as the training data, for comparison only.
  All met the original acceptance limits (steady residual < 2e-5, divergence RMS < 1e-8), and the new Re 300 field is
  bit-identical to the training-set Re 300 field.
- Main result: outside the range the network extends the trend it saw near Re 300, so its vortex centre drifts into the
  right wall while the CFD centre turns back toward the middle; velocity RMSE grows from about 0.02 at Re 300 to 0.14–0.19 at Re 1000.

## Rebuild

Requires Python with NumPy, SciPy and Numba.

```bash
python tools/cavity_cfd.py     # solve the Re 300-1000 reference cases -> data/cfd_extrapolation.npz
python tools/build_site.py     # embed model + CFD data into index.html
```

| Path | Content |
|---|---|
| `index.html` | Generated, self-contained page (model weights, CFD fields and drawing code embedded). |
| `data/model.json` | Exported network: layers, weights, input/output normalization, verification points. |
| `data/cfd_re20_300.npz` | The 50 training-range CFD cases (40 train, 10 test). |
| `data/cfd_extrapolation.npz` | The 15 extrapolation reference cases and their convergence diagnostics. |
| `tools/` | CFD solver, page template and build script. |

## Model and data

- Network: fully connected, 2 hidden layers × 32 nodes, tanh activation, linear output; trained with Adam on MSE.
- CFD reference: finite-difference projection solver on a 50 × 50 staggered (MAC) grid with first-order upwinding.
  It is a converged coarse-grid reference, not a grid-independent benchmark; at high Re it is noticeably diffusive.
- The network fits solver data only; it does not enforce the governing equations (not a PINN).

## Acknowledgements

The CFD workflow is adapted from [CFD Python](https://github.com/barbagroup/CFDPython)
(Barba & Forsyth, Step 11), © Barba group, used under its CC-BY 3.0 / BSD-3-Clause licenses.
Benchmark values: Ghia, Ghia & Shin (1982), *J. Comput. Phys.* 48, 387–411.
Typeface: [Pretendard](https://github.com/orioncactus/pretendard) (SIL Open Font License).
