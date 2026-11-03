# Cavity Flow Surrogate

A single-page demo of a neural-network surrogate for 2D lid-driven cavity flow.
The page evaluates a small MLP, (x, y, Re) → (u, v, p), directly in the browser at 50 × 50 cell centres,
so the flow field updates instantly as the Reynolds number changes (20–300). No CFD is solved and no server is needed.

The page also compares the network with a CFD reference on held-out Reynolds numbers and explains
where the surrogate is reliable (vortex position and strength) and where it is not (lid corners,
mass conservation, extrapolation outside the training range).

## View

- Online: GitHub Pages serves `index.html` from this repository.
- Offline: download `index.html` and open it in any modern browser.

## Contents

| File | Purpose |
|---|---|
| `index.html` | Self-contained page: model weights, normalization, CFD test fields, and all drawing code are embedded. |
| `.nojekyll` | Serves the file as-is on GitHub Pages. |

## Model and data

- Network: fully connected, 2 hidden layers × 32 nodes, tanh activation, linear output; trained with Adam on MSE.
- Training data: 40 steady cavity solutions, Re 20–300; 10 further cases held out for testing.
- CFD reference: finite-difference projection solver on a 50 × 50 staggered (MAC) grid. It is a converged
  coarse-grid reference, not a grid-independent benchmark.
- The network fits solver data only; it does not enforce the governing equations (not a PINN).

## Acknowledgements

The CFD workflow is adapted from [CFD Python](https://github.com/barbagroup/CFDPython)
(Barba & Forsyth, Step 11), © Barba group, used under its CC-BY 3.0 / BSD-3-Clause licenses.
Benchmark comparison values: Ghia, Ghia & Shin (1982), *J. Comput. Phys.* 48, 387–411.
