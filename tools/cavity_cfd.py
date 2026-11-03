"""Lid-driven cavity CFD reference solver and extrapolation-case runner.

The solver is the Week 9 notebook's finite-difference pressure-projection scheme
(staggered MAC grid, sign-aware upwinding, sparse pressure solve), itself adapted from
CFD Python Step 11 (Barba & Forsyth, (c) Barba group, CC-BY 3.0 / BSD-3-Clause).
Only `max_steps` is raised for the extrapolation cases, which need longer to reach steady state.

Usage (from the repository root):
    python tools/cavity_cfd.py
writes data/cfd_extrapolation.npz with fields for Re = 300, 350, ..., 1000.
These fields are references for checking the network; they are never used for training.
"""
import logging
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
N_GRID = 50
EXTRAPOLATION_RE = np.arange(300, 1001, 50, dtype=float)
STEADY_TOL = 2e-5          # max |du|/dt accepted as steady (same as the notebook)
DIV_TOL = 1e-8             # discrete divergence RMS limit (same as the notebook)
MAX_STEPS = 200_000        # notebook used 20,000; higher Re needs longer to settle


def pressure_factor(n):
    """Factorize the pressure-Poisson matrix (Neumann walls, first cell pinned).

    Args:
        n: cells per side.
    Returns:
        SuperLU factor with a .solve(rhs) method.
    """
    h = 1.0 / n
    A = lil_matrix((n * n, n * n))
    for j in range(n):
        for i in range(n):
            k = j * n + i
            for jj, ii in [(j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)]:
                if 0 <= jj < n and 0 <= ii < n:
                    A[k, jj * n + ii] = 1 / h**2
                    A[k, k] -= 1 / h**2
    A[0, :] = 0
    A[0, 0] = 1
    return splu(A.tocsc())


@njit
def tentative_velocity(u, v, nu, dt, h):
    """Explicit upwind advection + diffusion predictor on the MAC grid.

    Args:
        u: x-face velocity, shape (n, n+1).
        v: y-face velocity, shape (n+1, n).
        nu: kinematic viscosity, 1/Re.
        dt: time step.
        h: cell size.
    Returns:
        (us, vs) face velocities before the pressure projection.
    """
    n = u.shape[0]; us = u.copy(); vs = v.copy()
    for j in range(n):
        for i in range(1, n):
            uc = u[j, i]
            vc = .25 * (v[j, i - 1] + v[j, i] + v[j + 1, i - 1] + v[j + 1, i])
            below = u[j - 1, i] if j > 0 else -uc
            above = u[j + 1, i] if j < n - 1 else 2.0 - uc
            dx = (uc - u[j, i - 1]) / h if uc >= 0 else (u[j, i + 1] - uc) / h
            dy = (uc - below) / h if vc >= 0 else (above - uc) / h
            lap = (u[j, i - 1] + u[j, i + 1] + below + above - 4 * uc) / (h * h)
            us[j, i] = uc + dt * (-uc * dx - vc * dy + nu * lap)
    for j in range(1, n):
        for i in range(n):
            vc = v[j, i]
            uc = .25 * (u[j - 1, i] + u[j - 1, i + 1] + u[j, i] + u[j, i + 1])
            left = v[j, i - 1] if i > 0 else -vc
            right = v[j, i + 1] if i < n - 1 else -vc
            dx = (vc - left) / h if uc >= 0 else (right - vc) / h
            dy = (vc - v[j - 1, i]) / h if vc >= 0 else (v[j + 1, i] - vc) / h
            lap = (left + right + v[j - 1, i] + v[j + 1, i] - 4 * vc) / (h * h)
            vs[j, i] = vc + dt * (-uc * dx - vc * dy + nu * lap)
    return us, vs


def solve_cavity(reynolds, n=N_GRID, max_steps=MAX_STEPS, steady_tol=STEADY_TOL, factor=None):
    """March the cavity to steady state with pressure projection.

    Args:
        reynolds: Reynolds number (lid speed 1, cavity side 1).
        n: cells per side.
        max_steps: time-step cap.
        steady_tol: steady-state threshold on max |du|/dt.
        factor: optional prebuilt pressure factor.
    Returns:
        (field[n, n, 3] of cell-centred u, v, p; steps taken; final change; divergence RMS).
    """
    h = 1 / n; nu = 1 / reynolds; dt = min(.2 * h, .2 * h * h / nu)
    factor = pressure_factor(n) if factor is None else factor
    u = np.zeros((n, n + 1)); v = np.zeros((n + 1, n)); p = np.zeros((n, n))
    change = np.inf
    for step in range(max_steps):
        us, vs = tentative_velocity(u, v, nu, dt, h)
        rhs = ((us[:, 1:] - us[:, :-1] + vs[1:, :] - vs[:-1, :]) / h / dt).ravel()
        rhs -= rhs.mean(); rhs[0] = 0
        p = factor.solve(rhs).reshape(n, n)
        us[:, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / h
        vs[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / h
        change = max(np.max(np.abs(us - u)), np.max(np.abs(vs - v))) / dt
        u, v = us, vs
        if not np.isfinite(change) or np.max(np.abs(u)) > 10:
            raise ValueError(f'CFD unstable at Re={reynolds}: no field accepted.')
        if step > 100 and change < steady_tol:
            break
    div = (u[:, 1:] - u[:, :-1] + v[1:, :] - v[:-1, :]) / h
    p -= p.mean()
    field = np.stack(((u[:, 1:] + u[:, :-1]) / 2, (v[1:, :] + v[:-1, :]) / 2, p), axis=-1)
    return field, step + 1, change, float(np.sqrt(np.mean(div**2)))


def run_extrapolation_cases(out_path=ROOT / 'data' / 'cfd_extrapolation.npz'):
    """Solve every extrapolation Re, apply the notebook's acceptance checks, and save them.

    Args:
        out_path: destination .npz (fields, reynolds, diagnostics[steps, change, divergence]).
    Returns:
        Path of the written file.
    """
    factor = pressure_factor(N_GRID)
    fields, diagnostics = [], []
    for re in EXTRAPOLATION_RE:
        start = time.perf_counter()
        field, steps, change, div = solve_cavity(re, factor=factor)
        if change >= STEADY_TOL or div >= DIV_TOL:
            raise RuntimeError(f'Re={re:.0f} failed the acceptance checks '
                               f'(change {change:.2e}, divergence {div:.2e}).')
        fields.append(field); diagnostics.append((steps, change, div))
        log.info(f'Re={re:6.0f}: {steps:6d} steps, change {change:.2e}, div {div:.1e}, '
                 f'{time.perf_counter() - start:.1f} s')
    out_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(out_path, fields=np.array(fields), reynolds=EXTRAPOLATION_RE,
                        diagnostics=np.array(diagnostics))
    log.info(f'Wrote {out_path}')
    return out_path


if __name__ == '__main__':
    run_extrapolation_cases()
