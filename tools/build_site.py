"""Build index.html: embed the trained model, CFD reference fields and comparison statistics.

Usage (from the repository root):
    python tools/build_site.py
Inputs: data/model.json, data/cfd_re20_300.npz (notebook dataset), data/cfd_extrapolation.npz
(tools/cavity_cfd.py) and tools/template.html. Output: index.html, a single self-contained page.
"""
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = 'Yeongbeen Jang (장영빈)'
N = 50
H = 1 / N
CELL = (np.arange(N) + .5) / N
XX, YY = np.meshgrid(CELL, CELL)
TEST_IDS = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]          # the notebook's held-out cases
# Primary-vortex centres from Ghia, Ghia & Shin (1982), 129 x 129 / 257 x 257 grids.
GHIA_CENTRES = {100: (0.6172, 0.7344), 400: (0.5547, 0.6055), 1000: (0.5313, 0.5625)}
GHIA_PSI_MIN = {100: 0.103423, 400: 0.113909, 1000: 0.117929}   # |psi| at those centres


def predict(model, re):
    """Evaluate the exported MLP on the 50 x 50 cell centres.

    Args:
        model: browser-model dict (normalization + layers).
        re: Reynolds number.
    Returns:
        array [N, N, 3] of u, v, p with row index increasing in y.
    """
    a = (np.column_stack((XX.ravel(), YY.ravel(), np.full(N * N, re))) - model['input_mean']) / model['input_std']
    for layer in model['layers']:
        a = a @ np.array(layer['weight']).T + layer['bias'] if layer['type'] == 'Linear' else np.tanh(a)
    return (a * model['output_std'] + model['output_mean']).reshape(N, N, 3)


def vortex_centre(field):
    """Locate the primary vortex as the streamfunction minimum, refined by a parabola.

    Args:
        field: array [N, N, 3].
    Returns:
        (x, y, psi_min).
    """
    psi = np.cumsum(field[:, :, 0], axis=0) * H
    j, i = np.unravel_index(np.argmin(psi), psi.shape)

    def offset(fm, f0, fp):
        den = fm - 2 * f0 + fp
        return 0.0 if den == 0 else 0.5 * (fm - fp) / den
    dx = offset(psi[j, i - 1], psi[j, i], psi[j, i + 1]) if 0 < i < N - 1 else 0.0
    dy = offset(psi[j - 1, i], psi[j, i], psi[j + 1, i]) if 0 < j < N - 1 else 0.0
    return CELL[i] + dx * H, CELL[j] + dy * H, psi[j, i]


def divergence_rms(field):
    """Interior RMS of the central-difference divergence of the cell-centred velocity.

    Args:
        field: array [N, N, 3].
    Returns:
        float RMS of du/dx + dv/dy, two cells away from walls.
    """
    div = np.gradient(field[:, :, 0], H, axis=1) + np.gradient(field[:, :, 1], H, axis=0)
    return float(np.sqrt(np.mean(div[2:-2, 2:-2] ** 2)))


def case_stats(model, re, cfd, test=False):
    """Compare one CFD field with the network at the same Re.

    Args:
        model: browser-model dict.
        re: Reynolds number.
        cfd: CFD field [N, N, 3].
        test: whether the case was held out of training.
    Returns:
        dict of RMSE, vortex centres and strengths, and divergence for CFD and NN.
    """
    nn = predict(model, re)
    rmse = np.sqrt(((nn - cfd) ** 2).mean(axis=(0, 1)))
    ct, cp = vortex_centre(cfd), vortex_centre(nn)
    return {'re': round(float(re), 3), 'test': test, 'rmse': [round(float(v), 5) for v in rmse],
            'cfd_center': [round(float(v), 4) for v in ct[:2]], 'nn_center': [round(float(v), 4) for v in cp[:2]],
            'cfd_psi': round(float(ct[2]), 5), 'nn_psi': round(float(cp[2]), 5),
            'div': [round(divergence_rms(cfd), 4), round(divergence_rms(nn), 4)]}


def build_data():
    """Assemble everything the page embeds.

    Returns:
        dict serialized into the page's JSON block.
    """
    model = json.loads((ROOT / 'data' / 'model.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'data' / 'cfd_re20_300.npz') as d:
        fields, res = d['fields'], d['reynolds']
    with np.load(ROOT / 'data' / 'cfd_extrapolation.npz') as d:
        ext_fields, ext_res, ext_diag = d['fields'], d['reynolds'], d['diagnostics']

    # The extrapolation runner reuses the notebook solver; Re 300 exists in both sets, so they must agree.
    k300 = int(np.argmin(abs(res - 300)))
    solver_check = float(np.max(np.abs(ext_fields[0] - fields[k300])))
    log.info(f'Re 300 solver check, max |difference| vs notebook dataset: {solver_check:.2e}')

    cases = [case_stats(model, res[k], fields[k], k in TEST_IDS) for k in range(len(res))]
    ext_cases = [dict(case_stats(model, re, f), steps=int(dg[0])) for re, f, dg in zip(ext_res, ext_fields, ext_diag)]

    err = np.stack([predict(model, res[k]) - fields[k] for k in TEST_IDS])
    top = YY > 0.9
    share = [round(float((err[..., q] ** 2)[:, top].sum() / (err[..., q] ** 2).sum()), 3) for q in range(3)]
    corners = {}
    for re in (20, 300):
        k = int(np.argmin(abs(res - re))); t, p = fields[k], predict(model, res[k])
        corners[str(re)] = {'cfd': [round(float(t[-1, 0, 2]), 2), round(float(t[-1, -1, 2]), 2)],
                            'nn': [round(float(p[-1, 0, 2]), 2), round(float(p[-1, -1, 2]), 2)]}
    return {'model': model, 'cases': cases, 'ext_cases': ext_cases, 'top_band_share': share, 'corner_p': corners,
            'ghia': {str(k): v for k, v in GHIA_CENTRES.items()},
            'ghia_psi': {str(k): v for k, v in GHIA_PSI_MIN.items()}, 'solver_check': solver_check,
            'cfd_test': {str(round(float(res[k]), 3)): np.round(fields[k], 4).tolist() for k in TEST_IDS},
            'cfd_ext': {str(int(re)): np.round(f, 4).tolist() for re, f in zip(ext_res, ext_fields)}}


def build(out_path=ROOT / 'index.html'):
    """Render the template into a standalone HTML document.

    Args:
        out_path: destination file.
    Returns:
        Path of the written page.
    """
    data = build_data()
    template = (ROOT / 'tools' / 'template.html').read_text(encoding='utf-8')
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    head, body = template.replace('/*AUTHOR*/', AUTHOR).replace('/*DATA*/', payload).split('</style>', 1)
    head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<meta name="author" content="{AUTHOR}">\n'
            '<meta name="description" content="Neural-network surrogate for lid-driven cavity flow, compared with CFD '
            'inside and outside its training range (Re 20-1000).">\n'
            + head + '\n[hidden] { display: none !important; }\nimg { max-width: 100%; }\n</style>\n</head>\n<body>\n'
            + body.lstrip('\n') + '\n</body>\n</html>\n')
    out_path.write_text(page, encoding='utf-8', newline='\n')
    log.info(f'Wrote {out_path} ({len(page.encode()) / 1024:.0f} KB)')
    for c in data['ext_cases']:
        log.info(f"Re {c['re']:6.0f}: RMSE u {c['rmse'][0]:.4f} v {c['rmse'][1]:.4f} p {c['rmse'][2]:.4f} | "
                 f"centre CFD ({c['cfd_center'][0]:.3f}, {c['cfd_center'][1]:.3f}) NN ({c['nn_center'][0]:.3f}, "
                 f"{c['nn_center'][1]:.3f}) | psi CFD {c['cfd_psi']:.4f} NN {c['nn_psi']:.4f} | div NN {c['div'][1]:.3f}")
    return out_path


if __name__ == '__main__':
    build()
