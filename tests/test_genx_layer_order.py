"""The GenX fitting engines must use the stack the training data were simulated with (ReflectoExp#9,
item 2): Si / SiO2 / film, the native oxide under the film."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("genx")
from reflecto_exp.simulate.simul_genx import ParamSet, build_sample  # noqa: E402

RUNS = Path(__file__).parents[1] / "runs"
sys.path.insert(0, str(RUNS / "exp07_fourier_hires"))
from engine.fitting import GenXFitter, XRRConfig  # noqa: E402

SIO2, SI = 18.8, 20.07  # the engines' starting oxide SLD and fixed substrate SLD
Q = np.linspace(0.05, 0.35, 50)


def load_fitter(run):
    spec = importlib.util.spec_from_file_location(f"{run}_fitting_engine", RUNS / run / "fitting_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GenXFitter


def thicknesses_from_substrate(sample):
    return list(sample.resolveLayerParameters().d)


def parratt(q, layers):
    """|r|^2 for [(sld 1e-6/A^2, thickness, roughness of its top interface)], ambient first."""
    sld = np.array([lay[0] for lay in layers], complex) * 1e-6
    kz = np.sqrt((q[:, None] / 2) ** 2 - 4 * np.pi * (sld[None, :] - sld[0]) + 0j)
    x = np.zeros(len(q), complex)
    for j in range(len(layers) - 2, -1, -1):
        k1, k2 = kz[:, j], kz[:, j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * layers[j + 1][2] ** 2)
        d = layers[j + 1][1] if j + 1 < len(layers) - 1 else 0.0
        p = np.exp(2j * k2 * d)
        x = (r + x * p) / (1 + r * x * p)
    return np.abs(x) ** 2


def test_simulator_buries_the_oxide():
    sample = build_sample([ParamSet(300.0, 4.0, 40.0)], ParamSet(15.0, 3.0, SIO2))
    assert thicknesses_from_substrate(sample) == [0.0, 15.0, 300.0, 0.0]


@pytest.mark.parametrize("run", ["exp05_1layer_mask", "exp06_1lay_6out"])
def test_fitter_buries_the_oxide(run):
    film, oxide = ParamSet(300.0, 4.0, 40.0), ParamSet(15.0, 3.0, SIO2)
    args = (film, oxide) if run == "exp06_1lay_6out" else (film,)
    fitter = load_fitter(run)(Q, np.ones_like(Q), *args)
    assert thicknesses_from_substrate(fitter.model.script_module.sample) == [0.0, 15.0, 300.0, 0.0]


def test_exp07_fitter_buries_the_oxide():
    fitter = GenXFitter(Q, np.ones_like(Q), SimpleNamespace(thickness=300.0, roughness=4.0, sld=40.0))
    # the exp07 script starts the oxide at 15 A
    assert thicknesses_from_substrate(fitter.model.script_module.sample) == [0.0, 15.0, 300.0, 0.0]


def test_exp07_fit_recovers_a_film_on_buried_oxide():
    # Independent Parratt curve of air / film / SiO2 / Si, smeared with the script's Gaussian
    # resolution (sigma 0.002 1/A)
    q = np.linspace(0.05, 0.35, 300)
    qf = np.linspace(0.04, 0.36, 4000)
    w = np.exp(-0.5 * ((qf[None] - q[:, None]) / 0.002) ** 2)
    layers = [(0.0, 0.0, 0.0), (40.0, 200.0, 4.0), (SIO2, 20.0, 3.0), (SI, 0.0, 3.0)]
    r = (w @ parratt(qf, layers)) / w.sum(1)

    np.random.seed(0)
    guess = SimpleNamespace(thickness=206.0, roughness=4.12, sld=41.2)
    res = GenXFitter(q, r, guess, XRRConfig(sample_len_init=45.0, pop_size=20)).run(verbose=False)

    assert res["set_f_d"] == pytest.approx(200.0, rel=0.01)
    assert res["set_f_sig"] == pytest.approx(4.0, rel=0.1)
    assert res["set_f_sld"] == pytest.approx(40.0, rel=0.05)
    assert res["set_s_d"] == pytest.approx(20.0, abs=6.0)
    assert res["set_s_sld"] == pytest.approx(SIO2, abs=1.0)
