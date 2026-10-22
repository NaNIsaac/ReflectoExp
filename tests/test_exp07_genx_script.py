"""exp07's GenX script (runs/exp07_fourier_hires/engine): the fitted background must exist and reach
the simulation (ReflectoExp#9, item 1).

fitting.py registers "v.set_ibkg" as a fit parameter, but the script defined no ibkg, so GenX
raised AttributeError when the parameter was registered, and the instrument used a fixed 1e-10.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

genx = pytest.importorskip("genx")
from genx.data import DataList, DataSet  # noqa: E402
from genx.model import Model  # noqa: E402
from genx.parameters import Parameters  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "runs" / "exp07_fourier_hires"))
from engine.script_builder import GenXScriptBuilder  # noqa: E402

INIT = {
    "f_d": 300.0,
    "f_sig": 4.0,
    "f_sld": 40.0,
    "s_d": 15.0,
    "s_sig": 3.0,
    "s_sld": 18.9,
    "i0": 1.0,
    "ibkg": 1e-7,
    "s_len": 10.0,
    "beam_w": 0.1,
}


@pytest.fixture
def model():
    ds = DataSet(name="test")  # as in engine.fitting
    ds.x_raw = np.linspace(0.01, 0.5, 300)
    ds.y_raw = np.ones_like(ds.x_raw)
    ds.error_raw = np.full_like(ds.x_raw, 0.1)
    ds.run_command()
    m = Model()
    m.data = DataList([ds])
    m.set_script(GenXScriptBuilder().build(INIT))
    m.compile_script()
    return m


def simulate(model, **values):
    v = model.script_module.v
    for name, value in values.items():
        getattr(v, f"set_{name}")(value)
    # compile=True would rerun the script and reset v; a fit does not recompile either
    model.simulate(compile=False)
    return np.asarray(model.data[0].y_sim, float)


def test_background_can_be_registered_as_a_fit_parameter(model):
    pars = Parameters()
    pars.append("v.set_ibkg", model)  # what engine.fitting does
    model.parameters = pars


def test_background_reaches_the_simulation(model):
    low = simulate(model, ibkg=1e-7)
    high = simulate(model, ibkg=1e-5)
    assert high[-1] - low[-1] == pytest.approx(1e-5 - 1e-7, rel=1e-6)


def test_intensity_scale_reaches_the_simulation(model):
    one = simulate(model, i0=1.0)
    two = simulate(model, i0=2.0)
    assert two[0] / one[0] == pytest.approx(2.0, rel=1e-6)
