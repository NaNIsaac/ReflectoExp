import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reflecto_exp.simulate.torch_engine.abeles import abeles  # noqa: E402
from reflecto_exp.simulate.torch_engine.numpy_implementations import abeles_np  # noqa: E402

Q = np.linspace(0.005, 0.3, 120)


def _run_both(thickness, roughness, sld):
    got = abeles(torch.tensor(Q), torch.tensor(thickness), torch.tensor(roughness), torch.tensor(sld))
    want = abeles_np(Q, np.asarray(thickness), np.asarray(roughness), np.asarray(sld))
    return got.numpy(), np.asarray(want)


def test_torch_and_numpy_engines_agree_on_a_two_layer_stack():
    got, want = _run_both([[120.0, 30.0]], [[3.0, 2.0, 4.0]], [[40.0, 10.0, 20.0]])
    assert got.shape == want.shape == (1, Q.size)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-14)


def test_engines_agree_on_a_batch():
    thickness = [[100.0], [250.0], [40.0]]
    roughness = [[2.0, 5.0], [0.0, 3.0], [1.0, 1.0]]
    sld = [[30.0, 20.0], [45.0, 20.0], [10.0, 20.0]]
    got, want = _run_both(thickness, roughness, sld)
    assert got.shape == (3, Q.size)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-14)


def test_total_reflection_below_the_critical_q_and_decay_above():
    # One bare layer of SLD 30e-6 / A^2 on a substrate of the same SLD: a single Fresnel interface.
    # Critical q is 4 sqrt(pi * rho) = 4 * sqrt(pi * 30e-6) = 0.0388 A^-1.
    got, _ = _run_both([[0.0]], [[0.0, 0.0]], [[30.0, 30.0]])
    qc = 4 * np.sqrt(np.pi * 30e-6)
    below = Q < 0.9 * qc
    above = Q > 3 * qc
    assert np.all(got[0, below] > 0.999)
    assert np.all(got[0, above] < 1e-2)
    assert np.all(got <= 1.0)
