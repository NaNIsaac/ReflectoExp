import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reflecto_exp.simulate.torch_engine.abeles import abeles  # noqa: E402
from reflecto_exp.simulate.torch_engine.numpy_implementations import abeles_np  # noqa: E402

Q = np.linspace(0.005, 0.3, 120)


def parratt(q, thickness, roughness, sld):
    """Reflectivity by the Parratt recursion; SLD in 1e-6 A^-2, vacuum above the first layer.

    Written from the recursion itself so it shares no code with either engine.
    """
    rho = np.concatenate([[0.0], sld]) * 1e-6
    kz = [np.sqrt((q / 2) ** 2 - 4 * np.pi * r + 0j) for r in rho]
    amplitude = None
    for j in range(len(rho) - 2, -1, -1):
        r = (kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1]) * np.exp(-2 * kz[j] * kz[j + 1] * roughness[j] ** 2)
        if amplitude is None:
            amplitude = r
        else:
            phase = np.exp(2j * kz[j + 1] * thickness[j])
            amplitude = (r + amplitude * phase) / (1 + r * amplitude * phase)
    return np.abs(amplitude) ** 2


STACKS = [
    ([120.0, 30.0], [3.0, 2.0, 4.0], [40.0, 10.0, 20.0]),
    ([250.0], [0.0, 3.0], [45.0, 20.0]),
    ([40.0], [1.0, 1.0], [10.0, 20.0]),
]


@pytest.mark.parametrize("thickness, roughness, sld", STACKS)
def test_numpy_engine_matches_the_parratt_recursion_in_double_precision(thickness, roughness, sld):
    got = abeles_np(Q, np.array([thickness]), np.array([roughness]), np.array([sld]))[0]
    np.testing.assert_allclose(got, parratt(Q, thickness, roughness, sld), rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("thickness, roughness, sld", STACKS)
def test_torch_engine_matches_the_parratt_recursion(thickness, roughness, sld):
    # Tolerance 1e-5: near a critical edge the torch engine is off by up to 4e-7 relative.
    got = abeles(Q_T, *(torch.tensor([x]) for x in (thickness, roughness, sld))).numpy()[0]
    np.testing.assert_allclose(got, parratt(Q, thickness, roughness, sld), rtol=1e-5, atol=1e-12)


Q_T = torch.tensor(Q)


def test_engines_agree_on_a_batch():
    thickness = [[100.0], [250.0], [40.0]]
    roughness = [[2.0, 5.0], [0.0, 3.0], [1.0, 1.0]]
    sld = [[30.0, 20.0], [45.0, 20.0], [10.0, 20.0]]
    got = abeles(Q_T, torch.tensor(thickness), torch.tensor(roughness), torch.tensor(sld)).numpy()
    want = abeles_np(Q, np.array(thickness), np.array(roughness), np.array(sld))
    assert got.shape == want.shape == (3, Q.size)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-12)


def test_total_reflection_below_the_critical_q_and_decay_above():
    # One bare layer of SLD 30e-6 / A^2 on a substrate of the same SLD: a single Fresnel interface.
    # Critical q is 4 sqrt(pi * rho) = 4 * sqrt(pi * 30e-6) = 0.0388 A^-1.
    got = abeles(Q_T, torch.tensor([[0.0]]), torch.tensor([[0.0, 0.0]]), torch.tensor([[30.0, 30.0]])).numpy()
    qc = 4 * np.sqrt(np.pi * 30e-6)
    assert np.all(got[0, Q < 0.9 * qc] > 0.999)
    assert np.all(got[0, Q > 3 * qc] < 1e-2)
    assert np.all(got <= 1.0)
