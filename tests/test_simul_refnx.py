"""A refnx structure of n layers holds the n layers once, not the n-layer stack n times (#12)."""

import numpy as np
import pytest

pytest.importorskip("refnx")


def two_layers():
    from reflecto_exp.simulate.simul_refnx import ParamSet

    return [ParamSet(100.0, 3.0, 5.0), ParamSet(50.0, 2.0, 8.0)]


def test_two_layers_appear_once_each():
    from reflecto_exp.simulate.simul_refnx import build_structure

    slabs = build_structure(two_layers()).slabs()  # columns: thickness, SLD re, SLD im, roughness, solvent
    np.testing.assert_allclose(slabs[:, 0], [0.0, 100.0, 50.0, 0.0])
    np.testing.assert_allclose(slabs[:, 1], [0.0, 5.0, 8.0, 20.0])
    np.testing.assert_allclose(slabs[:, 3], [0.0, 3.0, 2.0, 3.0])


def test_two_layer_curve_matches_an_explicit_structure():
    from refnx.reflect import SLD, ReflectModel

    from reflecto_exp.simulate.simul_refnx import build_structure, calc_refl

    q = np.linspace(0.01, 0.3, 200)
    ours = calc_refl(build_structure(two_layers()), q, is_footprint=False)
    explicit = SLD(0.0)(0, 0) | SLD(5.0)(100.0, 3.0) | SLD(8.0)(50.0, 2.0) | SLD(20)(0, 3)
    np.testing.assert_allclose(ours, ReflectModel(explicit)(q), rtol=1e-10)
