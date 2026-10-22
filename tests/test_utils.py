import warnings

import numpy as np
import pytest

from reflecto_exp.io_utils import append_timestamp, load_xrr_dat, next_unique_file
from reflecto_exp.math_utils import fom_log, i0_normalize, powerspace
from reflecto_exp.physics_utils import tth2q


def test_powerspace_keeps_endpoints_and_reduces_to_linspace_for_power_one():
    x = powerspace(2.0, 8.0, 7, power=1.0)
    np.testing.assert_allclose(x, np.linspace(2.0, 8.0, 7))
    y = powerspace(2.0, 8.0, 7, power=1.5)
    assert y[0] == 2.0 and y[-1] == 8.0
    assert np.all(np.diff(y) > 0)


def test_fom_log_is_the_mean_decade_difference():
    assert fom_log(np.array([1.0, 10.0]), np.array([10.0, 100.0])) == pytest.approx(1.0)
    assert fom_log(np.array([3.0, 5.0]), np.array([3.0, 5.0])) == 0.0


def test_i0_normalize_scales_the_maximum_to_one():
    out = i0_normalize(np.array([2.0, 4.0, 8.0]))
    np.testing.assert_allclose(out, [0.25, 0.5, 1.0])


def test_i0_normalize_warns_and_returns_zeros_unchanged():
    zeros = np.zeros(3)
    with pytest.warns(RuntimeWarning):
        out = i0_normalize(zeros)
    np.testing.assert_array_equal(out, zeros)


def test_tth2q_matches_the_bragg_relation():
    # q = 4 pi sin(theta) / lambda; 2theta = 60 deg gives theta = 30 deg
    assert tth2q(60.0, 1.54) == pytest.approx(4 * np.pi * 0.5 / 1.54)
    assert isinstance(tth2q(60.0), float)
    arr = tth2q(np.array([0.0, 60.0]), 1.54)
    assert isinstance(arr, np.ndarray) and arr[0] == 0.0


def test_next_unique_file_numbers_after_the_highest_existing(tmp_path):
    base = tmp_path / "out.txt"
    assert next_unique_file(base) == base
    base.write_text("a")
    assert next_unique_file(base) == tmp_path / "out(1).txt"
    (tmp_path / "out(1).txt").write_text("b")
    (tmp_path / "out(4).txt").write_text("c")
    assert next_unique_file(base) == tmp_path / "out(5).txt"


def test_append_timestamp_keeps_the_suffix(tmp_path):
    with_suffix = append_timestamp(tmp_path / "scan.dat")
    assert with_suffix.suffix == ".dat" and with_suffix.name.startswith("scan_")
    no_suffix = append_timestamp(tmp_path / "scan")
    assert no_suffix.suffix == "" and no_suffix.name.startswith("scan_")


def test_load_xrr_dat_skips_comments_reads_scientific_notation_and_zeros_garbage(tmp_path):
    f = tmp_path / "x.dat"
    f.write_text("# header\n0.5  1.0e-05\n1.0\t2.5E-3  # trailing comment\n1.5  oops\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tth, r = load_xrr_dat(f)
    np.testing.assert_allclose(tth, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(r, [1.0e-05, 2.5e-3, 0.0])
