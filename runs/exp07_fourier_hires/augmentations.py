import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

from reflecto_exp.simulate.noise import apply_poisson_noise, get_background_noise


class XRRAugmentations:
    def __init__(
        self,
        intensity_noise_scale: float = 0.2,
        bg_range: tuple[float, float] = (-8, -5),
        res_sigma_range: tuple[float, float] = (0.0001, 0.005),
        q_shift_sigma: float = 0.002,
        delta_q: float = 0.0005,
        prob: float = 0.95,
        beam_width_range: tuple = (0.005, 0.015),
        sample_len_range: tuple = (5, 30.0),
        do_footprint: bool = True,
    ):
        self.intensity_noise_scale = intensity_noise_scale
        self.bg_range = bg_range
        self.res_sigma_range = res_sigma_range
        self.q_shift_sigma = q_shift_sigma
        self.delta_q = delta_q
        self.prob = prob

        # Footprint settings
        self.beam_width_range = beam_width_range
        self.sample_len_range = sample_len_range
        self.do_footprint = do_footprint

    def _gauss_footprint(self, theta_deg: np.ndarray, beam_w: float, sample_len: float) -> np.ndarray:
        """
        GenX GaussIntensity logic implemented for NumPy.
        theta_deg: incidence angle (deg)
        beam_w: standard deviation of the beam (mm)
        sample_len: sample length (mm)
        """
        # GenX logic: GaussIntensity(alpha, s1, s2, sigma_x) with s1 = s2 = sample_len / 2 (symmetric about the sample center)

        rad = np.pi / 180.0
        sqrt2 = np.sqrt(2.0)

        sinalpha = np.sin(theta_deg * rad)

        # avoid division by zero at theta = 0
        sinalpha = np.maximum(sinalpha, 1e-9)

        s_half = sample_len / 2.0

        # Gaussian beam spill-over (GenX formula): fraction of the beam inside [-s_half, +s_half]
        common = sinalpha / (sqrt2 * beam_w)

        # erf is odd, so the two erf terms are equal and (2 * erf(...)) / 2 = erf(...)
        correction = erf(s_half * common)

        return correction

    def apply_footprint(self, q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Apply a random beam width and sample size to simulate the low-angle intensity loss."""
        beam_w = np.random.uniform(*self.beam_width_range)
        sample_len = np.random.uniform(*self.sample_len_range)

        # q -> theta (X-ray wavelength 1.54 A)
        # q = 4pi/lambda * sin(theta) => theta = arcsin(q * lambda / 4pi)
        wavelength = 1.54
        sin_theta = q * wavelength / (4 * np.pi)
        # keep the arcsin argument in range
        sin_theta = np.clip(sin_theta, 0, 1)
        theta_deg = np.degrees(np.arcsin(sin_theta))

        foocor = self._gauss_footprint(theta_deg, beam_w, sample_len)

        return R * foocor

    def __call__(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return q, R

        R_aug = R.copy()

        # order follows the physics: footprint, instrument resolution, detector noise and scale, then q misalignment
        if self.do_footprint:
            R_aug = self.apply_footprint(q, R_aug)

        # resolution smears the footprint-cut curve (physical order)
        res_sigma_q = np.random.uniform(*self.res_sigma_range)
        sigma_px = res_sigma_q / self.delta_q

        # Gaussian Smearing (Convolution)
        if sigma_px > 0:
            R_aug = gaussian_filter1d(R_aug, sigma=sigma_px, mode="nearest")

        # Global Scaling (I0 fluctuation)
        scale = np.random.uniform(1 - self.intensity_noise_scale, 1 + self.intensity_noise_scale)
        R_aug *= scale

        # Background Noise (Dark current + Scattering)
        bg_level = np.random.uniform(*self.bg_range)
        N = len(R_aug)
        bg = get_background_noise(N, bg_level, bg_level + 0.5)
        R_aug = R_aug + np.abs(bg)

        # Poisson Noise (Counting statistics - Shot noise)
        # signal-dependent noise: a stronger signal has more noise
        s_rand = pow(10, np.random.uniform(5, 8))
        R_aug = apply_poisson_noise(R_aug, s=s_rand)

        # q-axis misalignment (calibration error)
        if self.q_shift_sigma > 0:
            q_shift = np.random.normal(0, self.q_shift_sigma)
            # a shifted q axis moves the data the opposite way
            R_aug = np.interp(q, q + q_shift, R_aug, left=R_aug[0], right=R_aug[-1])

        return q, R_aug
