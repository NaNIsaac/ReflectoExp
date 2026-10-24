import json
from pathlib import Path

import numpy as np
import torch
from config import CONFIG
from dataset import XRRPreprocessor
from xrr_model import XRR1DRegressor

from reflecto_exp.math_utils import powerspace


class XRRInferenceEngine:
    def __init__(self, exp_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Device: {self.device}")

        if exp_dir is None:
            exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
        else:
            exp_dir = Path(exp_dir)

        self.stats_file = exp_dir / "stats.pt"
        self.checkpoint_file = exp_dir / "best.pt"
        self.config_file = exp_dir / "config.json"

        if self.config_file.exists():
            with open(self.config_file) as f:
                self.train_config = json.load(f)
            print(f"[Inference] Loaded training config from {self.config_file}")
        else:
            print("[Inference] Warning: config.json not found. Fallback to current global CONFIG.")
            self.train_config = CONFIG

        # must match the training grid
        sim_conf = self.train_config["simulation"]
        q_min = sim_conf["q_min"]
        q_max = sim_conf["q_max"]
        n_points = sim_conf["q_points"]
        power = sim_conf["power"]

        print(f"[Inference] Grid Generation: Power={power}, Points={n_points}")

        self.target_qs = powerspace(q_min, q_max, n_points, power=power).astype(np.float32)

        # process_input interpolates the incoming evenly spaced data onto target_qs (a nonlinear grid)
        self.processor = XRRPreprocessor(qs=self.target_qs, stats_file=self.stats_file, device=self.device)

        self._load_model()

    def _load_model(self):
        ckpt = torch.load(self.checkpoint_file, map_location=self.device)

        # Config Fallback
        model_args = ckpt.get("config", {}).get(
            "model_args",
            {
                "q_len": len(self.target_qs),
                "input_channels": 2,
                "n_channels": CONFIG["model"]["n_channels"],
                "depth": CONFIG["model"]["depth"],
                "mlp_hidden": CONFIG["model"]["mlp_hidden"],
            },
        )

        self.model = XRR1DRegressor(**model_args).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[Inference] Model loaded from {self.checkpoint_file}")

    def predict(self, q_raw, R_raw):
        """
        q_raw: q of the measured data (need not be evenly spaced)
        R_raw: measured R
        """
        # np.interp resamples evenly spaced data onto the model's powerspace grid
        x = self.processor.process_input(q_raw, R_raw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred_norm = self.model(x).squeeze(0)

        y_pred = self.processor.denormalize_params(y_pred_norm)

        return y_pred
