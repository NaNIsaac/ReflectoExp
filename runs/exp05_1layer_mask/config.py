import json
from pathlib import Path

import numpy as np

from reflecto_exp.physics_utils import tth2q

CONFIG = {
    "exp_name": "wider_sld",
    "base_dir": Path(r"D:\Members\IsaacYong\Data\XRR_AI\one_layer"),
    # "base_dir": Path(r"D:\data\XRR_AI\mask"),
    "param_ranges": {
        "thickness": None,
        "roughness": None,
        "sld": None,
        "sio2_thickness": None,
        "sio2_roughness": None,
        "sio2_sld": None,
    },
    "simulation": {"wavelength": 1.54, "n_samples": int(1e6), "q_points": 1000, "q_min": tth2q(0.1), "q_max": tth2q(15), "power": 1},
    "model": {
        "n_channels": 32,
        "depth": 5,
        "mlp_hidden": 256,
        "dropout": 0.02507346016270837,
    },
    "training": {
        "batch_size": 32,
        "epochs": 150,
        "lr": 0.0006119557822568638,
        "weight_decay": 0.0005310131070268705,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "patience": 15,
        "num_workers": 0,
        "augment": True,
        "aug_prob": 0.5,
        "q_shift_sigma": 0.002,  # +/- 0.002 A^-1 shift
        "intensity_scale": 0.15,  # +/- 15% intensity variation
        "expand_factor": 10,
    },
}


def save_config(config: dict, file: Path | str):
    """
    Save a dict as a JSON file.
    - Path -> str
    - numpy types -> native Python types
    - recursive over dicts, lists and tuples
    - any other type is stored as its repr() string (fallback)
    """

    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]

        # JSON-native types pass through; otherwise a str would go through repr() and be stored with doubled quotes
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        try:
            return repr(obj)
        except Exception:
            return "<Unserializable Object>"

    safe_config = _to_serializable(config)

    save_path = Path(file)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(safe_config, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    from pprint import pprint

    pprint(CONFIG)
