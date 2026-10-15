# Reflecto: Deep Learning for X-ray Reflectivity Analysis

Automated extraction of thin film parameters (thickness, roughness, electron density) from X-ray Reflectivity measurements using deep learning.

---

## 🎯 Overview

**The Problem:**
- Traditional XRR analysis relies on manual fitting
- Time-consuming and requires expert knowledge
- Sensitive to initial parameter guesses

**Our Solution:**
- Physics-based simulation for training data generation
- End-to-end learning with CNN
- Real-time analysis (seconds)

---
## Preperation before use
### Windows

```bash
# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex

# Clone Repository
cd ReflectoExp
git clone https://github.com/NaNIsaac/ReflectoExp

# Make virtual environment
uv venv --python 3.13
.venv/scripts/activate

# Install packages
uv sync
uv run python -m ipykernel install --user --name=my-env --display-name="reflecto"
uv pip install -e .
```
---

## 🚀 Quick Start

```bash
# Run entire pipeline (data generation → training → evaluation)
python runs/exp05_1layer_mask/main.py
```

That's it! The script automatically handles data generation, model training, and evaluation.

---

## 📁 Project Structure

```
runs/
├── exp01_quan_class/      # Classification approach for multi-layer
├── exp02_one_layer/       # Regression for single layer (RefNX)
├── exp03_physics_fused/   # Physics-guided learning (experimental)
├── exp04_one_genx/        # Regression for single layer (GenX)
├── exp05_1layer_mask/
└── exp06_1lay_6out/
```

**Each experiment contains:**
- `config.py` - Experiment settings
- `main.py` - Full pipeline runner
- `model.py` - Neural network architecture
- `evaluate.py` - Performance analysis

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    "param_ranges": {
        "thickness": (20.0, 200.0),  # Å
        "roughness": (0.0, 10.0),    # Å
        "sld": (0.0, 140.0),         # 1e-6 Å^-2
    },
    "simulation": {
        "n_samples": 1_000_000,
        "q_points": 200,
    },
    "training": {
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
    }
}
```

---

## 🔬 Experiments Comparison (Keep develping)

| Experiment | Target | Approach | Best For |
|------------|--------|----------|----------|
| exp01 | Multi-layer | Classification | Fast screening |
| exp02/04 | Single layer | Regression | High accuracy |
| exp03 | Multi-layer | Physics-fused | Interpretability |

**Recommendation:** Start with **exp02** or **exp04** for single-layer analysis.

---

## 📊 Performance

Typical results on simulated data (single layer):
- Thickness: MAE ~2Å
- Roughness: MAE ~1Å
- SLD: MAE ~0.15 (×10⁻⁶ Å⁻²)

---

## 🔧 Key Features

**Simulation:**
- RefNX/GenX-based XRR generation
- Realistic noise models (Poisson + background)
- Physical constraints enforcement

**Training:**
- Automatic checkpoint management
- Mixed precision support
- Resume from interruption

**Evaluation:**
- Comprehensive metrics (MAE, RMSE)
- Visualization tools (parity plots, error distribution)
- Real experimental data support
