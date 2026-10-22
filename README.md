# ReflectoExp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Experiment record. The maintained library, with on-the-fly simulation and a differentiable Abeles engine, is [ReflectoLearn](https://github.com/NaNIsaac/ReflectoLearn).

Experiment scripts for training neural networks that predict thin-film thickness, roughness and SLD from X-ray reflectivity (XRR) curves, using GenX (and earlier refnx) simulations as training data.

Each folder under `runs/` is one self-contained experiment (config, simulation, dataset, model, training, evaluation). Shared simulation and physics helpers live in the `reflecto_exp` package under `src/`.

## Install

Uses [uv](https://docs.astral.sh/uv/) and Python 3.13. `torch` is pulled from the PyTorch CUDA 12.8 index.

```bash
git clone https://github.com/NaNIsaac/ReflectoExp.git
cd ReflectoExp
```

Install the dependencies and the `reflecto_exp` package:

<!-- test -->
```bash
uv sync
uv pip install -e .
```

The refnx-based experiments (exp02, exp03) also need `uv pip install refnx`, which is not a declared dependency.

## Usage

Experiment scripts import their siblings as top-level modules (`import config`, `import simulate`), so they are run from inside their folder. The root `main.py` does that:

```bash
uv run python main.py 7      # runs runs/exp07_fourier_hires/main.py with the .venv Python
```

`main.py` of an experiment generates the HDF5 training set if it is missing, then trains and evaluates. Before running, set `base_dir` (and `exp_name`) in that experiment's `config.py`; the committed values are local Windows paths.

## Experiments

| Folder | What it tries |
|---|---|
| `exp01_quan_class` | multilayer parameters as quantized classes (imports modules that no longer exist in `reflecto_exp`; kept for reference) |
| `exp02_one_layer` | single-layer regression on refnx simulations |
| `exp03_physics_fused` | CNN prediction blended with a physics-based thickness estimate, experimental |
| `exp04_one_genx` | single-layer regression on GenX simulations |
| `exp05_1layer_mask` | single layer, valid-q mask as a second input channel, instrument augmentation; GenX fit started from the CNN guess |
| `exp06_1lay_6out` | film + SiO2 native oxide (6 outputs) |
| `exp07_fourier_hires` | 6 outputs, 2000-point q grid, Fourier features of q; `ablation.py` and `metrics_report.py` for component ablation and extended metrics |

## Structure

```
src/reflecto_exp/
  simulate/simul_genx.py      GenX-based XRR simulator (XRRSimulator, ParamSet)
  simulate/simul_refnx.py     refnx-based simulator
  simulate/noise.py           Poisson and background noise
  simulate/torch_engine/      PyTorch Abeles / kinematical reflectivity, smearing
  physics_utils.py            unit conversions (tth2q, ...)
runs/expNN_*/                 experiments
notebooks/                    function tests
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
