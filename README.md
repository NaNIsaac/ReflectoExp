# ReflectoExp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Experiment scripts for training neural networks that predict thin-film thickness, roughness and SLD from X-ray reflectivity (XRR) curves, using GenX (and earlier refnx) simulations as training data.

Each folder under `runs/` is one experiment, most with their own config, simulation, dataset, model, training and evaluation (exp01 and exp03 lack some of these, see the table). Shared simulation and physics helpers live in the `reflecto_exp` package under `src/`.

This is one of the XRR research branches. [Reflecto](https://github.com/NaNIsaac/Reflecto) packages exp07 weights (as of December 2025) with a GenX refinement, [XRefine](https://github.com/NaNIsaac/XRefine) started from `src/reflecto_exp/simulate/torch_engine/` (both 2025-12-30), and [ReflectoLearn](https://github.com/NaNIsaac-Research/ReflectoLearn) took the exp07 model and training code in June 2026. exp07 kept being developed here afterwards (ablation, August 2026). None of these later repositories has been compared with the results here.

## Install

Uses [uv](https://docs.astral.sh/uv/) and Python 3.13. `torch` is pulled from the PyTorch CUDA 12.8 index (PyPI on macOS).

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

The refnx-based experiments (exp01, exp02, exp03) also need `uv pip install refnx`, which is not a declared dependency.

## Usage

Experiment scripts import their siblings as top-level modules (`import config`, `import simulate`), so they are run from inside their folder. The root `main.py` does that:

```bash
uv run python main.py 7      # runs runs/exp07_fourier_hires/main.py with the .venv Python
```

`main.py` of an experiment generates the HDF5 training set if it is missing, then trains and evaluates. Before running, set `base_dir` (and `exp_name`) in that experiment's `config.py`; exp02 and exp06 read it from `$DATA`, exp04, exp05 and exp07 still have local Windows paths.

## Experiments

| Folder | What it tries |
|---|---|
| `exp01_quan_class` | multilayer parameters as quantized classes (imports modules that no longer exist in `reflecto_exp`; kept for reference) |
| `exp02_one_layer` | single-layer regression on refnx simulations |
| `exp03_physics_fused` | CNN prediction blended with a physics-based thickness estimate, experimental (`train_fused.py` imports `fused_loss_with_confidence`, which is not defined; the loss in `model_fused.py` is `fused_loss_with_masking`) |
| `exp04_one_genx` | single-layer regression on GenX simulations |
| `exp05_1layer_mask` | single layer, valid-q mask as a second input channel, instrument augmentation; GenX fit started from the CNN guess |
| `exp06_1lay_6out` | film + SiO2 native oxide (6 outputs) |
| `exp07_fourier_hires` | 3 outputs (film thickness, roughness, SLD) with the SiO2 layer randomized in the simulation, 2000-point q grid, Fourier features of q; `ablation.py` and `metrics_report.py` for component ablation and extended metrics |

## Structure

```
src/reflecto_exp/
  simulate/simul_genx.py      GenX-based XRR simulator (XRRSimulator, ParamSet)
  simulate/simul_refnx.py     refnx-based simulator
  simulate/noise.py           Poisson and background noise
  simulate/torch_engine/      PyTorch Abeles / kinematical reflectivity, smearing (not used by any run; copied into XRefine)
  physics_utils.py            unit conversions (tth2q, ...)
runs/expNN_*/                 experiments
notebooks/                    function tests
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
