from pathlib import Path

from fitting_engine import GenXFitter
from inference import XRRInferenceEngine

from reflecto_exp.io_utils import load_xrr_dat
from reflecto_exp.physics_utils import tth2q
from reflecto_exp.simulate.simul_genx import ParamSet


def run_xrr_analysis(
    data_path: str | Path, weights_dir: str | Path, inference_engine: XRRInferenceEngine | None = None, verbose: bool = True, show_plot: bool = True
) -> dict:
    """
    Run the [load -> NN inference -> GenX fit] pipeline on one XRR data file.

    Args:
        data_path (str | Path): path of the .dat file to analyze
        weights_dir (str | Path): folder of the trained NN model
        inference_engine (XRRInferenceEngine, optional):
            Already loaded engine. Loaded inside the function when None.
            (Passing a preloaded engine is much faster for repeated calls.)
        verbose (bool): print progress
        show_plot (bool): show the result plot

    Returns:
        dict: analysis result {
            "nn_preds": (d, sig, sld),      # NN prediction
            "final_params": dict,           # final GenX fit parameters
            "fitter": GenXFitter,           # fitter object (holds the simulated data)
            "q": np.array,                  # q data
            "R_measured": np.array,         # measured R
            "R_fit": np.array               # fitted R
        }
    """
    data_path = Path(data_path)
    weights_dir = Path(weights_dir)

    if verbose:
        print("\n" + "=" * 60)
        print(f"🚀 XRR Analysis Pipeline: {data_path.name}")
        print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(f"[Error] 데이터 파일을 찾을 수 없습니다: {data_path}")

    tths, R_raw = load_xrr_dat(data_path)

    q_raw = tth2q(tths)

    if verbose:
        print(f"[Data] Loaded {len(q_raw)} points.")

    if inference_engine is None:
        if verbose:
            print("[Init] Loading Inference Engine...")
        inference_engine = XRRInferenceEngine(exp_dir=weights_dir)

    if verbose:
        print("[Step 1] Neural Network Inference...")
    preds = inference_engine.predict(q_raw, R_raw)
    pred_f_d, pred_f_sig, pred_f_sld = preds[0], preds[1], preds[2]
    pred_s_d, pred_s_sig, pred_s_sld = preds[3], preds[4], preds[5]

    if verbose:
        print("   >>> NN Prediction:")
        print(f"       [Film] Thickness: {pred_f_d:.2f}, Rough: {pred_f_sig:.2f}, SLD: {pred_f_sld:.3f}")
        print(f"       [SiO2] Thickness: {pred_s_d:.2f}, Rough: {pred_s_sig:.2f}, SLD: {pred_s_sld:.3f}")

    film_params = ParamSet(pred_f_d, pred_f_sig, pred_f_sld)
    sio2_params = ParamSet(pred_s_d, pred_s_sig, pred_s_sld)

    if verbose:
        print("\n[Step 2] GenX Fitting (Optimization)...")

    fitter = GenXFitter(q_raw, R_raw, film_params, sio2_params)

    final_results = fitter.run(verbose=verbose)

    if verbose:
        print("\n" + "-" * 40)
        print("FINAL ANALYSIS RESULT")
        print("-" * 40)
        for param_name, value in final_results.items():
            print(f"{param_name:15s}: {value:.4f}")
        print("=" * 40)

    if show_plot:
        fitter.plot()

    return {"nn_preds": preds, "final_params": final_results, "fitter": fitter, "q": q_raw, "R_measured": R_raw, "R_fit": fitter.model.data[0].y_sim}


def main():
    target_file = Path(r"C:\Users\IsaacYong\Documents\카카오톡 받은 파일\#1.dat")
    weights_path = Path(r"D:\data\XRR_AI\one_layer\test")

    try:
        result = run_xrr_analysis(data_path=target_file, weights_dir=weights_path, verbose=True, show_plot=True)

        print(f"최종 피팅된 두께: {result['final_params']['f_d']:.2f} Å")

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
