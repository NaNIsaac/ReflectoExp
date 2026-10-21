from pathlib import Path

from fitting_engine import GenXFitter
from inference import XRRInferenceEngine

from reflecto_exp.io_utils import load_xrr_dat
from reflecto_exp.physics_utils import tth2q
from reflecto_exp.simulate.simul_genx import ParamSet


def run_xrr_analysis(
    data_path: str | Path,
    weights_dir: str | Path,
    inference_engine: XRRInferenceEngine | None = None,
    verbose: bool = True,
    show_plot: bool = True
) -> dict:
    """
    단일 XRR 데이터 파일에 대해 [로드 -> NN 추론 -> GenX 피팅] 파이프라인을 실행합니다.

    Args:
        data_path (str | Path): 분석할 .dat 파일 경로
        weights_dir (str | Path): 학습된 NN 모델 폴더 경로
        inference_engine (XRRInferenceEngine, optional):
            이미 로드된 추론 엔진 객체. None이면 함수 내부에서 새로 로드합니다.
            (반복 호출 시 엔진을 미리 로드해서 넘겨주면 속도가 훨씬 빠릅니다)
        verbose (bool): 진행 상황 출력 여부
        show_plot (bool): 결과 그래프 표시 여부

    Returns:
        dict: 분석 결과 {
            "nn_preds": (d, sig, sld),      # NN 예측값
            "final_params": dict,           # GenX 최종 피팅 파라미터
            "fitter": GenXFitter 객체,      # 피팅 객체 (시뮬레이션 데이터 포함)
            "q": np.array,                  # q 데이터
            "R_measured": np.array,         # 측정된 R
            "R_fit": np.array               # 피팅된 R
        }
    """
    data_path = Path(data_path)
    weights_dir = Path(weights_dir)

    if verbose:
        print("\n" + "="*60)
        print(f"🚀 XRR Analysis Pipeline: {data_path.name}")
        print("="*60)

    # ---------------------------------------------------------
    # 1. 데이터 로드 (Data Loading)
    # ---------------------------------------------------------
    if not data_path.exists():
        raise FileNotFoundError(f"[Error] 데이터 파일을 찾을 수 없습니다: {data_path}")

    # Pandas Series 문제 방지를 위해 np.array로 명시적 변환
    tths, R_raw = load_xrr_dat(data_path)

    # tth -> q 변환
    q_raw = tth2q(tths)

    if verbose:
        print(f"[Data] Loaded {len(q_raw)} points.")

    # ---------------------------------------------------------
    # 2. NN 초기값 예측 (Neural Network Inference)
    # ---------------------------------------------------------
    # 엔진이 주입되지 않았으면 새로 로드 (단발성 실행용)
    if inference_engine is None:
        if verbose:
            print("[Init] Loading Inference Engine...")
        inference_engine = XRRInferenceEngine(exp_dir=weights_dir)

    if verbose:
        print("[Step 1] Neural Network Inference...")

    pred_d, pred_sig, pred_sld = inference_engine.predict(q_raw, R_raw)

    if verbose:
        print("   >>> NN Prediction:")
        print(f"       Thickness : {pred_d:.2f} Å")
        print(f"       Roughness : {pred_sig:.2f} Å")
        print(f"       SLD       : {pred_sld:.3f} (10⁻⁶ Å⁻²)")

    # GenXFitter용 파라미터 객체 생성
    nn_initial_params = ParamSet(pred_d, pred_sig, pred_sld)

    # ---------------------------------------------------------
    # 3. GenX 정밀 피팅 (GenX Refinement)
    # ---------------------------------------------------------
    if verbose:
        print("\n[Step 2] GenX Fitting (Optimization)...")

    fitter = GenXFitter(q_raw, R_raw, nn_initial_params)

    # 피팅 실행
    final_results = fitter.run(verbose=verbose)

    # ---------------------------------------------------------
    # 4. 결과 정리 및 시각화
    # ---------------------------------------------------------
    if verbose:
        print("\n" + "-"*40)
        print("FINAL ANALYSIS RESULT")
        print("-"*40)
        for param_name, value in final_results.items():
            print(f"{param_name:15s}: {value:.4f}")
        print("="*40)

    if show_plot:
        fitter.plot()

    # 결과 반환
    return {
        "nn_preds": (pred_d, pred_sig, pred_sld),
        "final_params": final_results,
        "fitter": fitter,
        "q": q_raw,
        "R_measured": R_raw,
        "R_fit": fitter.model.data[0].y_sim
    }

# =========================================================
# 사용 예시 (Main)
# =========================================================
def main():
    # 설정
    target_file = Path(r"C:\Users\IsaacYong\Documents\카카오톡 받은 파일\#1.dat")
    weights_path = Path(r"D:\data\XRR_AI\one_layer\test")

    try:
        # 함수 호출
        result = run_xrr_analysis(
            data_path=target_file,
            weights_dir=weights_path,
            verbose=True,
            show_plot=True
        )

        # 결과 데이터 활용 예시
        print(f"최종 피팅된 두께: {result['final_params']['f_d']:.2f} Å")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
