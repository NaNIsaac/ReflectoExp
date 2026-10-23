from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from engine.fitting import GenXFitter, XRRConfig

# 엔진 모듈
from engine.inference import XRRInferenceEngine

# 프로젝트 코어 유틸리티
from reflecto_exp.math_utils import i0_normalize
from reflecto_exp.simulate.simul_genx import ParamSet, param2refl


class XRRAnalyzer:
    """XRR 분석 및 테스트 통합 클래스 (AI + GenX)"""

    def __init__(self, exp_dir: Path, device: str = "cpu"):
        self.exp_dir = Path(exp_dir)
        self.engine = XRRInferenceEngine(self.exp_dir, device=device)
        print(f"🚀 XRRAnalyzer Initialized with model at: {self.exp_dir.name}")

    def generate_test_data(self, thick=350.0, rough=5.0, sld=15.0) -> tuple[np.ndarray, np.ndarray, ParamSet]:
        """[Test용] 직접 파라미터를 입력하여 시뮬레이션 데이터를 생성합니다."""
        print(f"🧪 Generating synthetic test data: d={thick}, sig={rough}, sld={sld}")

        # 모델이 학습된 표준 Grid 사용
        qs = self.engine.target_qs
        true_params = ParamSet(thickness=thick, roughness=rough, sld=sld)

        # 시뮬레이션 (약간의 노이즈 추가 가능)
        R_clean = param2refl(qs, [true_params])
        noise = np.random.normal(0, 0.02, size=len(R_clean))  # 가우시안 노이즈 시뮬레이션
        R_noisy = np.abs(R_clean * (1 + noise))

        return qs, R_noisy, true_params

    def analyze(
        self,
        q: np.ndarray,
        R: np.ndarray,
        run_fitting: bool = True,
        save_results: bool = False,
        filename: str = "analysis_result",
        true_params: ParamSet | None = None,
    ):
        """
        통합 분석 파이프라인
        - q, R: 입력 데이터
        - save_results: 파일 저장 여부 (옵션)
        - true_params: 테스트 데이터인 경우 Ground Truth 비교용
        """
        R = i0_normalize(R)

        # 1. AI Inference
        print("Step 1: AI Inference...")
        ai_guess = self.engine.predict(q, R)
        print(f"   [AI Guess] {ai_guess}")

        # 2. Plot AI Guess (요청하신 AI 단독 플롯)
        if save_results:
            save_path = self.exp_dir / f"guess_{filename}.png"
            self.engine.plot_ai_guess(q, R, ai_guess, save_path=save_path)
            print(f"   > AI Guess plot saved: {save_path.name}")

        final_params = ai_guess
        R_fit = param2refl(q, [ai_guess])

        # 3. GenX Refinement
        if run_fitting:
            print("Step 2: GenX Refinement (Fitting)...")
            fit_config = XRRConfig(thickness_search_ratio=(0.8, 1.2), steps_structure=400, steps_fine=600)
            fitter = GenXFitter(q, R, ai_guess, config=fit_config)
            res = fitter.run(verbose=False)

            final_params = ParamSet(res["set_f_d"], res["set_f_sig"], res["set_f_sld"])
            R_fit = res["R_sim"]
            print(f"   [Final Fit] {final_params}")

        # 4. Final Comparison Visualization
        self._visualize_final(q, R, ai_guess, final_params, R_fit, filename, save_results, true_params)

        return final_params

    def _visualize_final(self, q, R, ai_p, final_p, R_fit, filename, save_results, true_p):
        """결과 통합 시각화 및 리포트"""
        plt.figure(figsize=(10, 7))
        plt.plot(q, R, "ko", markersize=3, alpha=0.2, label="Measured Data")

        # AI Guess (파란 점선)
        R_ai = param2refl(q, [ai_p])
        plt.plot(q, R_ai, "b--", alpha=0.7, label=f"AI Guess (d={ai_p.thickness:.1f}Å)")

        # Final Fit (빨간 실선)
        plt.plot(q, R_fit, "r-", lw=2, label=f"Final Fit (d={final_p.thickness:.1f}Å)")

        # 만약 Ground Truth(테스트 데이터)가 있다면 표시
        if true_p:
            plt.axvline(x=0, color="green", label=f"GT: d={true_p.thickness:.1f}Å", alpha=0)
            print(f"🎯 Accuracy (Thick): {100 - abs(true_p.thickness - final_p.thickness) / true_p.thickness * 100:.2f}%")

        plt.yscale("log")
        plt.xlabel(r"$q$ ($\AA^{-1}$)")
        plt.ylabel("Reflectivity")
        plt.title(f"XRR Final Analysis: {filename}")
        plt.legend(loc="upper right")
        plt.grid(True, which="both", alpha=0.2)

        if save_results:
            save_path = self.exp_dir / f"final_{filename}.png"
            plt.savefig(save_path, dpi=150)
            print(f"📊 Final comparison plot saved: {save_path.name}")

        plt.show()


# -----------------------------------------------------------------------------
# Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 환경 설정
    EXP_PATH = Path(r"C:\Warehouse\data\XRR_AI\mask\exp07_fourier_hires")
    analyzer = XRRAnalyzer(EXP_PATH)

    # --- [MODE 1] 직접 데이터 생성해서 테스트 ---
    print("\n>>> Running Mode 1: Synthetic Data Test")
    q_syn, R_syn, gt_params = analyzer.generate_test_data(thick=425.5, rough=3.2, sld=22.1)
    analyzer.analyze(
        q_syn,
        R_syn,
        run_fitting=True,
        save_results=False,  # 테스트 시 저장 안 함
        filename="synthetic_test",
        true_params=gt_params,
    )

    # --- [MODE 2] 실측 파일 로드해서 분석 ---
    # print("\n>>> Running Mode 2: Real Data Analysis")
    # DATA_FILE = Path(r"C:\Warehouse\data\dat_files\jinhuan\#1_xrr.dat")
    # if DATA_FILE.exists():
    #     tth, R_raw = load_xrr_dat(DATA_FILE)
    #     q_real = tth2q(tth)
    #     analyzer.analyze(q_real, R_raw,
    #                      run_fitting=True,
    #                      save_results=True, # 결과 파일 저장
    #                      filename=DATA_FILE.stem)
