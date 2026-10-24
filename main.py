import argparse
import platform
import subprocess
import sys
from pathlib import Path


def get_venv_python() -> Path:
    base_path = Path.cwd()

    if platform.system() == "Windows":
        venv_python = base_path / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = base_path / ".venv" / "bin" / "python"

    if not venv_python.exists():
        print("[Error] 가상환경 Python을 찾을 수 없습니다.")
        print(f"  경로 확인: {venv_python}")
        sys.exit(1)

    return venv_python


def find_exp_folder(exp_num: int, runs_dir_path: Path = Path("./runs")) -> Path:
    """Folder under runs/ whose name starts with exp{num}_ or exp0{num}_."""
    if not runs_dir_path.exists():
        print(f"❌ [Error] '{runs_dir_path}' 디렉토리가 존재하지 않습니다.")
        sys.exit(1)

    target_prefixes = [f"exp{exp_num}_", f"exp{exp_num:02d}_"]

    found_folder = None
    for item in runs_dir_path.iterdir():
        if item.is_dir():
            if any(item.name.startswith(prefix) for prefix in target_prefixes):
                found_folder = item
                break

    if found_folder is None:
        print(f"[Error] 실험 번호 {exp_num}번에 해당하는 폴더를 못 찾았습니다.")
        print(f"   탐색 위치: {runs_dir_path.resolve()}")
        sys.exit(1)

    return found_folder


def run_main(target_folder: Path):
    venv_python = get_venv_python()
    target_script = target_folder / "main.py"

    if not target_script.exists():
        print(f"[Error] '{target_folder.name}' 폴더 안에 'main.py'가 없습니다.")
        sys.exit(1)

    print(f"[Start] Experiment: {target_folder.name}")
    print(f"   Path: {target_folder}")
    print("-" * 50)

    try:
        subprocess.run(
            [str(venv_python), "main.py"],
            cwd=target_folder,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"💥 [Fail] 실행 중 오류 발생 (Exit code: {e.returncode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment based on ID")
    parser.add_argument("exp_num", type=int, help="Experiment number (e.g., 5)")

    args = parser.parse_args()

    target_folder = find_exp_folder(args.exp_num)

    run_main(target_folder)
