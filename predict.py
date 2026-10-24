import os
import sys
import subprocess

if __name__ == "__main__":
    artvision_dir = os.path.join(os.path.dirname(__file__), "artvision")
    predict_script = os.path.join(artvision_dir, "predict.py")
    python_bin = os.path.join(artvision_dir, "venv/bin/python")
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    cmd = [python_bin, predict_script] + sys.argv[1:]
    result = subprocess.run(cmd, cwd=artvision_dir)
    sys.exit(result.returncode)
