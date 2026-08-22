import os
import sys
import subprocess
from pathlib import Path

# Resolve paths
project_root = Path.cwd().resolve()

def execute_notebook(nb_path):
    print(f"\nExecuting notebook: {nb_path} ...")
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=2400",  # 40 mins timeout
        str(nb_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Notebook {nb_path.name} executed successfully.")
        return True
    else:
        print(f"Failed to execute {nb_path.name}.")
        print("Stdout:\n", res.stdout)
        print("Stderr:\n", res.stderr)
        return False

# Execute CalIt2
calit2_nb = project_root / "notebooks_v4" / "CalIt2" / "CalIt2_SSM_Anomaly_Detection.ipynb"
execute_notebook(calit2_nb)

# Execute CICIDS
cicids_nb = project_root / "notebooks_v4" / "cicids" / "CICIDS_SSM_Anomaly_Detection.ipynb"
execute_notebook(cicids_nb)

# Execute Creditcard
creditcard_nb = project_root / "notebooks_v4" / "creditcard" / "Creditcard_SSM_Anomaly_Detection.ipynb"
execute_notebook(creditcard_nb)
