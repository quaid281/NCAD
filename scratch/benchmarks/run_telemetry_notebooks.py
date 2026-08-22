import os
import sys
import subprocess
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path.cwd().resolve()
while not (project_root / 'mTSBench_data').exists() and project_root != project_root.parent:
    project_root = project_root.parent

def execute_notebook(nb_path):
    log_dir = project_root / "results" / "notebook_runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{nb_path.stem}.log"
    
    print(f"Starting {nb_path.name} (logging to results/notebook_runs/logs/{log_file.name})...", flush=True)
    start_time = time.time()
    
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=28800",  # 8 hours timeout
        str(nb_path)
    ]
    
    with open(log_file, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        
    elapsed = time.time() - start_time
    if res.returncode == 0:
        print(f"SUCCESS: {nb_path.name} finished in {elapsed:.2f}s.", flush=True)
        return True, nb_path.name, elapsed
    else:
        print(f"FAILURE: {nb_path.name} failed after {elapsed:.2f}s. Check logs/{log_file.name} for details.", flush=True)
        return False, nb_path.name, elapsed

def main():
    telemetry_notebooks = [
        project_root / "notebooks_v4" / "SMD" / "SMD_SSM_Anomaly_Detection.ipynb",
        project_root / "notebooks_v4" / "MSL" / "MSL_SSM_Anomaly_Detection.ipynb",
        project_root / "notebooks_v4" / "SMAP" / "SMAP_SSM_Anomaly_Detection.ipynb",
    ]
    
    print(f"Found {len(telemetry_notebooks)} Telemetry notebooks to run simultaneously.", flush=True)
    
    start_total = time.time()
    # Run concurrently using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(execute_notebook, nb): nb for nb in telemetry_notebooks if nb.exists()}
        
        for future in as_completed(futures):
            nb = futures[future]
            try:
                success, name, elapsed = future.result()
            except Exception as e:
                print(f"Notebook {nb.name} generated an exception: {e}", flush=True)
                
    print(f"All Telemetry notebooks finished in {time.time() - start_total:.2f}s.", flush=True)

if __name__ == "__main__":
    main()
