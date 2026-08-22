from pathlib import Path

logs_dir = Path("results/notebook_runs/logs")
for name in ["GHL_SSM_Anomaly_Detection.log", "MSL_SSM_Anomaly_Detection.log"]:
    log_path = logs_dir / name
    if log_path.exists():
        print(f"=== {name} (size: {log_path.stat().st_size} bytes) ===")
        text = log_path.read_text(encoding="utf-8")
        # Print first 5 and last 10 lines
        lines = text.splitlines()
        print("First 5 lines:")
        for l in lines[:5]:
            print("  ", l)
        if len(lines) > 5:
            print("...")
            print("Last 10 lines:")
            for l in lines[-10:]:
                print("  ", l)
    else:
        print(f"{name} does not exist.")
    print("\n" + "="*40 + "\n")
