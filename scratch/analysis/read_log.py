from pathlib import Path

log_path = Path(r"C:\Users\andre\.gemini\antigravity\brain\7626b434-c6d7-4ea1-9fbe-7a2731d67cfc\.system_generated\tasks\task-3314.log")
if log_path.exists():
    print("=== task-3314.log ===")
    print(log_path.read_text(encoding="utf-8"))
else:
    print("task-3314.log does not exist on disk yet.")
