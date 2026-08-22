import json, os, re, sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\scratch\extracted_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for folder, papers in data.items():
    folder_name = os.path.basename(folder)
    print(f"\n=================== FOLDER: {folder_name} ===================")
    for filename, info in papers.items():
        print(f"\n--- FILE: {filename} ---")
        if "error" in info:
            print(f"Error: {info['error']}")
            continue
        text = info.get("text", "")
        # Clean whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        snippet = "\n".join(lines[:25])
        print(snippet[:1500])
