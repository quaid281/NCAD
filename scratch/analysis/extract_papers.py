import os, sys, glob, json

sys.stdout.reconfigure(encoding='utf-8')

try:
    import pypdf
    def extract_paper_info(pdf_path):
        try:
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            text = ""
            for page in reader.pages[:3]:
                text += page.extract_text() + "\n"
            return {"pages": num_pages, "text": text[:3000]}
        except Exception as e:
            return {"error": str(e)}
except Exception as e:
    print(f"Error importing pypdf: {e}")

folders = [
    r"C:\Users\andre\Downloads\Black-Kairos\docs\Papers",
    r"C:\Users\andre\Downloads\Black-Kairos\docs\Papers\JEPA"
]

results = {}
for folder in folders:
    results[folder] = {}
    for f in os.listdir(folder):
        if f.endswith('.pdf'):
            full_path = os.path.join(folder, f)
            results[folder][f] = extract_paper_info(full_path)

with open(r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\scratch\extracted_papers.json", "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=2, ensure_ascii=False)

print("Extraction complete. Extracted", sum(len(v) for v in results.values()), "papers.")
