import csv

def csv_to_md_table(csv_path):
    out = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Format headers nicely
        nice_headers = [h.replace('_', ' ').title() for h in headers]
        out.append("| " + " | ".join(nice_headers) + " |")
        out.append("| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |")
        for row in reader:
            if not row or not row[0]:
                continue
            # format floats to 4 decimals
            formatted_row = []
            for i, val in enumerate(row):
                if i == 0:
                    formatted_row.append(f"**{val}**")
                else:
                    try:
                        f_val = float(val)
                        if f_val.is_integer() and i == 1:
                            formatted_row.append(f"{int(f_val)}")
                        else:
                            formatted_row.append(f"{f_val:.4f}")
                    except ValueError:
                        formatted_row.append(val)
            out.append("| " + " | ".join(formatted_row) + " |")
    return "\n".join(out)

print("### SMD Table")
print(csv_to_md_table(r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\results\notebook_runs\smd_evaluation.csv"))
print("\n### SMAP Table")
print(csv_to_md_table(r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\results\notebook_runs\smap_evaluation.csv"))
