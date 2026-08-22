import csv

csv_path = r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\results\notebook_runs\smap_evaluation.csv"

unsub_std_f1_list = []
unsub_pa_f1_list = []
oracle_std_f1_list = []
oracle_pa_f1_list = []

with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Ignore empty lines
        if not row or not row.get('channel'):
            continue
        unsub_std_f1_list.append(float(row['unsub_std_f1']))
        unsub_pa_f1_list.append(float(row['unsub_pa_f1']))
        oracle_std_f1_list.append(float(row['oracle_std_f1']))
        oracle_pa_f1_list.append(float(row['oracle_pa_f1']))

n = len(unsub_std_f1_list)
print(f"Number of channels: {n}")
print(f"Mean Unsub Std F1: {sum(unsub_std_f1_list)/n:.4f}")
print(f"Mean Unsub PA F1:  {sum(unsub_pa_f1_list)/n:.4f}")
print(f"Mean Oracle Std F1: {sum(oracle_std_f1_list)/n:.4f}")
print(f"Mean Oracle PA F1:  {sum(oracle_pa_f1_list)/n:.4f}")
