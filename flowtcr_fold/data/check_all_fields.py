import pandas as pd

data = pd.read_csv(r'C:\Users\zwl62\OneDrive - National University of Singapore\Desktop\TCR\TCR\flowtcr_fold\data\trn_seq.csv')

print("Checking all fields for 'name but no sequence' issue:")
print("=" * 60)

for field in ['h_v', 'h_j', 'l_v', 'l_j']:
    field_seq = f'{field}_seq'

    # Rows with name but no sequence
    problematic = data[
        (data[field].notna()) &
        (data[field].astype(str).str.strip() != '') &
        ((data[field_seq].isna()) | (data[field_seq].astype(str).str.strip() == ''))
    ]

    print(f"\n{field}: {len(problematic)} rows with name but no sequence")

    if len(problematic) > 0:
        print(f"  Sample values in {field}:")
        for i, val in enumerate(problematic[field].head(3)):
            print(f"    [{i+1}] {str(val)[:80]}...")
