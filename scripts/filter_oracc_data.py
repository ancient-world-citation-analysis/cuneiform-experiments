"""
Filter finaldf.csv from Zenodo into Akkadian and Sumerian subsets.
Run: python filter_oracc_data.py

Expects finaldf.csv in the same directory.
Download from: https://zenodo.org/records/10794626/files/finaldf.csv
"""
import pandas as pd

print("Loading finaldf.csv (this may take a minute)...")
df = pd.read_csv("finaldf.csv", low_memory=False)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Find the language column — might be 'lang', 'language', or similar
lang_col = None
for col in df.columns:
    if 'lang' in col.lower():
        lang_col = col
        break

if lang_col is None:
    print("\nNo 'lang' column found. Columns are:")
    print(df.columns.tolist())
    print("\nShowing unique values in first few columns to help identify:")
    for col in df.columns[:30]:
        nunique = df[col].nunique()
        if nunique < 50:
            print(f"  {col}: {df[col].unique()[:20]}")
    exit()

print(f"\nUsing language column: '{lang_col}'")
print(f"Languages found:\n{df[lang_col].value_counts().head(20)}")

# Filter: keep rows with non-empty POS (lemmatized words only)
pos_col = 'pos' if 'pos' in df.columns else None
if pos_col:
    has_pos = df[pos_col].notna() & (df[pos_col] != '') & (df[pos_col] != 'NA')
else:
    has_pos = pd.Series(True, index=df.index)

# --- Akkadian ---
akk_mask = df[lang_col].str.contains('akk', na=False, case=False)
akk = df[akk_mask & has_pos]
print(f"\nAkkadian rows with POS: {len(akk)}")
akk.to_csv("alltexts_AKK.csv", index=False)
print("  Saved: alltexts_AKK.csv")

# --- Sumerian ---
sux_mask = df[lang_col].str.contains('sux', na=False, case=False)
sux = df[sux_mask & has_pos]
print(f"Sumerian rows with POS: {len(sux)}")
sux.to_csv("alltexts_SUX.csv", index=False)
print("  Saved: alltexts_SUX.csv")

# Quick stats
for name, subset in [("Akkadian", akk), ("Sumerian", sux)]:
    print(f"\n{name} summary:")
    print(f"  Unique texts: {subset['id_text'].nunique() if 'id_text' in subset.columns else 'N/A'}")
    print(f"  Unique forms: {subset['form'].nunique() if 'form' in subset.columns else 'N/A'}")
    if pos_col:
        print(f"  POS distribution:\n{subset[pos_col].value_counts().head(10).to_string()}")

print("\nDone! Upload alltexts_AKK.csv and alltexts_SUX.csv to Claude.")
