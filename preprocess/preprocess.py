from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "raw_data.xlsx"
OUT_PATH = ROOT / "data" / "train" / "training_data.csv"
REQUIRED = ["age", "cuts", "emo"]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.strip)  # trim header spaces
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED].copy()
    df = df.dropna(subset=REQUIRED)

    # age: decimal in Excel → integer years for the model
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].round().astype(int)
    df["cuts"] = pd.to_numeric(df["cuts"], errors="coerce").fillna(0).astype(int)
    df["emo"] = pd.to_numeric(df["emo"], errors="coerce").astype(int)
  
    # optional sanity checks
    df = df[(df["age"] >= 0) & (df["cuts"] >= 0)]
    df = df[df["emo"].isin([0, 1])]

    return df.reset_index(drop=True)

def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Put raw data at {RAW_PATH}")

    df = pd.read_excel(RAW_PATH)
    clean_df = clean(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(clean_df)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
