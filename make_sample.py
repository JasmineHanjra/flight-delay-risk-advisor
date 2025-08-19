import sys, pandas as pd

USECOLS = ["FL_DATE","MONTH","DAY_OF_WEEK","ORIGIN","DEST","OP_CARRIER","OP_UNIQUE_CARRIER","CARRIER",
           "CRS_DEP_TIME","DEP_TIME","DISTANCE","ARR_DEL15","ARR_DELAY","CANCELLED"]

def read_large_csv_sample(path, usecols, n_rows, seed=42):
    out, need = [], int(n_rows)
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=100_000, low_memory=False):
        if need <= 0: break
        take = min(len(chunk), need)
        out.append(chunk.sample(n=take, random_state=seed) if take < len(chunk) else chunk)
        need -= take
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python make_sample.py <input.csv> <output.csv> <rows>")
        sys.exit(1)
    inp, outp, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
    df = read_large_csv_sample(inp, USECOLS, rows)
    df.to_csv(outp, index=False)
    print(f"Wrote {len(df):,} rows to {outp}")
