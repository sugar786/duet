import os
import glob
import tarfile
import tempfile
import argparse
import pandas as pd


def read_result_file(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)

    if path.endswith(".tar.gz"):
        dfs = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(tmpdir)

            for csv_file in glob.glob(os.path.join(tmpdir, "**", "*.csv"), recursive=True):
                dfs.append(pd.read_csv(csv_file))

        if len(dfs) > 0:
            return pd.concat(dfs, ignore_index=True)

    return None


def parse_number_from_text(text, keys):
    if pd.isna(text):
        return "Unknown"

    text = str(text)

    for key in keys:
        for marker in [f"'{key}':", f'"{key}":']:
            if marker in text:
                try:
                    return int(text.split(marker)[1].split(",")[0].strip().strip("}"))
                except Exception:
                    pass

    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="duet_summary.csv")
    args = parser.parse_args()

    result_files = []
    result_files.extend(glob.glob(os.path.join(args.result_dir, "**", "*.csv"), recursive=True))
    result_files.extend(glob.glob(os.path.join(args.result_dir, "**", "*.tar.gz"), recursive=True))

    if len(result_files) == 0:
        raise FileNotFoundError(f"No result files found in {args.result_dir}")

    all_rows = []

    for file in result_files:
        df = read_result_file(file)
        if df is None or len(df) == 0:
            continue

        for _, row in df.iterrows():
            log_info = row.get("log_info", "")
            if pd.notna(log_info) and str(log_info).strip().lower() not in ["", "nan", "none"]:
                continue

            dataset = row.get("file_name", "Unknown")

            input_len = row.get("seq_len", "Unknown")
            if input_len == "Unknown" or pd.isna(input_len):
                input_len = parse_number_from_text(row.get("model_hyper_params", ""), ["seq_len"])

            pred_len = row.get("horizon", row.get("pred_len", "Unknown"))
            if pred_len == "Unknown" or pd.isna(pred_len):
                pred_len = parse_number_from_text(row.get("model_hyper_params", ""), ["horizon", "pred_len"])
            if pred_len == "Unknown" or pd.isna(pred_len):
                pred_len = parse_number_from_text(row.get("strategy_args", ""), ["horizon", "pred_len"])

            mse = row.get("mse", None)
            mae = row.get("mae", None)

            if pd.isna(mse) or pd.isna(mae):
                continue

            all_rows.append({
                "Dataset": dataset,
                "Input Length": input_len if input_len == "Unknown" else int(input_len),
                "Pred Length": pred_len if pred_len == "Unknown" else int(pred_len),
                "MSE": f"{float(mse):.3f}",
                "MAE": f"{float(mae):.3f}",
            })

    if len(all_rows) == 0:
        raise RuntimeError("No valid successful results found.")

    out = pd.DataFrame(all_rows)
    out = out.drop_duplicates()
    out = out.sort_values(["Dataset", "Input Length", "Pred Length"])

    out.to_csv(args.output, index=False)

    print("\n========== DUET RESULT SUMMARY ==========")
    print(out.to_string(index=False))
    print("=========================================")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
