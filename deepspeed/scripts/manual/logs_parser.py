import re                # For extracting trial results using regular expressions
import argparse          # For parsing command-line arguments
import pandas as pd      # For storing results in a structured DataFrame
from pathlib import Path # For safe and convenient file path handling


def parse_hpo_logs(log_file: str) -> pd.DataFrame:
    """
    Generic parser for manual HPO log files.

    Parameters
    ----------
    log_file : str
        Path to the log file.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed trial results.
        Columns may include (depending on log):
        ['tag', 'lr', 'bs', 'wd', 'eval_loss', 'runtime_s', ...]
    """

    # Ensure the provided log file exists
    log_file = Path(log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    # Read the entire log file as a single string
    with open(log_file, "r") as f:
        text = f.read()

    # Regex pattern to capture the lines like:
    # saved: {'tag': ..., 'lr': ..., 'bs': ..., 'wd': ..., 'eval_loss': ..., 'runtime_s': ...}
    # The pattern extracts everything between the curly braces { ... }
    pattern = r"saved:\s*{([^}]+)}"
    matches = re.findall(pattern, text)

    results = []  # List to store parsed trial dictionaries

    for match in matches:
        # Normalize key names for safe eval():
        # Converts `tag: 'lr1e-05_bs1_wd0.0'` → `"tag": 'lr1e-05_bs1_wd0.0'`
        formatted = re.sub(r"(\w+):", r'"\1":', match)

        try:
            # Evaluate the string as a Python dictionary and append it to results
            trial_dict = eval("{" + formatted + "}")
            results.append(trial_dict)
        except Exception as e:
            # Skip malformed entries gracefully
            print(f"Skipping malformed entry: {match[:50]}... Error: {e}")

    # If no valid trials were found, return an empty DataFrame
    if not results:
        print("⚠️ No trial results found in the log file.")
        return pd.DataFrame()

    # Return a Pandas DataFrame for easier filtering and export
    return pd.DataFrame(results)


if __name__ == "__main__":
    # ─────────────── CLI Argument Parser ───────────────
    parser = argparse.ArgumentParser(
        description="Parse manual HPO log files and export results as CSV."
    )

    # Required: Path to the log file to parse
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Path to the HPO log file (e.g., bloom_hpo_serial_5_epochs-39567495.out)"
    )

    # Optional: Name of the output CSV file (defaults to <log_file>.csv)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional: Path to save CSV (default: <log_file>.csv)"
    )

    args = parser.parse_args()

    # ─────────────── Parse & Save ───────────────
    df = parse_hpo_logs(args.logs)

    if not df.empty:
        # Default CSV name is the log file name without extension
        out_file = args.out if args.out else f"{Path(args.logs).stem}.csv"
        df.to_csv(out_file, index=False)
        print(f"✅ Parsed {len(df)} trials. Results saved to {out_file}")
    else:
        print("⚠️ No data parsed.")

