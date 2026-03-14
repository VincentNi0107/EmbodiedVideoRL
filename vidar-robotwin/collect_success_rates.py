#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Optional

RATE_RE = re.compile(r"(?P<rate>\d+(?:\.\d+)?)")


def extract_success_rate(text: str) -> Optional[float]:
    """
    Extract success rate from _result.txt.
    Prefer a line that only contains a number (e.g., "1.0").
    Fall back to the last float-like number if needed.
    """
    last_number = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Timestamp:") or line.startswith("Instruction Type:"):
            continue
        if RATE_RE.fullmatch(line):
            try:
                return float(line)
            except ValueError:
                pass
        m = RATE_RE.search(line)
        if m:
            try:
                last_number = float(m.group("rate"))
            except ValueError:
                continue
    return last_number


def collect_results(base_dir: str):
    """
    Walk eval_result/ar/<method>/<task>/_result.txt
    and yield (method, task, rate, file_path)
    """
    results = []
    for root, _, files in os.walk(base_dir):
        if "_result.txt" in files:
            file_path = os.path.join(root, "_result.txt")
            rel = os.path.relpath(root, base_dir)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue
            method, task = parts[0], parts[1]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                rate = None
            else:
                rate = extract_success_rate(content)
            results.append((method, task, rate, file_path))
    return results


def write_csv(rows, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "success_rates.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "success_rate", "source"])
        for task, rate, file_path in rows:
            writer.writerow([task, "" if rate is None else rate, file_path])
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect task success rates from eval_result/ar/<method>/*/_result.txt"
    )
    parser.add_argument(
        "--base",
        default=os.path.join("eval_result", "ar"),
        help="Base directory containing method folders (default: eval_result/ar)",
    )
    args = parser.parse_args()

    base_dir = args.base
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Base directory not found: {base_dir}")

    results = collect_results(base_dir)
    if not results:
        raise SystemExit("No _result.txt files found.")

    by_method = {}
    for method, task, rate, file_path in results:
        by_method.setdefault(method, []).append((task, rate, file_path))

    for method, rows in sorted(by_method.items()):
        output_dir = os.path.join(base_dir, method)
        out_path = write_csv(rows, output_dir)
        print(f"Wrote {out_path} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
