import argparse
import json
import math


def score_record(record, sigma):
    if record.get("status") == "failed_after_output_limit":
        return 0.0

    spatial_raw = record.get("spatial_raw")
    if not spatial_raw:
        return None

    items = spatial_raw.get("items", [])
    if not items:
        return None

    scores = []
    for item in items:
        if item.get("is_correct"):
            scores.append(100.0)
        else:
            dist = float(item["dist_abs_x"])
            scores.append(100.0 * math.exp(- (dist ** 2) / (2 * sigma ** 2)))
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description="Recompute spatial scores from raw JSONL records.")
    parser.add_argument("jsonl_path", help="Path to spatial_raw_records.jsonl")
    parser.add_argument("--sigma", type=float, required=True, help="Gaussian smoothing sigma")
    args = parser.parse_args()

    total_score = 0.0
    valid_count = 0
    skipped_count = 0

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            score = score_record(record, args.sigma)
            if score is None:
                skipped_count += 1
                continue
            total_score += score
            valid_count += 1

    average_score = total_score / valid_count if valid_count else 0.0
    print(f"sigma: {args.sigma}")
    print(f"valid_records: {valid_count}")
    print(f"skipped_records: {skipped_count}")
    print(f"average_score: {average_score}")


if __name__ == "__main__":
    main()
