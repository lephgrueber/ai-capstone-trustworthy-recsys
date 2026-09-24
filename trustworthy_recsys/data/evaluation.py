"""Export quantitative examples without changing the qualitative golden set."""

import csv
from itertools import groupby
import json
from pathlib import Path


def export_examples(raw, output, cutoffs, known_items, threshold):
    """Keep only one user's records in memory; source ordering is audited first."""
    handles, report = {}, {}
    for name in cutoffs:
        report[name] = {}
        for part in ("validation", "test"):
            report[name][part] = dict(users_with_events=0, excluded_no_positive_labels=0,
                                      all_examples=0, all_positive_labels=0, empty_positive_history_examples=0,
                                      warm_examples=0, warm_positive_labels=0,
                                      excluded_from_warm_no_training_user=0,
                                      excluded_from_warm_no_known_positive_items=0,
                                      positive_labels_excluded_from_warm=0)
            for population in ("all", "warm"):
                handles[name, part, population] = (Path(output)/name/f"{part}_{population}.jsonl").open("w", encoding="utf-8", newline="\n")
    try:
        with (Path(raw)/"ratings.csv").open(encoding="utf-8", newline="") as source:
            for user, rows in groupby(csv.DictReader(source), key=lambda row: row["userId"]):
                # Stable timestamp order with numeric movie ID as the tie breaker.
                events = sorted([(int(row["timestamp"]), int(row["movieId"]), float(row["rating"])) for row in rows])
                for name, (vstart, tstart) in cutoffs.items():
                    has_training = events[0][0] < vstart
                    history = [str(movie) for stamp, movie, rating in events if stamp < vstart and rating >= threshold]
                    for part, lo, hi in [("validation", vstart, tstart), ("test", tstart, None)]:
                        heldout = [(movie, rating) for stamp, movie, rating in events if stamp >= lo and (hi is None or stamp < hi)]
                        if not heldout:
                            continue
                        s = report[name][part]
                        s["users_with_events"] += 1
                        relevant = [str(movie) for movie, rating in heldout if rating >= threshold]
                        if not relevant:
                            s["excluded_no_positive_labels"] += 1
                            continue
                        record = {"user_id": str(int(user)), "history_items": history, "relevant_items": relevant}
                        handles[name, part, "all"].write(json.dumps(record, separators=(",", ":")) + "\n")
                        s["all_examples"] += 1
                        s["all_positive_labels"] += len(relevant)
                        s["empty_positive_history_examples"] += int(not history)
                        warm = [item for item in relevant if int(item) in known_items[name]] if has_training else []
                        s["positive_labels_excluded_from_warm"] += len(relevant)-len(warm)
                        if not has_training:
                            s["excluded_from_warm_no_training_user"] += 1
                        elif not warm:
                            s["excluded_from_warm_no_known_positive_items"] += 1
                        else:
                            record["relevant_items"] = warm
                            handles[name, part, "warm"].write(json.dumps(record, separators=(",", ":")) + "\n")
                            s["warm_examples"] += 1
                            s["warm_positive_labels"] += len(warm)
    finally:
        for handle in handles.values():
            handle.close()
    return report


def validate_exports(output, report):
    """Use the actual evaluator's loader so schema compatibility is checked directly."""
    from eval.harness import load_evaluation_examples
    for name, partitions in report.items():
        for part, counts in partitions.items():
            if counts["users_with_events"] != counts["all_examples"] + counts["excluded_no_positive_labels"]:
                raise ValueError("Evaluation eligibility counts do not reconcile")
            if counts["all_examples"] != counts["warm_examples"] + counts["excluded_from_warm_no_training_user"] + counts["excluded_from_warm_no_known_positive_items"]:
                raise ValueError("Warm eligibility counts do not reconcile")
            for population in ("all", "warm"):
                path = Path(output)/name/f"{part}_{population}.jsonl"
                expected = counts[f"{population}_examples"]
                if not expected:
                    if path.stat().st_size:
                        raise ValueError("Expected an empty population file")
                    # The harness intentionally rejects empty files; record and do not score them.
                    continue
                examples = load_evaluation_examples(path)
                if len(examples) != expected:
                    raise ValueError("Harness-loaded example count differs from report")
                for example in examples:
                    if set(example.history_items) & set(example.relevant_items):
                        raise ValueError("A historical movie leaked into held-out relevance")
                    if len(set(example.relevant_items)) != len(example.relevant_items):
                        raise ValueError("Repeated relevance label")
                del examples
