import json
import os

OUTPUT_PATH = "reflexion_baseline_stress_test.jsonl"

def simulate_downstream_pipeline():
    print("=== Simulating Downstream Evaluation Pipeline ===")
    
    if not os.path.exists(OUTPUT_PATH):
        print(f"[-] Error: {OUTPUT_PATH} not found. Run your test_file.py first!")
        return

    total_rows = 0
    strategy_counts = {"none": 0, "meta_reflection": 0, "iterative_prompting": 0}
    valid_drift_scores = 0
    errors_found = 0

    with open(OUTPUT_PATH, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[ERROR] Row {line_num}: Invalid JSON format. Details: {e}")
                errors_found += 1
                continue

            # 1. Validate Target Fields Presence
            required_keys = ["task_id", "iteration", "failure_type", "strategy_label", "rubric_drift_score"]
            for key in required_keys:
                if key not in data:
                    print(f"[ERROR] Row {line_num} ({data.get('task_id', 'UNKNOWN')}): Missing crucial key '{key}'")
                    errors_found += 1

            # 2. Verify Strategy Label (Target for Figure 3)
            label = data.get("strategy_label")
            if label in strategy_counts:
                strategy_counts[label] += 1
            else:
                print(f"[ERROR] Row {line_num}: Unexpected strategy_label '{label}'")
                errors_found += 1

            # 3. Verify Rubric Drift Score (Target for Figure 4)
            drift_score = data.get("rubric_drift_score")
            if drift_score is not None:
                if isinstance(drift_score, (int, float)):
                    valid_drift_scores += 1
                else:
                    print(f"[ERROR] Row {line_num}: 'rubric_drift_score' must be numeric, got {type(drift_score)}")
                    errors_found += 1
            else:
                print(f"[ERROR] Row {line_num}: 'rubric_drift_score' is None or missing.")
                errors_found += 1

    print("\n--- Pipeline Validation Metrics Summary ---")
    print(f"Total Logged Trace Rows Parsed: {total_rows}")
    print(f"Successfully Parsed Strategy Labels (Fig 3): {strategy_counts}")
    print(f"Successfully Parsed Numeric Drift Scores (Fig 4): {valid_drift_scores}/{total_rows}")
    
    print("\n--- Final Health Check Status ---")
    if errors_found == 0:
        print("SUCCESS: No integration bugs found. The data file is flawless and ready for plotting!")
    else:
        print(f" FAILURE: Found {errors_found} formatting or tracking anomalies that will break downstream scripts.")

if __name__ == "__main__":
    simulate_downstream_pipeline()