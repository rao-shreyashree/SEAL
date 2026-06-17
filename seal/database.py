import sqlite3
import json
from pathlib import Path

DB_PATH = Path("results/seal_results.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_results (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id                     TEXT NOT NULL,
            iteration                   INTEGER NOT NULL,
            strategy_used               TEXT,
            failure_type                TEXT NOT NULL,
            score                       REAL NOT NULL,
            success                     INTEGER NOT NULL,
            rubric_version              INTEGER NOT NULL,
            rubric_hash                 TEXT,
            raw_trace                   TEXT,           -- JSON-encoded list
            task_description            TEXT,
            oracle_failure_type         TEXT,
            agent_confidence            REAL,
            plan_coherence              REAL,
            total_steps                 INTEGER,
            judge_score                 REAL,
            judge_failure_type          TEXT,
            judge_explanation           TEXT,
            rubric_drift_score          REAL,
            stagnation_step_count       INTEGER,
            trajectory_stagnation_rate  REAL,
            unique_action_count         INTEGER,
            action_density_index        REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_task_result(result):
    """Accepts a TaskResult instance (from task_result.py)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    d = result.to_dict()

    cursor.execute("""
        INSERT INTO task_results (
            task_id, iteration, strategy_used, failure_type, score, success,
            rubric_version, rubric_hash, raw_trace, task_description,
            oracle_failure_type, agent_confidence, plan_coherence, total_steps,
            judge_score, judge_failure_type, judge_explanation, rubric_drift_score,
            stagnation_step_count, trajectory_stagnation_rate,
            unique_action_count, action_density_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        d["task_id"], d["iteration"], d["strategy_used"], d["failure_type"],
        d["score"], int(d["success"]), d["rubric_version"], d["rubric_hash"],
        json.dumps(d["raw_trace"]), d["task_description"], d["oracle_failure_type"],
        d["agent_confidence"], d["plan_coherence"], d["total_steps"],
        d["judge_score"], d["judge_failure_type"], d["judge_explanation"], d["rubric_drift_score"],
        d["stagnation_step_count"], d["trajectory_stagnation_rate"],
        d["unique_action_count"], d["action_density_index"]
    ))

    conn.commit()
    conn.close()