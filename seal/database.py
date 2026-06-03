import sqlite3
from pathlib import Path

DB_PATH = Path("results/seal_results.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id         TEXT NOT NULL,
            task_description TEXT,
            success         INTEGER NOT NULL,       -- 0 or 1
            failure_type    TEXT NOT NULL,
            score           REAL NOT NULL,
            explanation     TEXT,
            iteration       INTEGER NOT NULL,
            strategy_used   TEXT,
            rubric_version  INTEGER NOT NULL,
            timestamp       TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def insert_task_result(result):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO task_results (
            task_id, task_description, success, failure_type,
            score, explanation, iteration, strategy_used,
            rubric_version, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.task_id,
        result.task_description,
        int(result.success),
        result.failure_type.value,
        result.score,
        result.explanation,
        result.iteration,
        result.strategy_used,
        result.rubric_version,
        result.timestamp
    ))
    
    conn.commit()
    conn.close()