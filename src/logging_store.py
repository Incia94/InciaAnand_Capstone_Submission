import sqlite3

connection = sqlite3.connect("./storage/decisions.db")
connection.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id     TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        ticket_id       TEXT NOT NULL,
        stage           TEXT NOT NULL,
        prediction      TEXT,
        confidence      REAL,
        threshold       REAL,
        action_taken    TEXT NOT NULL,
        reason          TEXT NOT NULL,
        sources_used    TEXT,
        guardrails      TEXT,
        prompt_version  TEXT,
        requirement_ids TEXT
    )
""")
connection.commit()