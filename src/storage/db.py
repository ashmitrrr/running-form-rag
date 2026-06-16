import lancedb
import pandas as pd
from datetime import datetime
from pathlib import Path

DB_PATH = "data/lancedb"
TABLE_NAME = "form_metrics"


def get_db():
    """Connect to (or create) the local LanceDB database."""
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(DB_PATH)


def ingest_session(form_df: pd.DataFrame, session_id: str, session_date: str = None):
    """
    Store a run's per-window form data into LanceDB.
    """
    if session_date is None:
        session_date = datetime.now().strftime("%Y-%m-%d")

    df = form_df.copy()
    df.insert(0, "session_id", session_id)
    df.insert(1, "session_date", session_date)

    db = get_db()
    existing_tables = db.list_tables()

    if TABLE_NAME in existing_tables:
        table = db.open_table(TABLE_NAME)
        existing = table.to_pandas()
        if session_id in existing["session_id"].values:
            print(f"Session '{session_id}' already in DB. Skipping. "
                  f"(Delete it first if you want to re-ingest.)")
            return table
        table.add(df)
        print(f"Appended {len(df)} windows for session '{session_id}'.")
    else:
        table = db.create_table(TABLE_NAME, data=df)
        print(f"Created table '{TABLE_NAME}' with {len(df)} windows from session '{session_id}'.")

    return table


def query_session(session_id: str = None):
    """Return all windows, optionally filtered to one session."""
    db = get_db()
    if TABLE_NAME not in db.list_tables():
        print("No data stored yet.")
        return pd.DataFrame()

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()
    if session_id:
        df = df[df["session_id"] == session_id]
    return df


def find_form_drops(session_id: str, threshold: float = 85.0):
    """
    The headline query: find windows where form score dropped below a threshold.
    This is exactly what the agent's analytics tool will call later.
    """
    df = query_session(session_id)
    if df.empty:
        return df
    drops = df[df["form_score"] < threshold]
    return drops[["session_id", "window", "t_start_sec", "form_score"]]


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from biomechanics.form_score import compute_form_score

    # compute form score for the on-spot clip, then ingest it
    form_df = compute_form_score(
        "data/processed/onspot_run_keypoints.parquet",
        fps=29.88,
        window_sec=5.0,
    )

    ingest_session(form_df, session_id="onspot_run_2026_06_17")

    # verify it's queryable
    print("\n--- All stored windows ---")
    stored = query_session()
    print(stored[["session_id", "window", "t_start_sec", "form_score"]].to_string(index=False))

    print("\n--- Form drops below 85 ---")
    drops = find_form_drops("onspot_run_2026_06_17", threshold=85.0)
    print(drops.to_string(index=False))