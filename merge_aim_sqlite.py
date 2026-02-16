"""Merge Repo2's run_metadata.sqlite into Repo1's run_metadata.sqlite.

- Experiments: insert-or-skip by name, remap IDs
- Runs: insert with remapped experiment_id, new auto-increment id
"""
import sqlite3
import sys

REPO1_DB = "/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Aim/.aim/run_metadata.sqlite"
REPO2_DB = "/vol/data/OctreeNCA_Video/train_scripts/<path>/<path>/octree_study_new/Aim/.aim/run_metadata.sqlite"


def merge():
    dst = sqlite3.connect(REPO1_DB)
    src = sqlite3.connect(REPO2_DB)
    dst.row_factory = sqlite3.Row
    src.row_factory = sqlite3.Row

    # ---- Experiments ----
    # Build name -> id map for destination
    dst_experiments = {
        row["name"]: row["id"]
        for row in dst.execute("SELECT id, name FROM experiment")
    }

    src_experiments = src.execute(
        "SELECT id, uuid, name, created_at, updated_at, is_archived, description FROM experiment"
    ).fetchall()

    exp_id_map = {}  # src_id -> dst_id
    for exp in src_experiments:
        src_id = exp["id"]
        name = exp["name"]
        if name in dst_experiments:
            # Experiment already exists in destination
            exp_id_map[src_id] = dst_experiments[name]
            print(f"  Experiment '{name}': already exists (dst id={dst_experiments[name]})")
        else:
            cursor = dst.execute(
                "INSERT INTO experiment (uuid, name, created_at, updated_at, is_archived, description) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exp["uuid"], name, exp["created_at"], exp["updated_at"], exp["is_archived"], exp["description"]),
            )
            new_id = cursor.lastrowid
            exp_id_map[src_id] = new_id
            dst_experiments[name] = new_id
            print(f"  Experiment '{name}': inserted (dst id={new_id})")

    # ---- Runs ----
    dst_hashes = {
        row[0] for row in dst.execute("SELECT hash FROM run").fetchall()
    }

    src_runs = src.execute(
        "SELECT hash, name, description, is_archived, created_at, updated_at, experiment_id, finalized_at FROM run"
    ).fetchall()

    inserted = 0
    skipped = 0
    for run in src_runs:
        if run["hash"] in dst_hashes:
            print(f"  Run {run['hash']}: already exists, skipping")
            skipped += 1
            continue
        new_exp_id = exp_id_map.get(run["experiment_id"], run["experiment_id"])
        dst.execute(
            "INSERT INTO run (hash, name, description, is_archived, created_at, updated_at, experiment_id, finalized_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run["hash"], run["name"], run["description"], run["is_archived"],
             run["created_at"], run["updated_at"], new_exp_id, run["finalized_at"]),
        )
        inserted += 1
        print(f"  Run {run['hash']}: inserted (experiment_id {run['experiment_id']} -> {new_exp_id})")

    dst.commit()
    print(f"\nDone: {inserted} runs inserted, {skipped} skipped.")

    # Final count
    total = dst.execute("SELECT COUNT(*) FROM run").fetchone()[0]
    print(f"Destination now has {total} runs total.")

    dst.close()
    src.close()


if __name__ == "__main__":
    merge()
