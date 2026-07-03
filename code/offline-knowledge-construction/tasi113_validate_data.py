#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sqlite3
import time
import multiprocessing as mp

# ==============================
# CONFIG
# ==============================
INPUT_FILE  = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/data.json"
OUTPUT_FILE = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/validate-iter-01/validate-iter-01.json"
ERROR_LOG   = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/validate-iter-01/error-log-validate-iter-01.jsonl"
DB_FOLDER = "../../../dataset/train/train/train_databases"

TIMEOUT_SEC = 3

# ==============================
# SETUP
# ==============================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def run_query(db_path, sql, return_dict):
    conn = None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        start = time.time()

        cursor.execute(sql)
        cursor.fetchone()

        exec_time = time.time() - start

        return_dict["result"] = (
            True,
            {
                "execution_time": exec_time
            }
        )

    except sqlite3.Error as e:
        return_dict["result"] = (
            False,
            {
                "type": type(e).__name__,
                "sqlite_errorcode": getattr(e, "sqlite_errorcode", None),
                "sqlite_errorname": getattr(e, "sqlite_errorname", None),
                "message": str(e)
            }
        )

    except Exception as e:
        return_dict["result"] = (
            False,
            {
                "type": type(e).__name__,
                "message": str(e)
            }
        )

    finally:
        if conn:
            conn.close()


def execute_sql(sql: str, db_id: str):
    db_path = os.path.join(DB_FOLDER, db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        return False, {
            "type": "DB_ERROR",
            "message": "database file not found"
        }

    sql_clean = sql.strip().rstrip(";")

    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(
        target=run_query,
        args=(db_path, sql_clean, return_dict)
    )

    p.start()
    p.join(TIMEOUT_SEC)

    if p.is_alive():
        p.terminate()
        p.join()

        return False, {
            "type": "TIMEOUT",
            "message": "execution exceeded time limit"
        }

    return return_dict.get(
        "result",
        (
            False,
            {
                "type": "UNKNOWN_ERROR",
                "message": "no result returned"
            }
        )
    )


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    print("=" * 70)
    print("🛡️ VALIDATION (FULL DATASET)")
    print("=" * 70)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    total_data = len(raw_data)

    print(f"📦 Total dataset : {total_data:,}")
    print("📊 Processing    : ALL DATA")
    print()

    valid_data = []
    error_data = []

    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "timeout": 0
    }

    start_time = time.time()
    last_log_time = time.time()

    LOG_INTERVAL = 5.0

    for i, item in enumerate(raw_data):
        stats["total"] += 1

        db_id = item.get("db_id")
        sql = item.get("sql")

        if not sql or not db_id:
            item["status"] = "rejected"
            item["validation_error"] = "MISSING_FIELD"

            item["error_detail"] = {
                "type": "MISSING_FIELD",
                "message": "sql or db_id is empty"
            }

            error_data.append(item)
            stats["invalid"] += 1
            continue

        is_valid, result = execute_sql(sql, db_id)

        if is_valid:
            item["status"] = "validated"
            item["execution_time"] = result["execution_time"]

            item.pop("validation_error", None)
            item.pop("error_detail", None)

            valid_data.append(item)
            stats["valid"] += 1

        else:
            item["status"] = "rejected"
            item["validation_error"] = result["type"]
            item["error_detail"] = result

            error_data.append(item)
            stats["invalid"] += 1

            if result.get("type") == "TIMEOUT":
                stats["timeout"] += 1

        now = time.time()

        if (now - last_log_time >= LOG_INTERVAL) or (i + 1 == total_data):
            elapsed = now - start_time

            rate = (
                (i + 1) / elapsed
                if elapsed > 0
                else 0
            )

            pct = ((i + 1) / total_data) * 100

            print(
                f"\rProgress: {i+1}/{total_data} "
                f"({pct:.1f}%) | "
                f"Rate: {rate:.2f}/s | "
                f"Valid: {stats['valid']} | "
                f"Invalid: {stats['invalid']} | "
                f"Timeout: {stats['timeout']}",
                end="",
                flush=True
            )

            last_log_time = now

    print("\n\n💾 Saving results...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            valid_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"✅ Saved VALID   : "
        f"{len(valid_data):,} -> {OUTPUT_FILE}"
    )

    if error_data:
        with open(ERROR_LOG, "w", encoding="utf-8") as f:
            json.dump(
                error_data,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"❌ Saved INVALID : "
            f"{len(error_data):,} -> {ERROR_LOG}"
        )

    total_time = (time.time() - start_time) / 60

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"⏱ Total Time      : {total_time:.1f} minutes")
    print(f"📊 Total Processed: {stats['total']:,}")
    print(f"✅ Valid          : {stats['valid']:,}")
    print(f"❌ Invalid        : {stats['invalid']:,}")
    print(f"⏱ Timeout        : {stats['timeout']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()