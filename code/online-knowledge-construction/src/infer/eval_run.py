# src/infer/eval_run.py
"""
Evaluation runner untuk arsitektur API-friendly (run.py / share_models.py baru).

Perbedaan dengan run_b2.py:
- run_b2.py memproses per-STAGE untuk SELURUH dataset sekaligus (batch besar per stage),
  lalu unload model antar stage.
- Script ini memanggil process_prompt() per-SAMPLE (karena arsitektur baru memuat
  SEMUA model sekaligus di VRAM dan mengekspos API per-pertanyaan, bukan per-stage).
  Konsekuensi: seluruh model (generator + BAM + SAM + LOM) harus muat bareng di GPU
  yang sama (LLM_DEVICE = SLM_DEVICE = "cuda:1" di run.py Anda saat ini).

Usage:
    python -m src.infer.eval_run \
        --data_config_path ../../../dataset/dev/data_config.json \
        --output_dir outputs/eval_api_v1 \
        --limit 50            # opsional, untuk smoke test
"""

import os
import sys
import json
import time
import re
import argparse
from tqdm import tqdm

BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

from src.infer.run import initialize_system, process_prompt  # noqa: E402


# ──────────────────────────────────────────────
# Schema builder (setara SAM._generate_schema_for_db di b2,
# karena arsitektur baru tidak punya fungsi ini di tempat lain)
# ──────────────────────────────────────────────
def _generate_schema_for_db(db_info: dict) -> str:
    otn_list = db_info["table_names_original"]
    otn_idx_dic = dict(enumerate(otn_list))
    otn_ocn_dic = {i: [] for i in range(len(otn_list))}

    for t_idx, col in db_info["column_names_original"][1:]:
        table = otn_idx_dic[t_idx]
        table_fmt = f"`{table}`" if " " in table else table
        col_fmt = f"`{col}`" if " " in col else col
        otn_ocn_dic[t_idx].append(f"{table_fmt}.{col_fmt}")

    parts = []
    for t_idx, cols in otn_ocn_dic.items():
        parts.append(f"### Table: {otn_idx_dic[t_idx]}\n{cols}")
    return "\n\n".join(parts)


def build_schema_lookup(table_json: list) -> dict:
    return {db["db_id"]: _generate_schema_for_db(db) for db in table_json}


# ──────────────────────────────────────────────
# SQL post-processing (setara postprocess_sql di run_b2.py)
# ──────────────────────────────────────────────
def postprocess_sql(raw_sql: str, db_id: str) -> str:
    sql = None
    for pattern in (r"```sqlite(.*?)```", r"```sql(.*?)```", r"```(.*?)```"):
        match = re.search(pattern, raw_sql or "", re.DOTALL)
        if match:
            sql = match.group(1).strip()
            break
    if not sql:
        sql = (raw_sql or "").strip()
    return f"{sql}\t----- bird -----\t{db_id}"


# ──────────────────────────────────────────────
# Config / path resolution
# ──────────────────────────────────────────────
def load_data_config(data_config_path: str):
    base_dir = os.path.dirname(os.path.abspath(data_config_path))
    cfg = json.load(open(data_config_path, "r"))

    def rp(p):
        return os.path.normpath(os.path.join(base_dir, p))

    table_json = json.load(open(rp(cfg["dev_tables"]), "r"))
    data_json = json.load(open(rp(cfg["dev_data"]), "r"))
    column_meaning_json = json.load(open(rp(cfg["dev_column_meaning"]), "r"))
    db_dir = rp(cfg["dev_db_dir"])

    return table_json, data_json, column_meaning_json, db_dir


# ──────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────
def run_evaluation(
    data_config_path: str,
    output_dir: str,
    limit: int = None,
    resume: bool = True,
    save_every: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    final_sql_path = os.path.join(output_dir, "final_sql.json")
    latency_path = os.path.join(output_dir, "latency.json")
    error_log_path = os.path.join(output_dir, "errors.jsonl")

    table_json, data_json, column_meaning_json, db_dir = load_data_config(
        data_config_path
    )
    schema_lookup = build_schema_lookup(table_json)

    print(f"[*] Total samples in dev_data : {len(data_json)}")
    items = data_json[:limit] if limit else data_json

    results = {}
    if resume and os.path.exists(final_sql_path):
        results = json.load(open(final_sql_path, "r"))
        print(f"[*] Resuming: {len(results)} samples already done")

    print("[*] Initializing system (generator + BAM + SAM + LOM + RAG)...")
    models = initialize_system()
    print("[✓] System ready\n")

    per_sample_times = []
    err_f = open(error_log_path, "a", encoding="utf-8")

    for idx, info in enumerate(tqdm(items, desc="Evaluating")):
        key = str(idx)
        if key in results:
            continue

        db_id = info["db_id"]
        question = info["question"]
        evidence = info.get("evidence", "")
        schema = schema_lookup.get(db_id, "")

        if not schema:
            print(f"[!] WARNING idx={idx}: db_id '{db_id}' tidak ditemukan di dev_tables.json")

        t0 = time.time()
        try:
            out = process_prompt(question, evidence, schema, models)
            raw_sql = out["final_sql"]
        except Exception as e:
            raw_sql = ""
            err_f.write(json.dumps({"idx": idx, "db_id": db_id, "error": str(e)}) + "\n")
            err_f.flush()
        elapsed = time.time() - t0
        per_sample_times.append(elapsed)

        results[key] = postprocess_sql(raw_sql, db_id)

        if (idx + 1) % save_every == 0:
            json.dump(results, open(final_sql_path, "w"), indent=4)

    err_f.close()
    json.dump(results, open(final_sql_path, "w"), indent=4)

    n = len(per_sample_times)
    latency_summary = {
        "num_samples_this_run": n,
        "total_seconds": round(sum(per_sample_times), 4),
        "avg_seconds": round(sum(per_sample_times) / max(1, n), 4),
        "min_seconds": round(min(per_sample_times), 4) if per_sample_times else 0.0,
        "max_seconds": round(max(per_sample_times), 4) if per_sample_times else 0.0,
    }
    json.dump(latency_summary, open(latency_path, "w"), indent=4)

    print(f"\n[✓] Done. {len(results)} total samples in {final_sql_path}")
    print(f"[✓] Latency summary -> {latency_path}")
    if os.path.exists(error_log_path) and os.path.getsize(error_log_path) > 0:
        print(f"[!] Beberapa sample gagal, cek {error_log_path}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate API-style pipeline on dev dataset")
    p.add_argument("--data_config_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=None, help="Batasi jumlah sample (smoke test)")
    p.add_argument("--no_resume", action="store_true", help="Jangan lanjutkan dari final_sql.json lama")
    p.add_argument("--save_every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    run_evaluation(
        data_config_path=opt.data_config_path,
        output_dir=opt.output_dir,
        limit=opt.limit,
        resume=not opt.no_resume,
        save_every=opt.save_every,
    )