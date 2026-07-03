#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYNTHETIC SQL DATA GENERATOR — 4-CLASS ICRL (REVISED)

Kelas difficulty berbasis bucket presence:
  simple      → B1 only         (SELECT/FROM/WHERE/JOIN, no agg, no cond-logic)
  conditional → B1 + B3         (+ CASE/IN/BETWEEN/LIKE, NO aggregation)
  aggregation → B1 + B4 (pure)  (+ COUNT/AVG/SUM/GROUP BY, NO B3 keywords)
  complex     → B1 + B3 + B4    (aggregation + conditional logic)

Aturan 'aggregation' sangat ketat:
  - WHERE hanya boleh equality/comparison (=, !=, >, <, >=, <=)
  - FORBIDDEN: IN, BETWEEN, LIKE, CASE, WHEN, THEN, ELSE
  - AND/OR di luar konteks conditional dianggap B3 → redirect ke complex

Redirect: TIDAK ADA — mismatch langsung refine sampai MAX_RETRY habis.

Referensi paper:
  Toteja et al. (2025). In-Context Reinforcement Learning based
  Retrieval-Augmented Generation for Text-to-SQL. COLING 2025.
  Section 3.1, Eq. (1) dan (2).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import glob
import re
import random
import itertools
import sqlite3
import time
import datetime
import warnings
import pandas as pd
import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
import signal

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_ID        = "../../../models/Qwen3-32B"
JSON_PATH       = "../../../dataset/train/train/train_tables.json"
TRAIN_DB_FOLDER = "../../../dataset/train/train/train_databases"
OUTPUT_FILE     = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/data.json"
LOG_FILE        = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/log.jsonl"

TARGET_TOTAL_DATA = 9428

# ------------------------------------------------------------------------------
# 4-Class Difficulty Config
# 9428 / 4 = 2357 tepat — distribusi merata
# ------------------------------------------------------------------------------
DIFFICULTY_CONFIG = {
    "simple": {
        "quota_pct": 0.25,
        "hint": (
            "Basic SELECT/FROM/WHERE only. At most 1 JOIN. "
            "NO aggregation (COUNT/AVG/SUM/MIN/MAX). NO GROUP BY. NO HAVING. "
            "NO conditional logic (no CASE/IN/BETWEEN/LIKE)."
        ),
    },
    "conditional": {
        "quota_pct": 0.25,
        "hint": (
            "Use JOIN + conditional logic: at least one of CASE/WHEN, IN (...), "
            "BETWEEN ... AND ..., or LIKE '...'. "
            "STRICTLY NO aggregation functions (COUNT/AVG/SUM/MIN/MAX). "
            "NO GROUP BY. NO HAVING."
        ),
    },
    "aggregation": {
        "quota_pct": 0.25,
        "hint": (
            "Use aggregation (COUNT/AVG/SUM/MIN/MAX) with GROUP BY. "
            "WHERE clause is ALLOWED but ONLY with simple equality/comparison "
            "(=, !=, >, <, >=, <=). "
            "STRICTLY FORBIDDEN: IN, BETWEEN, LIKE, CASE, WHEN, THEN, ELSE, AND, OR."
        ),
    },
    "complex": {
        "quota_pct": 0.25,
        "hint": (
            "Use aggregation (COUNT/AVG/SUM/MIN/MAX + GROUP BY/HAVING) "
            "AND conditional logic (CASE/WHEN, IN, BETWEEN, or LIKE). "
            "Multiple JOINs or subquery strongly encouraged."
        ),
    },
}

TARGET_COUNTS = {
    level: int(TARGET_TOTAL_DATA * cfg["quota_pct"])
    for level, cfg in DIFFICULTY_CONFIG.items()
}
# Sesuaikan rounding error ke 'complex'
TARGET_COUNTS["complex"] = (
    TARGET_TOTAL_DATA
    - TARGET_COUNTS["simple"]
    - TARGET_COUNTS["conditional"]
    - TARGET_COUNTS["aggregation"]
)

# Generation hyperparameters
GENERATION_CONFIG = {
    "max_new_tokens":     512,
    "temperature":        0.7,
    "top_p":              0.8,
    "top_k":              20,
    "do_sample":          True,
    "repetition_penalty": 1.05,
}

STOP_STRINGS       = ["\n\n", "Explanation:", "Note:", "This SQL", "```", "<think>", "</think>"]
MAX_RETRY_PER_ITEM = 4      # dinaikkan karena tidak ada redirect
GENERATION_TIMEOUT = 30
MAX_PROMPT_LENGTH  = 8000

# Keyword B3 yang dilarang di kelas 'aggregation'
# AND/OR termasuk karena hanya boleh equality di WHERE
AGGREGATION_FORBIDDEN_B3 = {
    "IN", "BETWEEN", "LIKE", "CASE", "WHEN", "THEN", "ELSE", "END",
    "AND", "OR", "NOT",
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# ==============================================================================
# REWARD VALIDATOR — 4-CLASS
# Paper-compliant: c(Bi) = f(Bi)/total_hits,  R(St) = Σ f(Bi)·c(Bi)
# Classifier: bucket presence pattern
# ==============================================================================

class RewardValidator:
    """
    Reward function sesuai paper Section 3.1, Eq. (1) & (2).

    Klasifikasi 4 kelas berbasis bucket presence:
        simple      → B1 hadir, B3=0, B4=0
        conditional → B1 + B3 hadir, B4=0
        aggregation → B1 + B4 hadir, B3=0 (SANGAT ketat — lihat _is_pure_aggregation)
        complex     → B1 + B3 + B4 semua hadir

    B2 (DML) sengaja dihilangkan — Spider/Bird adalah read-only benchmarks.
    """

    BUCKET_B1 = {
        "SELECT": 1, "FROM": 1, "LIMIT": 1, "DISTINCT": 2,
        "JOIN": 2, "ON": 2, "INNER JOIN": 3, "LEFT JOIN": 3,
        "RIGHT JOIN": 3, "WHERE": 2, "GROUP BY": 3, "HAVING": 3, "ORDER BY": 2,
    }

    BUCKET_B3 = {
        "AND": 1, "OR": 1, "NOT": 1, "IN": 2, "BETWEEN": 2,
        "LIKE": 2, "CASE": 3, "WHEN": 2, "THEN": 2, "ELSE": 2, "END": 1,
    }

    BUCKET_B4 = {
        "AVG": 3, "SUM": 3, "COUNT": 3, "MIN": 3, "MAX": 3,
        "ASC": 1, "DESC": 1,
    }

    BUCKETS = {"B1": BUCKET_B1, "B3": BUCKET_B3, "B4": BUCKET_B4}

    @staticmethod
    def _clean_sql(sql: str) -> str:
        sql = sql.upper()
        sql = re.sub(r"`[^`]*`", "", sql)
        sql = re.sub(r"'[^']*'", "", sql)
        sql = re.sub(r'"[^"]*"', "", sql)
        return sql

    @staticmethod
    def _count_hits(sql_clean: str, bucket: dict) -> dict:
        hits = {}
        for kw in bucket:
            count = len(re.findall(r"\b" + re.escape(kw) + r"\b", sql_clean))
            if count > 0:
                hits[kw] = count
        return hits

    @classmethod
    def compute_reward(cls, sql_query: str) -> dict:
        if not sql_query or not sql_query.strip():
            return cls._empty_result()

        sql_clean = cls._clean_sql(sql_query)

        bucket_hits = {
            name: cls._count_hits(sql_clean, kw_dict)
            for name, kw_dict in cls.BUCKETS.items()
        }
        bucket_freq = {
            name: sum(bucket_hits[name].values())
            for name in cls.BUCKETS
        }
        total_hits = sum(bucket_freq.values())

        if total_hits > 0:
            bucket_score = {
                name: bucket_freq[name] / total_hits
                for name in cls.BUCKETS
            }
        else:
            bucket_score = {name: 0.0 for name in cls.BUCKETS}

        reward = sum(
            bucket_freq[name] * bucket_score[name]
            for name in cls.BUCKETS
        )

        buckets_present = [
            name for name in cls.BUCKETS if bucket_freq[name] > 0
        ]

        difficulty = cls._classify_difficulty(bucket_freq, sql_clean)

        return {
            "reward":          reward,
            "bucket_hits":     bucket_hits,
            "bucket_freq":     bucket_freq,
            "bucket_score":    bucket_score,
            "total_hits":      total_hits,
            "difficulty":      difficulty,
            "buckets_present": buckets_present,
        }

    @classmethod
    def _classify_difficulty(cls, bucket_freq: dict, sql_clean: str) -> str | None:
        """
        4-class classifier berdasarkan bucket presence pattern.

        Hierarki:
            complex     → B1 + B3 + B4
            aggregation → B1 + B4, DAN lolos _is_pure_aggregation (B3 forbidden keywords = 0)
            conditional → B1 + B3, B4 = 0
            simple      → B1 only
            None        → tidak ada bucket aktif
        """
        b1 = bucket_freq.get("B1", 0) > 0
        b3 = bucket_freq.get("B3", 0) > 0
        b4 = bucket_freq.get("B4", 0) > 0

        if b1 and b3 and b4:
            return "complex"
        elif b1 and b4 and not b3:
            # Cek sangat ketat: pastikan tidak ada forbidden B3 keyword
            if cls._is_pure_aggregation(sql_clean):
                return "aggregation"
            else:
                # Ada B3 forbidden keyword tersembunyi → naik ke complex
                return "complex"
        elif b1 and b3 and not b4:
            return "conditional"
        elif b1 and not b3 and not b4:
            return "simple"
        else:
            return None

    @staticmethod
    def _is_pure_aggregation(sql_clean: str) -> bool:
        """
        Validasi sangat ketat untuk kelas 'aggregation':
        Tidak boleh ada satupun keyword dari AGGREGATION_FORBIDDEN_B3.
        AND/OR juga dilarang karena hanya equality WHERE yang diizinkan.
        """
        for kw in AGGREGATION_FORBIDDEN_B3:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, sql_clean):
                return False
        return True

    @classmethod
    def calculate_score(cls, sql_query: str) -> int:
        """Backward-compatible alias — raw weighted integer sum untuk debug."""
        if not sql_query:
            return 0
        sql_clean = cls._clean_sql(sql_query)
        raw = 0
        for bucket in cls.BUCKETS.values():
            for kw, weight in bucket.items():
                count = len(re.findall(r"\b" + re.escape(kw) + r"\b", sql_clean))
                raw += count * weight
        return raw

    @staticmethod
    def _empty_result() -> dict:
        return {
            "reward":          0.0,
            "bucket_hits":     {"B1": {}, "B3": {}, "B4": {}},
            "bucket_freq":     {"B1": 0,  "B3": 0,  "B4": 0},
            "bucket_score":    {"B1": 0.0, "B3": 0.0, "B4": 0.0},
            "total_hits":      0,
            "difficulty":      None,
            "buckets_present": [],
        }


# ==============================================================================
# EVENT LOGGER
# ==============================================================================

class EventLogger:

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._write({
            "event":            "SESSION_START",
            "timestamp":        self._now(),
            "output_file":      OUTPUT_FILE,
            "target_total":     TARGET_TOTAL_DATA,
            "target_counts":    TARGET_COUNTS,
            "classifier":       "4-class bucket presence (simple/conditional/aggregation/complex)",
            "redirect_policy":  "NONE — mismatch → refine until MAX_RETRY",
            "aggregation_rule": "pure: no IN/BETWEEN/LIKE/CASE/AND/OR in WHERE",
            "difficulty_config": {
                k: {"quota_pct": v["quota_pct"]}
                for k, v in DIFFICULTY_CONFIG.items()
            },
        })

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().isoformat(timespec="milliseconds")

    def _write(self, record: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_generated(
        self, item_id: int, db_id: str, target_diff: str,
        actual_diff: str, reward: float, bucket_freq: dict,
        attempt: int, sql: str, question: str, was_refined: bool,
    ):
        self._write({
            "event":             "ACCEPTED",
            "timestamp":         self._now(),
            "item_id":           item_id,
            "db_id":             db_id,
            "target_difficulty": target_diff,
            "actual_difficulty": actual_diff,
            "reward":            round(reward, 4),
            "bucket_freq":       bucket_freq,
            "attempt_number":    attempt,
            "was_refined":       was_refined,
            "question_preview":  question[:120],
            "sql_preview":       sql[:200],
        })

    def log_refinement(
        self, item_id: int, db_id: str, target_diff: str,
        attempt: int, reward: float, bucket_freq: dict,
        missing_buckets: list, forbidden_found: list, old_sql: str,
    ):
        self._write({
            "event":             "REFINEMENT_TRIGGERED",
            "timestamp":         self._now(),
            "item_id":           item_id,
            "db_id":             db_id,
            "target_difficulty": target_diff,
            "attempt_number":    attempt,
            "reward_before":     round(reward, 4),
            "bucket_freq":       bucket_freq,
            "missing_buckets":   missing_buckets,
            "forbidden_found":   forbidden_found,
            "sql_before":        old_sql[:200],
        })

    def log_rejected(
        self, item_id: int, db_id: str, target_diff: str,
        reason: str, attempt: int, reward: float = None,
        bucket_freq: dict = None, sql: str = None, detail: str = "",
    ):
        self._write({
            "event":             "REJECTED",
            "timestamp":         self._now(),
            "item_id":           item_id,
            "db_id":             db_id,
            "target_difficulty": target_diff,
            "reject_reason":     reason,
            "attempt_number":    attempt,
            "reward":            round(reward, 4) if reward is not None else None,
            "bucket_freq":       bucket_freq,
            "sql_preview":       sql[:200] if sql else None,
            "detail":            detail,
        })

    def log_session_end(self, collected: dict, stats: dict, elapsed_min: float):
        self._write({
            "event":           "SESSION_END",
            "timestamp":       self._now(),
            "elapsed_minutes": round(elapsed_min, 2),
            "collected":       collected,
            "stats":           stats,
        })


# ==============================================================================
# TIMEOUT CONTEXT
# ==============================================================================

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError()


@contextmanager
def timeout_context(seconds):
    if os.name != "posix":
        yield
        return
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ==============================================================================
# SCHEMA GRAPH BUILDER
# ==============================================================================

class SchemaGraphBuilder:

    def __init__(self, json_path, train_databases_path):
        self.json_path = json_path
        self.train_databases_path = train_databases_path
        self.G = nx.DiGraph()
        self.semantic_map = {}

    def load_semantics(self):
        print("[INIT] Memuat Metadata Semantik...")
        csv_files = glob.glob(
            os.path.join(self.train_databases_path, "**", "*.csv"),
            recursive=True,
        )
        count = 0
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding="latin1")
                table_name = os.path.basename(file).replace(".csv", "")
                for _, row in df.iterrows():
                    col_raw  = row.get("original_column_name") or row.get("Field") or ""
                    desc_raw = row.get("column_description") or row.get("Description") or ""
                    col_name = str(col_raw).strip()
                    if col_name:
                        key = f"{table_name}.{col_name}".lower()
                        self.semantic_map[key] = str(desc_raw).strip()
                count += 1
            except Exception:
                pass
        print(f"[INIT] Semantik dimuat dari {count} file CSV.")

    def build_topology(self):
        print("[INIT] Membangun Graf Skema...")
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] File tidak ditemukan: {self.json_path}")
            return None

        for db in data:
            db_id     = db.get("db_id")
            tables    = db.get("table_names_original", [])
            col_names = db.get("column_names_original", [])
            fks       = db.get("foreign_keys", [])
            col_idx_map = {}

            for t_idx, table_name in enumerate(tables):
                t_node = f"{db_id}.{table_name}"
                self.G.add_node(t_node, type="table", label=table_name, db_id=db_id)

            for c_idx, (t_idx, col_name) in enumerate(col_names):
                if t_idx == -1:
                    continue
                table_name = tables[t_idx]
                col_node   = f"{db_id}.{table_name}.{col_name}"
                key        = f"{table_name}.{col_name}".lower()
                desc       = self.semantic_map.get(key, "")
                self.G.add_node(col_node, type="column", label=col_name, description=desc)
                t_node = f"{db_id}.{table_name}"
                self.G.add_edge(t_node, col_node, relation="type_2_containment")
                col_idx_map[c_idx] = (t_node, col_node)

            for src_idx, tgt_idx in fks:
                if src_idx in col_idx_map and tgt_idx in col_idx_map:
                    src_t, _ = col_idx_map[src_idx]
                    tgt_t, _ = col_idx_map[tgt_idx]
                    self.G.add_edge(src_t, tgt_t, relation="type_3_fk")
                    self.G.add_edge(tgt_t, src_t, relation="type_3_fk")

        print(
            f"[INIT] Graf selesai: "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges."
        )
        return self.G


# ==============================================================================
# RANDOM WALK GENERATOR
# ==============================================================================

class RandomWalkGenerator:

    def __init__(self, graph):
        self.G = graph

    def generate_walk(self, db_id):
        tables = [
            n for n, a in self.G.nodes(data=True)
            if a["type"] == "table" and n.startswith(f"{db_id}.")
        ]
        if not tables:
            return []
        curr         = random.choice(tables)
        path         = [curr]
        target_depth = random.choice([2, 3, 4])

        for _ in range(target_depth - 1):
            neis = [
                t for _, t, a in self.G.out_edges(curr, data=True)
                if a.get("relation") == "type_3_fk"
            ]
            if not neis:
                break
            curr = random.choice(neis)
            if curr not in path:
                path.append(curr)
        return path

    def serialize_path_with_samples(self, path, db_id):
        db_path = os.path.join(TRAIN_DB_FOLDER, db_id, f"{db_id}.sqlite")
        has_db  = os.path.exists(db_path)
        desc_list   = []
        total_chars = 0

        for node in path:
            attrs      = self.G.nodes[node]
            table_name = attrs.get("label", node.split(".")[-1])

            all_cols = []
            for _, c_node, rel in self.G.out_edges(node, data=True):
                if rel.get("relation") == "type_2_containment":
                    c_attr  = self.G.nodes[c_node]
                    desc    = c_attr.get("description", "")
                    col_str = f"{c_attr['label']}"
                    if desc:
                        col_str += f" ({desc})"
                    all_cols.append(col_str)

            schema_str = f"Table '{table_name}' columns: [{', '.join(all_cols)}]"

            if has_db:
                try:
                    conn   = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1;")
                    rows = cursor.fetchall()
                    cols = [d[0] for d in cursor.description]
                    conn.close()
                    if rows:
                        schema_str += f"\n  Sample: {cols} → {list(rows[0])}"
                except Exception:
                    pass

            if total_chars + len(schema_str) + 500 > MAX_PROMPT_LENGTH:
                break

            desc_list.append(schema_str)
            total_chars += len(schema_str) + 2

        return "\n\n".join(desc_list)[:MAX_PROMPT_LENGTH]


# ==============================================================================
# QWEN GENERATOR — 4-CLASS AWARE
# ==============================================================================

class QwenGenerator:

    def __init__(self, model_path):
        print(f"[AI] Memuat Model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[AI] Model siap | Device: {next(self.model.parameters()).device}")

    def _clean_output_to_json(self, text: str) -> dict | None:
        if not text:
            return None
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            start = text.find("{")
            if start != -1:
                text = text[start:]
        text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s : e + 1]
        try:
            return json.loads(text)
        except Exception:
            return None

    def _generate(self, prompt: str) -> dict | None:
        messages = [
            {
                "role":    "system",
                "content": (
                    "You are a SQLite Expert. Output JSON ONLY. "
                    "DO NOT think step-by-step. Start directly with {."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        gen_kwargs = {k: v for k, v in GENERATION_CONFIG.items() if v is not None}
        if self.tokenizer.pad_token_id:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        try:
            with timeout_context(GENERATION_TIMEOUT):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **gen_kwargs,
                        tokenizer=self.tokenizer,
                        stop_strings=STOP_STRINGS,
                    )
        except (TimeoutError, Exception):
            return None

        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return self._clean_output_to_json(response)

    # --------------------------------------------------------------------------
    # PROMPT TEMPLATES per kelas
    # --------------------------------------------------------------------------

    def _build_generation_instruction(self, target_difficulty: str) -> str:
        if target_difficulty == "simple":
            return (
                "Generate a SIMPLE query.\n"
                "RULES:\n"
                "  - Use SELECT, FROM, WHERE (optional), at most 1 JOIN.\n"
                "  - NO aggregation: COUNT/AVG/SUM/MIN/MAX are FORBIDDEN.\n"
                "  - NO GROUP BY, NO HAVING.\n"
                "  - NO conditional logic: IN/BETWEEN/LIKE/CASE/WHEN are FORBIDDEN.\n"
                "  - AND/OR allowed only to combine simple equality conditions.\n"
                "GOOD example: SELECT name FROM students WHERE age > 20\n"
                "BAD example:  SELECT COUNT(*) FROM students GROUP BY dept_id"
            )
        elif target_difficulty == "conditional":
            return (
                "Generate a CONDITIONAL query.\n"
                "RULES:\n"
                "  - Must use at least ONE of: CASE/WHEN, IN (...), BETWEEN ... AND ..., LIKE '...'.\n"
                "  - JOIN is strongly encouraged.\n"
                "  - NO aggregation: COUNT/AVG/SUM/MIN/MAX are FORBIDDEN.\n"
                "  - NO GROUP BY, NO HAVING.\n"
                "GOOD example: SELECT name FROM students JOIN dept ON students.dept_id = dept.id "
                "WHERE dept.name IN ('CS', 'Math')\n"
                "BAD example:  SELECT COUNT(*) FROM students GROUP BY dept_id"
            )
        elif target_difficulty == "aggregation":
            return (
                "Generate a PURE AGGREGATION query.\n"
                "RULES:\n"
                "  - Must use at least ONE aggregation function: COUNT/AVG/SUM/MIN/MAX.\n"
                "  - Must use GROUP BY.\n"
                "  - WHERE clause ALLOWED but ONLY with simple equality/comparison: =, !=, >, <, >=, <=.\n"
                "  - STRICTLY FORBIDDEN in WHERE and entire query:\n"
                "      IN, BETWEEN, LIKE, CASE, WHEN, THEN, ELSE, AND, OR, NOT.\n"
                "  - HAVING is FORBIDDEN (it introduces conditional logic).\n"
                "GOOD example: SELECT dept_id, COUNT(*) FROM students WHERE year = 2023 "
                "GROUP BY dept_id\n"
                "BAD example:  SELECT dept_id, COUNT(*) FROM students WHERE dept_id IN (1,2,3) "
                "GROUP BY dept_id  ← IN is FORBIDDEN"
            )
        else:  # complex
            return (
                "Generate a COMPLEX query.\n"
                "RULES:\n"
                "  - Must use aggregation (COUNT/AVG/SUM/MIN/MAX) with GROUP BY.\n"
                "  - Must use conditional logic: at least one of CASE/WHEN, IN, BETWEEN, LIKE.\n"
                "  - HAVING is encouraged.\n"
                "  - Multiple JOINs or subquery strongly encouraged.\n"
                "GOOD example: SELECT dept_id, AVG(score) FROM students "
                "WHERE dept_id IN (1,2,3) GROUP BY dept_id HAVING AVG(score) > 80"
            )

    def _build_refinement_instruction(
        self,
        target_diff:    str,
        missing_buckets: list,
        forbidden_found: list,
        bucket_freq:    dict,
    ) -> str:
        """
        Buat instruksi refinement yang spesifik berdasarkan:
        - Bucket apa yang masih kurang
        - Keyword forbidden apa yang terdeteksi (khusus kelas aggregation)
        """
        lines = [f"The SQL does NOT match target class '{target_diff}'."]

        if missing_buckets:
            lines.append(f"Missing: {missing_buckets}")
            for b in missing_buckets:
                if b == "B3":
                    lines.append(
                        "→ Add conditional logic: "
                        "CASE/WHEN/THEN/ELSE, IN (...), BETWEEN ... AND ..., or LIKE '...'."
                    )
                elif b == "B4":
                    lines.append(
                        "→ Add aggregation: COUNT()/AVG()/SUM()/MIN()/MAX() with GROUP BY."
                    )
                elif b == "B1":
                    lines.append(
                        "→ Ensure the query has SELECT, FROM, and at least one JOIN or WHERE."
                    )

        if forbidden_found:
            lines.append(f"Forbidden keywords detected: {forbidden_found}")
            if target_diff == "aggregation":
                lines.append(
                    "→ REMOVE ALL of these: IN, BETWEEN, LIKE, CASE, WHEN, THEN, ELSE, AND, OR. "
                    "Replace complex WHERE with simple equality (=, >, <, >=, <=) only."
                )
            elif target_diff in ("simple", "conditional"):
                if "COUNT" in forbidden_found or "AVG" in forbidden_found or "SUM" in forbidden_found:
                    lines.append(
                        "→ REMOVE aggregation functions (COUNT/AVG/SUM/MIN/MAX) and GROUP BY."
                    )

        lines.append("Rewrite the SQL strictly following the class rules above.")
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------------------

    def generate_targeted(self, context: str, target_difficulty: str) -> dict | None:
        instr = self._build_generation_instruction(target_difficulty)

        prompt = f"""You are an expert SQLite developer.

[SCHEMA CONTEXT]
{context}

[TASK]
{instr}

[CRITICAL SQLITE RULES]
1. Use ONLY standard SQLite syntax.
   FORBIDDEN dialects: ILIKE, FULL OUTER JOIN, RIGHT JOIN, STRING_AGG, MySQL LIMIT offset.
2. Wrap identifiers with spaces/hyphens in backticks (`).
3. Use ONLY columns and tables listed in [SCHEMA CONTEXT]. No hallucination.
4. Return ONLY raw JSON. No markdown, no explanation.

[EVIDENCE REQUIREMENT]
Explain any abbreviations or value mappings (e.g., "std = student").

[OUTPUT FORMAT]
{{"question": "...", "sql": "SELECT ...", "evidence": "..."}}

Generate the JSON now:"""

        result = self._generate(prompt)
        if result and "sql" in result:
            if not result.get("evidence", "").strip():
                result["evidence"] = self._auto_generate_evidence(result["sql"], context)
        return result

    def generate_refinement(
        self,
        context:         str,
        old_q:           str,
        old_sql:         str,
        target_diff:     str,
        reward_detail:   dict,
        missing_buckets: list,
        forbidden_found: list,
    ) -> dict | None:
        """
        ICRL Refinement dengan feedback spesifik per kelas.
        Menggunakan reward_detail dari RewardValidator untuk bucket-aware hints.
        """
        refinement_instr = self._build_refinement_instruction(
            target_diff, missing_buckets, forbidden_found, reward_detail.get("bucket_freq", {})
        )
        class_rule = self._build_generation_instruction(target_diff)

        prompt = f"""[SCHEMA]
{context}

[REFINEMENT TASK]
{refinement_instr}

Current reward R = {reward_detail.get('reward', 0):.4f}
Active buckets  : {reward_detail.get('buckets_present', [])}
Bucket freq     : {reward_detail.get('bucket_freq', {})}

[OLD QUESTION]
{old_q}

[OLD SQL]
{old_sql}

[TARGET CLASS RULES]
{class_rule}

[SQLITE RULES]
- Standard SQLite syntax only.
- Use ONLY columns/tables from [SCHEMA].
- No hallucination.

[OUTPUT]
JSON only: {{"question": "...", "sql": "SELECT ...", "evidence": "..."}}"""

        return self._generate(prompt)

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _auto_generate_evidence(self, sql: str, context: str) -> str:
        evidences = []
        for _, col in re.findall(r"(\w+)\.(\w+)", sql):
            if len(col) <= 3 and col.islower():
                full = self._expand_abbreviation(col)
                if full != col:
                    evidences.append(f"{col}={full}")
        return (
            "; ".join(evidences) if evidences
            else "Filter based on domain knowledge"
        )

    @staticmethod
    def _expand_abbreviation(abbr: str) -> str:
        abbr_map = {
            "std": "student", "dept": "department", "prof": "professor",
            "crs": "course",  "enr": "enrollment",  "sem": "semester",
            "yr":  "year",    "id":  "identifier",   "num": "number",
        }
        return abbr_map.get(abbr.lower(), abbr)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def _detect_forbidden_in_sql(sql: str) -> list:
    """
    Deteksi keyword forbidden yang muncul di SQL untuk kelas aggregation.
    Digunakan untuk feedback refinement.
    """
    sql_upper = sql.upper()
    # Hapus string literals dulu
    sql_upper = re.sub(r"'[^']*'", "", sql_upper)
    sql_upper = re.sub(r'"[^"]*"', "", sql_upper)
    found = []
    for kw in AGGREGATION_FORBIDDEN_B3:
        if re.search(r"\b" + re.escape(kw) + r"\b", sql_upper):
            found.append(kw)
    return found


def _detect_forbidden_for_class(sql: str, target_diff: str) -> list:
    """
    Deteksi keyword yang seharusnya tidak ada berdasarkan target kelas.
    """
    sql_upper = sql.upper()
    sql_upper = re.sub(r"'[^']*'", "", sql_upper)
    sql_upper = re.sub(r'"[^"]*"', "", sql_upper)

    forbidden = []

    if target_diff == "aggregation":
        # Semua B3 + AND/OR dilarang
        for kw in AGGREGATION_FORBIDDEN_B3:
            if re.search(r"\b" + re.escape(kw) + r"\b", sql_upper):
                forbidden.append(kw)

    elif target_diff in ("simple", "conditional"):
        # Aggregation functions dilarang
        for kw in ["COUNT", "AVG", "SUM", "MIN", "MAX", "GROUP BY", "HAVING"]:
            if re.search(r"\b" + re.escape(kw.replace(" ", r"\s+")) + r"\b", sql_upper):
                forbidden.append(kw)

    return forbidden


def main():
    print("=" * 70)
    print("🚀 SYNTHETIC SQL GENERATOR — 4-CLASS ICRL")
    print("=" * 70)
    print(f"Target Total    : {TARGET_TOTAL_DATA:,}")
    print(f"Target per class: {TARGET_COUNTS}")
    print(f"Classifier      : 4-class bucket presence")
    print(f"Redirect policy : NONE (mismatch → refine)")
    print(f"Aggregation rule: VERY STRICT (no IN/BETWEEN/LIKE/CASE/AND/OR)")
    print(f"Output          : {OUTPUT_FILE}")
    print(f"Log             : {LOG_FILE}")
    print("=" * 70)

    # ── Init components ───────────────────────────────────────────────
    builder = SchemaGraphBuilder(JSON_PATH, TRAIN_DB_FOLDER)
    builder.load_semantics()
    graph = builder.build_topology()
    if not graph:
        return

    with open(JSON_PATH, "r") as f:
        all_db_ids = [db["db_id"] for db in json.load(f)]

    walker    = RandomWalkGenerator(graph)
    validator = RewardValidator()
    generator = QwenGenerator(MODEL_ID)
    logger    = EventLogger(LOG_FILE)

    # ── Resume logic ──────────────────────────────────────────────────
    dataset   = []
    collected = {k: 0 for k in DIFFICULTY_CONFIG}

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            for e in dataset:
                if e.get("difficulty") in collected:
                    collected[e["difficulty"]] += 1
            print(f"[RESUME] Loaded {len(dataset):,} existing entries.")
            print(f"[RESUME] Collected: {collected}")
        except Exception as e:
            print(f"[WARN] Resume failed: {e}")

    db_cycle = itertools.cycle(all_db_ids)
    stats = {
        "attempt_total":        0,
        "accepted":             0,
        "refined":              0,
        "parse_err":            0,
        "timeout":              0,
        "no_bucket_active":     0,
        "wrong_class_rejected": 0,
        "forbidden_kw_rejected": 0,
    }

    start_time    = time.time()
    last_log_time = time.time()
    LOG_INTERVAL  = 5.0
    item_counter  = len(dataset)

    def select_target() -> str | None:
        """Pilih kelas yang proporsinya paling jauh dari target."""
        avail = [l for l in DIFFICULTY_CONFIG if collected[l] < TARGET_COUNTS[l]]
        if not avail:
            return None
        return min(avail, key=lambda l: collected[l] / TARGET_COUNTS[l])

    print(f"\n📝 Starting generation loop...\n")

    while sum(collected.values()) < TARGET_TOTAL_DATA:
        target_diff = select_target()
        if target_diff is None:
            break

        current_db    = next(db_cycle)
        item_counter += 1

        path = walker.generate_walk(current_db)
        if len(path) < 2:
            continue
        context = walker.serialize_path_with_samples(path, current_db)

        # ── State per item ────────────────────────────────────────────
        final_data       = None
        accepted_diff    = None
        was_refined      = False
        last_reward      = None
        last_result      = None
        last_sql         = None
        last_q           = None

        # ── ICRL Loop ─────────────────────────────────────────────────
        for attempt in range(MAX_RETRY_PER_ITEM):
            stats["attempt_total"] += 1

            # ── Generate ────────────────────────────────────────────
            if attempt == 0:
                data = generator.generate_targeted(context, target_diff)
            else:
                # Hitung missing dan forbidden untuk feedback
                missing_buckets = []
                forbidden_found = []

                if last_result:
                    bfreq = last_result["bucket_freq"]

                    # Bucket yang seharusnya ada tapi kosong
                    required = {
                        "simple":      ["B1"],
                        "conditional": ["B1", "B3"],
                        "aggregation": ["B1", "B4"],
                        "complex":     ["B1", "B3", "B4"],
                    }.get(target_diff, [])
                    missing_buckets = [b for b in required if bfreq.get(b, 0) == 0]

                if last_sql:
                    forbidden_found = _detect_forbidden_for_class(last_sql, target_diff)

                logger.log_refinement(
                    item_id=item_counter,
                    db_id=current_db,
                    target_diff=target_diff,
                    attempt=attempt,
                    reward=last_reward or 0.0,
                    bucket_freq=last_result["bucket_freq"] if last_result else {},
                    missing_buckets=missing_buckets,
                    forbidden_found=forbidden_found,
                    old_sql=last_sql or "",
                )

                data = generator.generate_refinement(
                    context=context,
                    old_q=last_q or "",
                    old_sql=last_sql or "",
                    target_diff=target_diff,
                    reward_detail=last_result or RewardValidator._empty_result(),
                    missing_buckets=missing_buckets,
                    forbidden_found=forbidden_found,
                )
                stats["refined"] += 1
                was_refined = True

            # ── Parse check ─────────────────────────────────────────
            if not data or "sql" not in data or not data.get("sql", "").strip():
                stats["parse_err"] += 1
                logger.log_rejected(
                    item_id=item_counter, db_id=current_db,
                    target_diff=target_diff, reason="parse_error",
                    attempt=attempt, detail="Output tidak bisa di-parse ke JSON valid",
                )
                continue

            sql    = data["sql"].strip()
            last_q = data.get("question", "")

            # ── Reward computation (paper Eq. 1 & 2) ────────────────
            result     = RewardValidator.compute_reward(sql)
            reward     = result["reward"]
            difficulty = result["difficulty"]
            last_reward = reward
            last_result = result
            last_sql    = sql

            # ── Kasus 1: Tidak ada bucket aktif ──────────────────────
            if difficulty is None:
                stats["no_bucket_active"] += 1
                logger.log_rejected(
                    item_id=item_counter, db_id=current_db,
                    target_diff=target_diff, reason="no_bucket_active",
                    attempt=attempt, reward=reward,
                    bucket_freq=result["bucket_freq"], sql=sql,
                    detail="Tidak ada bucket B1/B3/B4 aktif",
                )
                final_data = data
                continue

            # ── Kasus 2: Cocok dengan target → accept ────────────────
            if difficulty == target_diff:
                final_data    = data
                accepted_diff = target_diff
                break

            # ── Kasus 3: Tidak cocok → TIDAK redirect, langsung refine
            if target_diff == "aggregation" and difficulty == "complex":
                stats["forbidden_kw_rejected"] += 1
                reason_detail = (
                    f"aggregation target tapi B3 keywords ditemukan "
                    f"→ classified as '{difficulty}', akan di-refine"
                )
            else:
                stats["wrong_class_rejected"] += 1
                reason_detail = (
                    f"Target '{target_diff}' tapi hasil '{difficulty}' "
                    f"→ refine (no redirect policy)"
                )

            logger.log_rejected(
                item_id=item_counter, db_id=current_db,
                target_diff=target_diff, reason="wrong_class_no_redirect",
                attempt=attempt, reward=reward,
                bucket_freq=result["bucket_freq"], sql=sql,
                detail=reason_detail,
            )
            final_data = data
            continue

        # ── Cek hasil akhir loop ──────────────────────────────────────
        if final_data is None or accepted_diff is None:
            logger.log_rejected(
                item_id=item_counter, db_id=current_db,
                target_diff=target_diff, reason="max_retry_exhausted",
                attempt=MAX_RETRY_PER_ITEM,
                reward=last_reward,
                bucket_freq=last_result["bucket_freq"] if last_result else None,
                sql=last_sql,
                detail=f"Semua {MAX_RETRY_PER_ITEM} attempt habis tanpa hasil valid",
            )
            continue

        # ── Catat accepted item ───────────────────────────────────────
        stats["accepted"] += 1
        logger.log_generated(
            item_id=item_counter, db_id=current_db,
            target_diff=target_diff, actual_diff=accepted_diff,
            reward=last_reward, bucket_freq=last_result["bucket_freq"],
            attempt=attempt + 1, sql=final_data["sql"],
            question=final_data.get("question", ""), was_refined=was_refined,
        )

        entry = {
            "question_id":       len(dataset) + 1,
            "db_id":             current_db,
            "question":          final_data.get("question", ""),
            "sql":               final_data["sql"],
            "evidence":          final_data.get("evidence", "Auto-generated"),
            "reward":            round(last_reward, 4) if last_reward else 0.0,
            "bucket_freq":       last_result["bucket_freq"] if last_result else {},
            "buckets_present":   last_result["buckets_present"] if last_result else [],
            "difficulty":        accepted_diff,
            "target_difficulty": target_diff,
            "was_refined":       was_refined,
            "status":            "pending_validation",
            "metadata": {
                "tables":        [n.split(".")[-1] for n in path],
                "gen_attempts":  attempt + 1,
                "gen_timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
            },
        }
        dataset.append(entry)
        collected[accepted_diff] += 1

        # Periodic save setiap 50 item
        if len(dataset) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

        # Progress log (throttled)
        now = time.time()
        if now - last_log_time >= LOG_INTERVAL:
            total   = sum(collected.values())
            elapsed = now - start_time
            rate    = total / elapsed if elapsed > 0 else 0
            eta_sec = (TARGET_TOTAL_DATA - total) / rate if rate > 0 else 0
            h, rem  = divmod(int(eta_sec), 3600)
            m, s    = divmod(rem, 60)
            print(
                f"\rProgress: {total}/{TARGET_TOTAL_DATA} | "
                f"Rate: {rate:.2f} it/s | ETA: {h}h {m}m | "
                f"S:{collected['simple']} "
                f"C:{collected['conditional']} "
                f"A:{collected['aggregation']} "
                f"X:{collected['complex']} | "
                f"Refined:{stats['refined']} "
                f"Err:{stats['parse_err']} "
                f"WrongClass:{stats['wrong_class_rejected']}",
                end="", flush=True,
            )
            last_log_time = now

    # ── Final save ────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60
    logger.log_session_end(collected=collected, stats=stats, elapsed_min=total_time)

    print("\n\n" + "=" * 70)
    print("✅ GENERATION COMPLETE")
    print(f"⏱  Total Time          : {total_time:.1f} minutes")
    print(f"💾 Saved              : {OUTPUT_FILE}")
    print(f"📋 Event Log          : {LOG_FILE}")
    print()
    print("📊 Distribution:")
    for cls, count in collected.items():
        pct = count / TARGET_TOTAL_DATA * 100 if TARGET_TOTAL_DATA > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {cls:>12} : {count:>5} / {TARGET_COUNTS[cls]:>5}  ({pct:5.1f}%)  {bar}")
    print()
    print("📊 Attempt Stats:")
    print(f"   Total attempts      : {stats['attempt_total']}")
    print(f"   ├─ Accepted         : {stats['accepted']}")
    print(f"   ├─ Refined (ICRL)   : {stats['refined']}")
    print(f"   ├─ Parse Error      : {stats['parse_err']}")
    print(f"   ├─ No Bucket Active : {stats['no_bucket_active']}")
    print(f"   ├─ Wrong Class      : {stats['wrong_class_rejected']}")
    print(f"   └─ Forbidden KW     : {stats['forbidden_kw_rejected']} (agg→complex)")
    print()
    print("⚠️  NEXT STEP: Run 'data_syn_validate.py' to filter invalid SQLs.")
    print("=" * 70)


if __name__ == "__main__":
    main()