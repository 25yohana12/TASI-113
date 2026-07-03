import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import re
import sqlite3
import datetime
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional
import time
from contextlib import contextmanager
import signal

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_ID          = "../../../models/Qwen3-32B"
TRAIN_DB_FOLDER   = "../../../dataset/train/train/train_databases"

INPUT_FILE        = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/validate-iter-01/validate-iter-01.json"
OUTPUT_FILE       = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/quality-iter-01/quality-iter-01.json"
VALIDATION_LOG    = "../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/quality-iter-01/log-quality-iter-01.jsonl"

# ------------------------------------------------------------------------------
# DEBUG TOGGLE
# Jika True  → cetak semua detail: prompt, raw LLM output, SQL result, reasoning, error
# Jika False → hanya cetak ringkasan per item (mode production)
# ------------------------------------------------------------------------------
DEBUG = False

# ------------------------------------------------------------------------------
# Generation config untuk LLM Judge
# Menggunakan Qwen3 recommended config untuk thinking mode
# (ref: https://huggingface.co/Qwen/Qwen3-32B#best-practices)
#   Temperature=0.6, TopP=0.95, TopK=20, MinP=0 (default generation_config.json)
# DO NOT use greedy decoding (do_sample=True wajib) → mencegah repetisi tak terbatas
# ------------------------------------------------------------------------------
JUDGE_CONFIG = {
    "max_new_tokens":     8192,   # Harus besar: thinking block bisa ratusan token sebelum JSON
    "temperature":        0.6,    # Qwen3 thinking mode default
    "top_p":              0.95,   # Qwen3 thinking mode default
    "top_k":              20,     # Qwen3 thinking mode default
    "min_p":              0.0,    # Qwen3 thinking mode default
    "do_sample":          True,   # WAJIB — jangan greedy decoding
    "repetition_penalty": 1.05,
}

# STOP_STRINGS: <think> dan </think> DIHAPUS karena enable_thinking=True
# Model harus bebas generate blok thinking sebelum output JSON.
# Hanya stop di sinyal yang muncul SETELAH thinking selesai.
STOP_STRINGS = ["\n\n", "Explanation:", "Note:", "This SQL", "```"]

# Qwen3-32B thinking mode butuh waktu lebih lama:
#   - Model perlu generate <think>...</think> dulu sebelum output JSON
#   - 32B parameter + 480 input tokens → estimasi 60-120 detik wajar
#   - Terlalu kecil (misal 30 detik) → TimeoutError sebelum model selesai berpikir
GENERATION_TIMEOUT     = 180  # detik — cukup untuk thinking + output Qwen3-32B
MAX_SQL_RESULT_DISPLAY = 50   # Max rows yang ditampilkan ke LLM
BATCH_LOG_INTERVAL     = 10   # Log progress setiap N item

os.makedirs(os.path.dirname(OUTPUT_FILE),    exist_ok=True)
os.makedirs(os.path.dirname(VALIDATION_LOG), exist_ok=True)

# ==============================================================================
# DEBUG HELPER
# ==============================================================================

def dprint(*args, **kwargs):
    """Print hanya ketika DEBUG=True. Prefix [DEBUG] otomatis ditambahkan."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# ==============================================================================
# TIMEOUT CONTEXT  (diambil langsung dari gen_data, tidak diubah)
# ==============================================================================

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

@contextmanager
def timeout_context(seconds):
    if os.name != 'posix':
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
# VALIDATION LOGGER  (mengikuti pattern EventLogger di gen_data)
# ==============================================================================

class ValidationLogger:
    """
    Mencatat setiap event validasi ke file JSONL (1 event per baris).
    Format: newline-delimited JSON agar mudah di-stream dan di-parse ulang.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._write({
            "event":       "VALIDATION_SESSION_START",
            "timestamp":   self._now(),
            "input_file":  INPUT_FILE,
            "output_file": OUTPUT_FILE,
            "model":       MODEL_ID,
        })

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().isoformat(timespec='milliseconds')

    def _write(self, record: dict):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_validation_success(self, question_id: int, db_id: str, is_correct: bool,
                               reasoning: str, sql_output: Optional[Any] = None,
                               row_count: int = 0):
        """Event: Validasi berhasil dilakukan."""
        record = {
            "event":             "VALIDATED",
            "timestamp":         self._now(),
            "question_id":       question_id,
            "db_id":             db_id,
            "is_correct":        is_correct,
            "reasoning_preview": reasoning[:200],
            "sql_result_rows":   row_count,
        }
        if not is_correct and sql_output is not None:
            if isinstance(sql_output, list) and len(sql_output) > 0:
                record["sql_output_preview"] = str(sql_output[:3])
        self._write(record)

    def log_execution_error(self, question_id: int, db_id: str, error: str, sql: str):
        """Event: SQL execution gagal."""
        self._write({
            "event":       "EXECUTION_FAILED",
            "timestamp":   self._now(),
            "question_id": question_id,
            "db_id":       db_id,
            "error":       error,
            "sql_preview": sql[:200],
        })

    def log_judge_error(self, question_id: int, db_id: str, error: str):
        """Event: LLM Judge gagal memberikan output valid."""
        self._write({
            "event":       "JUDGE_FAILED",
            "timestamp":   self._now(),
            "question_id": question_id,
            "db_id":       db_id,
            "error":       error,
        })

    def log_session_end(self, total: int, valid: int, invalid: int,
                        exec_errors: int, judge_errors: int, elapsed_min: float):
        """Event: Sesi selesai, ringkasan statistik."""
        self._write({
            "event":             "VALIDATION_SESSION_END",
            "timestamp":         self._now(),
            "total_items":       total,
            "valid_count":       valid,
            "invalid_count":     invalid,
            "execution_errors":  exec_errors,
            "judge_errors":      judge_errors,
            "elapsed_minutes":   round(elapsed_min, 2),
        })

# ==============================================================================
# LLM JUDGE  (mengikuti pola LocalModelInference dari qwen.py)
# ==============================================================================

class LLMJudge:
    """LLM-based validator untuk menilai apakah SQL result menjawab question."""

    def __init__(self, model_path: str):
        # Device eksplisit cuda:0 (mengikuti qwen.py)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = JUDGE_CONFIG.copy()

        print(f"[AI] Memuat Model: {model_path}")
        print(f"[AI] Device: {self.device}")
        print(f"[AI] Mode: THINKING")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded.\n")

    # --------------------------------------------------------------------------
    # _extract_reasoning_and_content
    # Mengikuti qwen.py: pisah blok <think> (reasoning) dari output JSON (content)
    # via </think> sebagai delimiter, lalu strip <|im_end|> dari content.
    # --------------------------------------------------------------------------
    def _extract_reasoning_and_content(self, raw_output: str):
        think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
        reasoning   = think_match.group(1).strip() if think_match else ""

        if '</think>' in raw_output:
            content = raw_output.split('</think>')[1].strip()
        else:
            content = raw_output

        content = content.replace('<|im_end|>', '').strip()
        return reasoning, content

    # --------------------------------------------------------------------------
    # _parse_json_from_content
    # Ekstrak dan parse JSON dari content (sudah bebas dari blok thinking).
    # --------------------------------------------------------------------------
    def _parse_json_from_content(self, content: str) -> dict:
        if not content:
            dprint("_parse_json_from_content: content kosong")
            return None

        # Bersihkan markdown code-fence jika ada
        text = re.sub(r'```(?:json)?\s*', '', content, flags=re.IGNORECASE)
        text = re.sub(r'```', '', text).strip()

        # Ekstrak JSON terluar { ... }
        start = text.find('{')
        end   = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        else:
            dprint(f"_parse_json_from_content: tidak ditemukan JSON object. Content: {repr(text[:200])}")
            return None

        try:
            return json.loads(text)
        except Exception as e:
            dprint(f"_parse_json_from_content: json.loads gagal → {e}")
            dprint(f"Teks yang dicoba parse: {repr(text[:300])}")
            return None

    # --------------------------------------------------------------------------
    # _generate
    # Mengikuti qwen.py:
    #   • enable_thinking=True
    #   • inputs di-move ke self.device per key (bukan .to(device) bulk)
    #   • input_length dari input_ids.shape[1]
    #   • semua generate params dipass eksplisit (bukan **kwargs) → min_p aman
    #   • decode dengan skip_special_tokens=False → </think> tetap ada
    #   • _extract_reasoning_and_content untuk pisah thinking vs output
    # --------------------------------------------------------------------------
    def _generate(self, prompt: str) -> dict:
        messages = [
            {
                "role":    "system",
                "content": "You are a SQL validation expert. Output JSON ONLY. "
                           "Think carefully, then output a single JSON object."
            },
            {"role": "user", "content": prompt}
        ]

        # ── Tokenize ──────────────────────────────────────────────────────────
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True          # izinkan blok <think>...</think>
        )
        inputs       = self.tokenizer(text, return_tensors="pt")
        inputs       = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        dprint(f"── PROMPT SENT TO LLM ({'─' * 40})")
        dprint(prompt)
        dprint(f"{'─' * 56}")
        dprint(f"Input tokens: {input_length}")

        # ── Generate (semua param eksplisit, mengikuti qwen.py) ───────────────
        try:
            with timeout_context(GENERATION_TIMEOUT):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens     = self.config["max_new_tokens"],
                        do_sample          = self.config["do_sample"],
                        temperature        = self.config["temperature"],
                        top_p              = self.config["top_p"],
                        top_k              = self.config["top_k"],
                        min_p              = self.config["min_p"],
                        repetition_penalty = self.config["repetition_penalty"],
                        eos_token_id       = self.tokenizer.eos_token_id,
                        pad_token_id       = self.tokenizer.eos_token_id,
                    )
        except TimeoutError:
            dprint("⏱  GENERATION TIMEOUT — returning None")
            return None
        except Exception as e:
            dprint(f"💥 GENERATION EXCEPTION: {e}")
            return None

        # ── Decode: skip_special_tokens=False agar </think> tetap ada ─────────
        raw = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=False
        ).strip()

        token_usage = {
            "prompt_tokens":     input_length,
            "completion_tokens": outputs[0].shape[0] - input_length,
            "total_tokens":      outputs[0].shape[0],
        }

        dprint(f"── RAW LLM RESPONSE ({'─' * 43})")
        dprint(raw)
        dprint(f"{'─' * 56}")
        dprint(f"Token usage: {token_usage}")

        # ── Pisah thinking vs content, lalu parse JSON ────────────────────────
        reasoning, content = self._extract_reasoning_and_content(raw)
        dprint(f"── REASONING ({'─' * 50})")
        dprint(reasoning if reasoning else "(tidak ada reasoning)")
        dprint(f"── CONTENT ({'─' * 52})")
        dprint(content)
        dprint(f"{'─' * 56}")

        parsed = self._parse_json_from_content(content)
        dprint(f"── PARSED JSON: {parsed}")
        return parsed

    # --------------------------------------------------------------------------
    # _build_judge_prompt
    # --------------------------------------------------------------------------
    def _build_judge_prompt(self, question: str, sql: str,
                            sql_result: List[tuple], evidence: str = "") -> str:
        if not sql_result:
            result_display = "[]  # Empty result"
        else:
            result_display = "[\n"
            for row in sql_result[:10]:
                result_display += f"  {row},\n"
            if len(sql_result) > 10:
                result_display += f"  ... ({len(sql_result) - 10} more rows)\n"
            result_display += "]"

        prompt = f"""You are an expert SQLlite validator. Your task is to determine if the SQL query result correctly answers the given question.

[QUESTION]
{question}

[EVIDENCE/CONTEXT]
{evidence if evidence else "No additional context provided"}

[SQL QUERY]
{sql}

[SQL EXECUTION RESULT]
{result_display}

[YOUR TASK]
Analyze whether the SQL execution result correctly answers the question. Consider:
1. Does the result contain the right type of data (e.g., names vs IDs, numbers vs text)?
2. Does the result match what the question is asking for?
3. Is the result complete and relevant?

[OUTPUT FORMAT - CRITICAL]
Output ONLY raw JSON. No markdown, no code blocks, no explanation outside JSON.

If CORRECT:
{{"is_correct": true, "reasoning": "Your detailed explanation here"}}

If INCORRECT:
{{"is_correct": false, "reasoning": "Explain what the question wants vs what SQL returned", "sql_output": {result_display}}}

Generate JSON now:"""

        return prompt

    # --------------------------------------------------------------------------
    # judge
    # --------------------------------------------------------------------------
    def judge(self, question: str, sql: str, sql_result: List[tuple],
              evidence: str = "") -> Dict[str, Any]:
        """
        Judge apakah SQL result menjawab question dengan benar.

        Returns:
            {
                "is_correct": bool,
                "reasoning":  str,
                "sql_output": Any  (only if is_correct=false)
            }
        """
        prompt = self._build_judge_prompt(question, sql, sql_result, evidence)
        result = self._generate(prompt)

        if not result:
            return {
                "is_correct":  False,
                "reasoning":   "LLM Judge gagal generate response valid (timeout atau parse error)",
                "sql_output":  sql_result,
                "judge_error": "generation_failed"
            }

        if "is_correct" not in result or "reasoning" not in result:
            return {
                "is_correct":    False,
                "reasoning":     "LLM Judge output tidak memiliki field wajib",
                "sql_output":    sql_result,
                "judge_error":   "invalid_output_format",
                "raw_response":  str(result)[:300]
            }

        # Tambahkan sql_output jika belum ada dan is_correct=false
        if not result["is_correct"] and "sql_output" not in result:
            result["sql_output"] = sql_result

        return result

# ==============================================================================
# SQL EXECUTOR
# ==============================================================================

class SQLExecutor:
    """Execute SQL queries pada SQLite databases."""

    def __init__(self, db_folder: str):
        self.db_folder = db_folder

    def execute_query(self, db_id: str, sql: str,
                      max_rows: int = 50) -> Dict[str, Any]:
        """
        Execute SQL query dan return hasilnya.

        Returns:
            {
                "success":   bool,
                "result":    List[tuple] | None,
                "error":     str | None,
                "row_count": int
            }
        """
        db_path = os.path.join(self.db_folder, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            dprint(f"DB file not found: {db_path}")
            return {
                "success":   False,
                "result":    None,
                "error":     f"Database file not found: {db_path}",
                "row_count": 0
            }

        try:
            conn   = sqlite3.connect(db_path)
            cursor = conn.cursor()
            dprint(f"Executing SQL on [{db_id}]:")
            dprint(sql)
            cursor.execute(sql)
            rows = cursor.fetchmany(max_rows)
            conn.close()
            dprint(f"SQL OK → {len(rows)} row(s) returned")
            if rows:
                dprint(f"Sample rows (max 5): {rows[:5]}")
            return {
                "success":   True,
                "result":    rows,
                "error":     None,
                "row_count": len(rows)
            }
        except Exception as e:
            dprint(f"SQL EXCEPTION: {e}")
            return {
                "success":   False,
                "result":    None,
                "error":     str(e),
                "row_count": 0
            }

# ==============================================================================
# MAIN VALIDATION PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("🚀 PHASE 2: LLM-BASED SQL VALIDATION")
    print("=" * 70)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Log:    {VALIDATION_LOG}")
    mode_label = "🐛 DEBUG (verbose)" if DEBUG else "🚀 PRODUCTION (ringkas)"
    print(f"Mode:   {mode_label}")
    print("=" * 70)

    # Load input data
    print(f"\n📂 Loading data from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"✅ Loaded {len(dataset)} items")

    # Filter hanya item yang pending validation
    pending_items = [item for item in dataset if item.get("status") == "validated"]
    print(f"📋 Items pending validation: {len(pending_items)}")

    if len(pending_items) == 0:
        print("⚠️  No items to validate. Exiting.")
        return

    # Inisialisasi komponen
    logger   = ValidationLogger(VALIDATION_LOG)
    executor = SQLExecutor(TRAIN_DB_FOLDER)
    judge    = LLMJudge(MODEL_ID)

    # Statistik (mengikuti pattern stats dari gen_data)
    stats = {
        "total":               len(pending_items),
        "validated_correct":   0,
        "validated_incorrect": 0,
        "execution_failed":    0,
        "judge_failed":        0,
    }

    start_time    = time.time()
    last_log_time = time.time()
    LOG_INTERVAL  = 5.0          # throttle log — sama dengan gen_data

    validated_dataset = []

    print("\n" + "=" * 70)
    print("🔍 Starting Validation Loop...")
    print("=" * 70 + "\n")

    for idx, item in enumerate(pending_items, 1):
        question_id = item.get("question_id", idx)
        db_id       = item["db_id"]
        question    = item["question"]
        sql         = item["sql"]
        evidence    = item.get("evidence", "")

        print(f"[{idx}/{len(pending_items)}] Validating Q#{question_id} | DB: {db_id}")
        dprint(f"Question : {question}")
        dprint(f"SQL      : {sql}")
        dprint(f"Evidence : {evidence if evidence else '(none)'}")

        # ── Step 1: Execute SQL ───────────────────────────────────────────────
        exec_result = executor.execute_query(db_id, sql, MAX_SQL_RESULT_DISPLAY)

        if not exec_result["success"]:
            print(f"  ❌ SQL Execution Error: {exec_result['error'][:100]}...")
            dprint(f"Full error : {exec_result['error']}")
            dprint(f"DB path    : {os.path.join(TRAIN_DB_FOLDER, db_id, db_id + '.sqlite')}")
            stats["execution_failed"] += 1

            item["status"] = "execution_failed"
            item["validation_result"] = {
                "is_correct":      False,
                "reasoning":       f"SQL execution failed: {exec_result['error']}",
                "execution_error": exec_result["error"]
            }
            item["validation_timestamp"] = datetime.datetime.now().isoformat(timespec='milliseconds')

            logger.log_execution_error(
                question_id=question_id,
                db_id=db_id,
                error=exec_result["error"],
                sql=sql
            )
            validated_dataset.append(item)
            continue

        # ── Step 2: LLM Judge ─────────────────────────────────────────────────
        sql_result = exec_result["result"]
        row_count  = exec_result["row_count"]
        print(f"  ✅ SQL executed. Rows: {row_count}")
        dprint(f"Full SQL result ({row_count} rows): {sql_result}")
        print(f"  🤖 Judging with LLM...")

        judgment = judge.judge(question, sql, sql_result, evidence)
        dprint(f"Judgment dict: {judgment}")

        if judgment.get("judge_error"):
            print(f"  ⚠️  Judge Error: {judgment.get('judge_error')}")
            dprint(f"Raw judgment: {judgment}")
            stats["judge_failed"] += 1
            item["status"] = "judge_failed"

            logger.log_judge_error(
                question_id=question_id,
                db_id=db_id,
                error=judgment.get("judge_error", "unknown")
            )
        else:
            is_correct = judgment["is_correct"]
            reasoning  = judgment["reasoning"]

            if is_correct:
                print(f"  ✅ VALID")
                dprint(f"Reasoning: {reasoning}")
                stats["validated_correct"] += 1
                item["status"] = "valid_quality"
            else:
                print(f"  ❌ INVALID")
                dprint(f"Reasoning  : {reasoning}")
                dprint(f"SQL output : {judgment.get('sql_output')}")
                stats["validated_incorrect"] += 1
                item["status"] = "invalid_quality"

            logger.log_validation_success(
                question_id=question_id,
                db_id=db_id,
                is_correct=is_correct,
                reasoning=reasoning,
                sql_output=judgment.get("sql_output"),
                row_count=row_count
            )

        item["validation_result"]    = judgment
        item["validation_timestamp"] = datetime.datetime.now().isoformat(timespec='milliseconds')
        validated_dataset.append(item)

        # ── Periodic save setiap 50 item (sama dengan gen_data) ──────────────
        if idx % 50 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(validated_dataset, f, indent=2, ensure_ascii=False)
            print(f"  💾 Progress saved ({idx}/{len(pending_items)})")

        # ── Progress log throttled (sama dengan gen_data) ─────────────────────
        now = time.time()
        if now - last_log_time >= LOG_INTERVAL:
            elapsed = now - start_time
            rate    = idx / elapsed if elapsed > 0 else 0
            eta_sec = (len(pending_items) - idx) / rate if rate > 0 else 0
            h, rem  = divmod(int(eta_sec), 3600)
            m, s    = divmod(rem, 60)
            print(
                f"\rProgress: {idx}/{len(pending_items)} | Rate: {rate:.2f} it/s | ETA: {h}h {m}m | "
                f"Valid:{stats['validated_correct']} Invalid:{stats['validated_incorrect']} "
                f"ExecErr:{stats['execution_failed']} JudgeErr:{stats['judge_failed']}",
                end='', flush=True
            )
            last_log_time = now

    # ── Final save ────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(validated_dataset, f, indent=2, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60

    logger.log_session_end(
        total=stats["total"],
        valid=stats["validated_correct"],
        invalid=stats["validated_incorrect"],
        exec_errors=stats["execution_failed"],
        judge_errors=stats["judge_failed"],
        elapsed_min=total_time
    )

    # ── Final summary (mengikuti format gen_data) ─────────────────────────────
    print("\n\n" + "=" * 70)
    print("✅ PHASE 2 VALIDATION COMPLETE")
    print("=" * 70)
    print(f"⏱  Total Time         : {total_time:.2f} minutes")
    print(f"💾 Validated Data     : {OUTPUT_FILE}")
    print(f"📋 Validation Log     : {VALIDATION_LOG}")
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Total Items        : {stats['total']}")
    print(f"   ├─ Valid (Correct) : {stats['validated_correct']} ({stats['validated_correct']/stats['total']*100:.1f}%)")
    print(f"   ├─ Invalid (Wrong) : {stats['validated_incorrect']} ({stats['validated_incorrect']/stats['total']*100:.1f}%)")
    print(f"   ├─ Execution Error : {stats['execution_failed']} ({stats['execution_failed']/stats['total']*100:.1f}%)")
    print(f"   └─ Judge Error     : {stats['judge_failed']} ({stats['judge_failed']/stats['total']*100:.1f}%)")
    print("=" * 70)

    accuracy = stats['validated_correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    if accuracy < 50:
        print("\n⚠️  WARNING: Validation accuracy < 50%")
        print("   Consider reviewing the SQL generation prompts or difficulty scoring.")
    elif accuracy >= 80:
        print("\n🎉 EXCELLENT: Validation accuracy >= 80%")
        print("   The generated SQL queries are highly accurate!")
    else:
        print(f"\n✅ GOOD: Validation accuracy = {accuracy:.1f}%")

    print("\n✅ Validated data ready for training/evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    main()