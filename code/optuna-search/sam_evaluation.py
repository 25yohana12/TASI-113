import re
import os
import json
import time
import argparse
import warnings
import statistics
import torch
from tqdm import tqdm
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Metrics Libraries
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from bert_score import score as bert_score_func
except ImportError:
    bert_score_func = None

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION CLASS
# ==========================================


class EvalConfig:
    BASE_MODEL = "../../models/Phi-4-mini-instruct"
    LORA_PATH = "outputs/sam_model/"
    TEST_DATA_PATH = "dataset/sam_data/test.json"
    OUTPUT_PATH = "evaluation/sam_model_evaluation.json"

    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.1
    STOP_STRINGS = ["<|end|>", "```\n", "\n```"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================


def extract_sr(text: str) -> str:
    """Mengekstrak konten di dalam blok SR secara presisi."""
    patterns = [r'```SR\s*(.*?)(?:```|$)', r'```\s*(.*?)(?:```|$)']
    for p in patterns:
        match = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def normalize(text: str) -> str:
    """
    Normalisasi teks untuk evaluasi Exact Match (EM).
    Menangani variasi fungsional seperti distinct(x) vs distinct x.
    """
    text = text.lower()
    text = text.replace("(", "").replace(")", "").replace("`", "")
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# 3. CORE EVALUATOR
# ==========================================


class SAMEvaluator:
    """
    Evaluator untuk modul SAM berdasarkan tiga metrik yang ditetapkan
    pada subbab 3.5.2: Exact Match (EM), ROUGE-L, dan BERTScore.
    Setiap metrik berbasis similarity dilaporkan lengkap: Precision, Recall, F1.
    """

    def __init__(self, config: EvalConfig):
        self.cfg = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[*] Loading model: {config.BASE_MODEL}")
        base = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"[*] Merging SAM LoRA: {config.LORA_PATH}")
        self.model = PeftModel.from_pretrained(
            base, config.LORA_PATH).merge_and_unload()
        self.model.eval()

        # Inisialisasi ROUGE-L scorer (Metrik 2)
        if rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
            print("[!] WARNING: rouge_score tidak terinstal. ROUGE-L akan bernilai 0.")

        if not bert_score_func:
            print("[!] WARNING: bert_score tidak terinstal. BERTScore akan bernilai 0.")

    def prompt_sam(self, item: Dict) -> str:
        """Format prompt khusus SAM sesuai trigger training."""
        sys = item.get(
            "system", "You are an expert about text-to-SQL and pandas code.")
        user = item.get("instruction", "")

        # Trigger khusus SAM: Fill in the masked SR
        if "```SR" not in user:
            user += "\n\nNow, fill in the masked SR and give me the final SR:\n```SR\n"
        else:
            # Bersihkan placeholder [Your Answer] jika ada
            user = re.sub(r'```SR\s*\[Your Answer\]\s*```', '', user).strip()
            if not user.endswith("```SR\n"):
                user += "\n```SR\n"

        msgs = [
            {"role": "system", "content": sys},
            {"role": "user",   "content": user}
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def run(self, mode: str = "prod", num_samples: int = None):
        with open(self.cfg.TEST_DATA_PATH, "r") as f:
            data = json.load(f)

        test_data = data[:num_samples] if mode == "debug" else data
        results = []
        all_preds = []
        all_golds = []

        # Akumulator metrik
        metrics_acc = {
            "em":        0.0,  # Metrik 1: Exact Match
            "rougeL_p":  0.0,  # Metrik 2: ROUGE-L Precision
            "rougeL_r":  0.0,  # Metrik 2: ROUGE-L Recall
            "rougeL_f1": 0.0,  # Metrik 2: ROUGE-L F1
            # Metrik 3: BERTScore P/R/F1 dihitung secara batch di akhir
            "latency":   []
        }

        print(
            f"\n🚀 Running in {mode.upper()} mode ({len(test_data)} samples)\n")
        print("📐 Evaluation Metrics : Exact Match (EM) | ROUGE-L (P/R/F1) | BERTScore (P/R/F1)")
        print("=" * 75)

        for idx, item in enumerate(tqdm(test_data, disable=(mode == "debug"))):
            prompt = self.prompt_sam(item)
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.cfg.DEVICE)

            t0 = time.perf_counter()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.MAX_NEW_TOKENS,
                temperature=self.cfg.TEMPERATURE,
                do_sample=(self.cfg.TEMPERATURE > 0),
                stop_strings=self.cfg.STOP_STRINGS,
                tokenizer=self.tokenizer
            )
            latency = time.perf_counter() - t0

            gen_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            pred_sr = extract_sr(gen_text)
            gold_sr = extract_sr(item.get("output", ""))

            all_preds.append(pred_sr if pred_sr else " ")
            all_golds.append(gold_sr if gold_sr else " ")

            # --- Metrik 1: Exact Match (EM) ---
            em = 1.0 if normalize(pred_sr) == normalize(gold_sr) else 0.0

            # --- Metrik 2: ROUGE-L (Precision, Recall, F1) ---
            if self.rouge_scorer:
                rl_scores = self.rouge_scorer.score(gold_sr, pred_sr)['rougeL']
                rougeL_p = rl_scores.precision
                rougeL_r = rl_scores.recall
                rougeL_f1 = rl_scores.fmeasure
            else:
                rougeL_p = rougeL_r = rougeL_f1 = 0.0

            # Debug output per sampel
            if mode == "debug":
                print(f"\n--- SAMPLE {idx + 1} ---")
                print(f"GOLD          : \033[92m{gold_sr}\033[0m")
                print(f"PRED          : \033[94m{pred_sr}\033[0m")
                print(f"EM            : {'✅ 1.0' if em else '❌ 0.0'}")
                print(
                    f"ROUGE-L P/R/F1: {rougeL_p:.4f} / {rougeL_r:.4f} / {rougeL_f1:.4f}")
                print(f"Latency       : {latency:.3f}s")

            metrics_acc["em"] += em
            metrics_acc["rougeL_p"] += rougeL_p
            metrics_acc["rougeL_r"] += rougeL_r
            metrics_acc["rougeL_f1"] += rougeL_f1
            metrics_acc["latency"].append(latency)

            results.append({
                "id":        idx,
                "pred":      pred_sr,
                "gold":      gold_sr,
                "em":        em,
                "rougeL_p":  round(rougeL_p,  4),
                "rougeL_r":  round(rougeL_r,  4),
                "rougeL_f1": round(rougeL_f1, 4),
                # bert_* akan diisi setelah kalkulasi batch
            })

        # --- Metrik 3: BERTScore — Precision, Recall, F1 (kalkulasi batch) ---
        avg_bert_p = avg_bert_r = avg_bert_f1 = 0.0
        if bert_score_func and len(all_preds) > 0:
            print(
                f"\n[*] Menghitung BERTScore untuk {len(all_preds)} sampel...")
            P, R, F1 = bert_score_func(
                all_preds, all_golds,
                lang="en",
                device=self.cfg.DEVICE,
                verbose=False
            )
            avg_bert_p = P.mean().item()
            avg_bert_r = R.mean().item()
            avg_bert_f1 = F1.mean().item()

            for i, res in enumerate(results):
                res["bert_precision"] = round(P[i].item(),  4)
                res["bert_recall"] = round(R[i].item(),  4)
                res["bert_f1"] = round(F1[i].item(), 4)

            # Tabel BERTScore per sampel pada mode debug
            if mode == "debug":
                print("\n--- BERTScore per Sampel ---")
                print(f"  {'ID':<5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
                print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
                for i, res in enumerate(results):
                    print(f"  {i+1:<5} {res['bert_precision']:>10.4f} "
                          f"{res['bert_recall']:>10.4f} {res['bert_f1']:>10.4f}")
        else:
            for res in results:
                res["bert_precision"] = 0.0
                res["bert_recall"] = 0.0
                res["bert_f1"] = 0.0

        # --- Agregasi Akhir ---
        n = len(test_data)
        summary = {
            "exact_match":    round(metrics_acc["em"] / n, 4),
            "rougeL_p":       round(metrics_acc["rougeL_p"] / n, 4),
            "rougeL_r":       round(metrics_acc["rougeL_r"] / n, 4),
            "rougeL_f1":      round(metrics_acc["rougeL_f1"] / n, 4),
            "bert_precision": round(avg_bert_p,                   4),
            "bert_recall":    round(avg_bert_r,                   4),
            "bert_f1":        round(avg_bert_f1,                  4),
            "avg_latency_s":  round(statistics.mean(metrics_acc["latency"]), 4),
            "num_samples":    n,
        }

        # Simpan hasil ke file jika mode produksi
        if mode == "prod":
            os.makedirs(os.path.dirname(self.cfg.OUTPUT_PATH), exist_ok=True)
            output_data = {"summary": summary, "details": results}
            with open(self.cfg.OUTPUT_PATH, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\n✅ Hasil evaluasi disimpan ke: {self.cfg.OUTPUT_PATH}")

        # Laporan akhir
        print("\n" + "=" * 50)
        print("📊 RINGKASAN EVALUASI MODUL SAM")
        print("=" * 50)
        print(f"  {'Exact Match (EM)':<24}: {summary['exact_match']:.4f}")
        print(f"  {'-'*48}")
        print(f"  {'ROUGE-L Precision':<24}: {summary['rougeL_p']:.4f}")
        print(f"  {'ROUGE-L Recall':<24}: {summary['rougeL_r']:.4f}")
        print(f"  {'ROUGE-L F1':<24}: {summary['rougeL_f1']:.4f}")
        print(f"  {'-'*48}")
        print(
            f"  {'BERTScore Precision':<24}: {summary['bert_precision']:.4f}")
        print(f"  {'BERTScore Recall':<24}: {summary['bert_recall']:.4f}")
        print(f"  {'BERTScore F1':<24}: {summary['bert_f1']:.4f}")
        print(f"  {'='*48}")
        print(f"  {'Avg. Latency':<24}: {summary['avg_latency_s']:.4f} s")
        print(f"  {'Total Sampel':<24}: {summary['num_samples']} sampel")
        print("=" * 50)

        return summary


# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluator Modul SAM — EM | ROUGE-L (P/R/F1) | BERTScore (P/R/F1)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["debug", "prod"], default="debug",
        help="'debug': tampilkan output per sampel. 'prod': simpan ke file JSON."
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Jumlah sampel yang digunakan pada mode debug."
    )
    parser.add_argument(
        "--gpu", type=str, default="4",
        help="ID GPU yang digunakan (CUDA_VISIBLE_DEVICES)."
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    evaluator = SAMEvaluator(EvalConfig())
    evaluator.run(mode=args.mode, num_samples=args.samples)
