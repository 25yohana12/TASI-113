# src.infer.share_models.py

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.prompts.for_infer import (
    SLM_SYSTEM_PROMPT,
    sql2sr_prompt,
    mask_schema_prompt,
    fill_in_schema_prompt,
    sr2sr_prompt,
)

VERBOSE = False
STOP_STRINGS = [
    "<|tool_call|>",
    "```<|tool_call|>",
    "<|end|>",
    "<|user|>",
    "<|system|>",
    "<|endoftext|>",
]

def log(msg: str):
    if VERBOSE:
        print(msg)

class BaseModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = torch.device(device)

        log(f"🔧 Loading tokenizer from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        except ValueError as e:
            if "TokenizersBackend" in str(e):
                log("⚠️  TokenizersBackend tidak dikenal, fallback ke use_fast=True tanpa tokenizer_class override")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    tokenizer_class=None,
                    use_fast=True,
                )
            else:
                raise

        log(f"🔧 Loading model: {model_path} on {device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()
        self.stop_token_ids = self._build_stop_token_ids()

        log("✅ Model loaded\n")

    def _build_stop_token_ids(self) -> list[int]:
        stop_ids = []
        for tok in ["<|end|>", "<|endoftext|>"]:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            stop_ids.extend(ids)
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        return list(set(stop_ids))

    def re_keywords(self, input_text: str, keyword: str) -> str:
        patterns = [
            rf"```{keyword}\s*(.*?)```",
            rf"```{keyword}\s*(.*?)$",
            r"```\s*(.*?)```",
            r"```\s*(.*?)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, input_text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                extracted = re.sub(
                    rf"^{keyword}\s*", "", extracted, flags=re.IGNORECASE
                ).strip()
                return extracted

        return input_text.strip()

    def infer_single(
        self,
        prompt: str,
        system_prompt_text: str = None,
        max_new_tokens: int = 800,
        keyword: str = "SR",
    ) -> dict:
        messages = []
        if system_prompt_text:
            messages.append({"role": "system", "content": system_prompt_text})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        VALID_GENERATE_KEYS = {"input_ids", "attention_mask"}
        raw_inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {
            k: v.to(self.device)
            for k, v in raw_inputs.items()
            if k in VALID_GENERATE_KEYS
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.05,
                eos_token_id=self.stop_token_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        for stop in STOP_STRINGS:
            if stop in raw_output:
                raw_output = raw_output.split(stop)[0]
        raw_output = raw_output.strip()

        block_match = re.search(rf"(```{keyword}.*?```)", raw_output, re.DOTALL)
        sr_block = block_match.group(1).strip() if block_match else raw_output
        return {"response": sr_block}

class BAM(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)

    def sql2traj_single(self, sql: str, schema_info: str) -> dict:
        prompt = sql2sr_prompt.format(schema=schema_info, sql=sql)
        return self.infer_single(prompt, system_prompt_text=SLM_SYSTEM_PROMPT, keyword="SR")

    def mask_traj_single(self, sr_text: str) -> dict:
        sr_content = self.re_keywords(sr_text, "SR") if "```SR" in sr_text else sr_text
        prompt = mask_schema_prompt.format(sr=sr_content)
        return self.infer_single(prompt, system_prompt_text=SLM_SYSTEM_PROMPT, keyword="SR")

class SAM(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)

    def schema_augment_single(
        self,
        full_schema: str,
        highlighted_schema: str,
        question: str,
        evidence: str,
        masked_sr: str
    ) -> dict:
        masked_content = self.re_keywords(masked_sr, "SR") if "```SR" in masked_sr else masked_sr
        prompt = fill_in_schema_prompt.format(
            schema=full_schema,
            highlighted_schema=highlighted_schema,
            question=question,
            evidence=evidence,
            masked_sr=masked_content,
        )
        return self.infer_single(prompt, system_prompt_text=SLM_SYSTEM_PROMPT, keyword="SR")

class LOM(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)

    def modify_traj_single(
        self,
        schema_text: str,
        question: str,
        evidence: str,
        sr_text: str,
    ) -> dict:
        sr_content = self.re_keywords(sr_text, "SR") if "```SR" in sr_text else sr_text
        prompt = sr2sr_prompt.format(
            schema=schema_text,
            question=question,
            evidence=evidence,
            sr=sr_content,
        )
        return self.infer_single(
            prompt,
            system_prompt_text=SLM_SYSTEM_PROMPT,
            max_new_tokens=800,
            keyword="SR",
        )