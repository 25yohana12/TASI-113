# src.call_models.qwen.py

import re
import torch
from typing import Dict, Tuple
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prompts.for_infer import LLM_SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# =========================================================
# CONFIG
# =========================================================
THINKING_CONFIG = {
    "max_new_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.05,
    "do_sample": True,
}

class LocalModelInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.device = torch.device("cuda:1")
        self.config = THINKING_CONFIG.copy()
        self.verbose = verbose

        if verbose:
            print(f"[*] Loading local model: {model_path}")
            print(f"[*] Device: {self.device}")
            print(f"[*] Mode: THINKING")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:1"},
            trust_remote_code=True
        )

        self.model.eval()

        if verbose:
            print("Model loaded.\n")

    def _extract_reasoning_and_content(self, raw_output: str) -> Tuple[str, str]:
        think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""

        if '</think>' in raw_output:
            content = raw_output.split('</think>')[1].strip()
        else:
            content = raw_output

        content = content.replace('<|im_end|>', '').strip()
        return reasoning, content

    def api_infer(
        self,
        prompt: str,
    ) -> Tuple[str, str, Dict]:

        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        VALID_GENERATE_KEYS = {"input_ids", "attention_mask"}
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k in VALID_GENERATE_KEYS
        }
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=self.config["do_sample"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"],
                min_p=self.config["min_p"],
                repetition_penalty=self.config["repetition_penalty"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=False
        ).strip()

        reasoning, content = self._extract_reasoning_and_content(raw)

        token_usage = {
            "prompt_tokens": input_length,
            "completion_tokens": outputs[0].shape[0] - input_length,
            "total_tokens": outputs[0].shape[0],
        }

        return reasoning, content, token_usage

    def api_infer_stream(
            self,
            prompt: str,
            system_prompt: str = "You are a helpful assistant. Return only the final answer.",
            max_new_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 0.8,
            top_k: int = 20,
        ):
            """Generator: yield token satu per satu"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
    
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
    
            model_inputs = self.tokenizer(
                [text], return_tensors="pt"
            ).to(self.device)
    
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
    
            generate_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True if temperature > 0 else False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
    
            # Generate di thread terpisah agar tidak blocking
            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()
    
            for token in streamer:
                yield token
    
            thread.join()