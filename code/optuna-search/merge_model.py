import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = {"": 0} if DEVICE == "cuda" else "auto"

base_model_id = "../../models/Phi-4-mini-instruct"
adapter_path = "outputs/lom_model/"
output_path = "../tasi113-evaluation/pipeline/models/lom"

print("Memuat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Memuat base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    return_dict=True,
    torch_dtype=torch.bfloat16, 
    device_map=DEVICE_MAP
)

print("Menyematkan adapter LoRA ke base model...")
# Load model PEFT dengan menggabungkan base model dan adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Proses merging bobot...")
model = model.merge_and_unload()

print("Menyimpan model baru yang sudah disatukan...")
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print("Selesai! Model baru siap dieksekusi.")