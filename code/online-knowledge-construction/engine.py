# engine.py

from src.infer.run import initialize_system, process_prompt

models, devices = initialize_system()

def run_inference(question: str, evidence: str = ""):
    return process_prompt(question, evidence, models)