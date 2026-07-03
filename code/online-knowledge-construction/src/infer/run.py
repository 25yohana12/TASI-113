# src.infer.run.py
import os
import logging
# Models
from src.infer.share_models import BAM, SAM, LOM
from src.infer.rag_models import RAGRetriever
from src.call_models.qwen import LocalModelInference
# Prompts
from src.prompts.for_infer import (
    initial_sql_prompt,
    ft_sr_to_sql,
)
VERBOSE = False
LLM_DEVICE = "cuda:1"
SLM_DEVICE = "cuda:1"
logger = logging.getLogger(__name__)

def log(msg: str):
    if VERBOSE:
        print(msg)

CHATBOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(CHATBOT_DIR))

BAM_MODEL_PATH = os.path.join(CHATBOT_DIR, "models/bam")
SAM_MODEL_PATH = os.path.join(CHATBOT_DIR, "models/sam")
LOM_MODEL_PATH = os.path.join(CHATBOT_DIR, "models/lom")
GEN_MODEL_PATH = os.path.join(ROOT_DIR, "models/Qwen3-32B")


def initialize_system():
    """ Initialize models with GPU separation """
    log("🚀 Initializing system (Multi-GPU mode)...")
    try:
        generator = LocalModelInference(
            GEN_MODEL_PATH,
            device=LLM_DEVICE
        )
        models = {
            "generator": generator,
            "bam": BAM(BAM_MODEL_PATH, device=SLM_DEVICE),
            "sam": SAM(SAM_MODEL_PATH, device=SLM_DEVICE),
            "lom": LOM(LOM_MODEL_PATH, device=SLM_DEVICE),
            "retriever": RAGRetriever()
        }
        log(f"LLM → {LLM_DEVICE}")
        log(f"SLM → {SLM_DEVICE}")
        log("Models loaded")
        return models
    except Exception as e:
        logger.exception("Initialization failed")
        raise RuntimeError(f"Gagal memuat sistem: {e}")


def process_prompt(question: str, evidence: str, schema: str, models: dict) -> dict:
    try:
        rag_block, _, _ = models['retriever'].retrieve(question)

        schema_context = schema if schema else ""

        init_prompt = initial_sql_prompt.format(
            schema=schema_context,
            rag_block=rag_block,
            question=question,
            evidence=evidence
        )

        _, initial_sql, _ = models['generator'].api_infer(init_prompt)

        sr_initial = models['bam'].sql2traj_single(
            initial_sql, schema_context
        )["response"]
        sr_masked = models['bam'].mask_traj_single(sr_initial)["response"]
        sr_augmented = models['sam'].schema_augment_single(
            full_schema=schema_context,
            highlighted_schema=schema_context,
            question=question,
            evidence=evidence,
            masked_sr=sr_masked
        )["response"]
        final_reasoning = models['lom'].modify_traj_single(
            schema_text=schema_context,
            question=question,
            evidence=evidence,
            sr_text=sr_augmented
        )["response"]

        final_prompt = ft_sr_to_sql.format(
            schema=schema_context,
            fk_dic="{}",
            question=question,
            evidence=evidence,
            sr=final_reasoning
        )

        _, final_sql, _ = models['generator'].api_infer(final_prompt)

        return {"final_sql": final_sql}

    except Exception as e:
        logger.exception("Pipeline error")
        raise RuntimeError(f"Pipeline gagal: {e}")


def process_prompt_stream(question: str, evidence: str, schema: str, models: dict):
    """
    Pipeline sama, tapi final generation pakai streaming.
    BAM/SAM/LOM tetap non-stream (intermediate steps).
    """
    try:
        rag_block, _, _ = models['retriever'].retrieve(question)

        schema_context = schema if schema else ""

        init_prompt = initial_sql_prompt.format(
            schema=schema_context,
            rag_block=rag_block,
            question=question,
            evidence=evidence
        )

        _, initial_sql, _ = models['generator'].api_infer(init_prompt)

        sr_initial   = models['bam'].sql2traj_single(initial_sql, schema_context)["response"]
        sr_masked    = models['bam'].mask_traj_single(sr_initial)["response"]
        sr_augmented = models['sam'].schema_augment_single(
            full_schema=schema_context,
            highlighted_schema=schema_context,
            question=question,
            evidence=evidence,
            masked_sr=sr_masked
        )["response"]
        final_reasoning = models['lom'].modify_traj_single(
            schema_text=schema_context,
            question=question,
            evidence=evidence,
            sr_text=sr_augmented
        )["response"]

        final_prompt = ft_sr_to_sql.format(
            schema=schema_context,
            fk_dic="{}",
            question=question,
            evidence=evidence,
            sr=final_reasoning
        )

        yield from models['generator'].api_infer_stream(final_prompt)

    except Exception as e:
        logger.exception("Stream pipeline error")
        raise