from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import json
from src.infer.run import initialize_system, process_prompt_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SQLRequest(BaseModel):
    question: str
    evidence: str
    db_schema: str

models = None

@app.on_event("startup")
def startup_event():
    global models
    models = initialize_system()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/generate-sql-stream")
def generate_sql_stream(request: SQLRequest):
    """Streaming endpoint — token dikirim langsung ke client"""

    def event_generator():
        try:
            for chunk in process_prompt_stream(
                question=request.question,
                evidence=request.evidence,
                schema=request.db_schema,
                models=models
            ):
                # Format SSE (Server-Sent Events)
                data = json.dumps({"token": chunk})
                yield f"data: {data}\n\n"

            # Sinyal selesai
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # penting untuk nginx/ngrok
        }
    )