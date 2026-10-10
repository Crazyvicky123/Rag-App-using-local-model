import io
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer, pipeline


LLM_DIR = os.getenv("LLM_DIR", "Model")                # chat model, e.g. Qwen3-0.6B
EMBED_DIR = os.getenv("EMBED_DIR", "EmbedModel")       # embedding model, e.g. Qwen3-Embedding-0.6B
EMBED_POOLING = os.getenv("EMBED_POOLING", "last")     # "last" for Qwen3-Embedding, "mean" for BGE/MiniLM-style models
EMBED_MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "512"))

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

DOC_PREFIX = os.getenv("DOC_PREFIX", "")
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX",
    "Instruct: Given a question about code or documents, retrieve relevant passages\nQuery: ",
)
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "4"))  # small batches = smoother progress

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

state: dict = {"index": None, "chunks": []}
models: dict = {"llm": None, "embed_tok": None, "embed": None}
store_lock = threading.Lock()
llm_lock = threading.Lock()    # one generation at a time
embed_lock = threading.Lock()  # one embedding batch at a time
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading chat model from '{LLM_DIR}' on {DEVICE}...")
    models["llm"] = pipeline("text-generation", model=LLM_DIR, device=DEVICE, torch_dtype=DTYPE)

    print(f"Loading embedding model from '{EMBED_DIR}' on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(EMBED_DIR, padding_side="left")
    models["embed_tok"] = tok
    models["embed"] = AutoModel.from_pretrained(EMBED_DIR, torch_dtype=DTYPE).to(DEVICE).eval()
    print("Models ready.")

    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        state["index"] = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            state["chunks"] = json.load(f)
    yield


app = FastAPI(title="Local transformers RAG", lifespan=lifespan)


class NoCacheStatic(StaticFiles):
    """Serve the UI with no-store so the browser never shows an outdated page."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store"
        return resp


def log(job: dict, msg: str, keep: bool = True):
    """Set the live message; optionally keep it in the job's log (last 60 lines)."""
    job["message"] = msg
    job["updated_at"] = time.time()
    if keep:
        job["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del job["log"][:-60]


def job_view(job: dict) -> dict:
    t = time.time()
    v = {k: job[k] for k in ("id", "file", "status", "done", "total", "error", "message", "log")}
    v["elapsed"] = round(t - job["started_at"], 1)
    v["idle"] = round(t - job["updated_at"], 1)
    v["batch_running"] = round(t - job["batch_started"], 1) if job["batch_started"] else None
    v["embed_elapsed"] = round(t - job["embed_started"], 1) if job["embed_started"] else None
    return v


def embed(texts: list[str], prefix: str, job: dict | None = None) -> np.ndarray:
    """Embed texts with the local model and L2-normalize."""
    tok, model = models["embed_tok"], models["embed"]
    if model is None:
        raise RuntimeError("Embedding model is not loaded.")
    vecs = []
    n_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH
    for b, i in enumerate(range(0, len(texts), EMBED_BATCH), 1):
        batch = [prefix + t for t in texts[i:i + EMBED_BATCH]]
        if job:
            job["batch_started"] = time.time()
            log(job, f"Embedding batch {b}/{n_batches} (chunks {i+1}-{i+len(batch)} of {len(texts)})")
        enc = tok(batch, padding=True, truncation=True, max_length=EMBED_MAX_LEN,
                  return_tensors="pt").to(DEVICE)
        with embed_lock, torch.no_grad():
            hidden = model(**enc).last_hidden_state
        if EMBED_POOLING == "last":  # left padding -> final position is the last real token
            pooled = hidden[:, -1]
        else:                        # mean pooling over real tokens
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vecs.append(pooled.float().cpu().numpy())
        if job:
            job.update(done=min(i + len(batch), len(texts)), batch_started=None, updated_at=time.time())
    arr = np.concatenate(vecs, axis=0).astype("float32")
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def extract_text(filename: str, data: bytes, job: dict | None = None) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        total, pages = len(reader.pages), []
        for n, page in enumerate(reader.pages, 1):
            pages.append(page.extract_text() or "")
            if job:
                log(job, f"Reading PDF page {n}/{total}", keep=(n == total))
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


def persist():
    faiss.write_index(state["index"], INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(state["chunks"], f)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(4, ge=1, le=10)
    max_tokens: int = Field(512, ge=16, le=4096)
    temperature: float = Field(0.2, ge=0, le=1.5)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(state["chunks"]),
            "llm_loaded": models["llm"] is not None,
            "embed_loaded": models["embed"] is not None,
            "device": DEVICE,
            "active_jobs": sum(j["status"] in ("queued", "reading", "embedding", "saving") for j in jobs.values())}


def run_ingest(job_id: str, filename: str, data: bytes):
    """Background worker. Every step updates the job so the UI can show what is happening."""
    job = jobs[job_id]
    try:
        job["status"] = "reading"
        log(job, f"Reading {filename} ({len(data)/1024:.0f} KB)")
        text = extract_text(filename, data, job)
        chunks = chunk_text(text)
        if not chunks:
            raise RuntimeError("No text found in this file (a scanned PDF needs OCR first).")
        log(job, f"Split into {len(chunks)} chunks")

        job.update(total=len(chunks), done=0, status="embedding", embed_started=time.time())
        vectors = embed(chunks, DOC_PREFIX, job)

        job["status"] = "saving"
        log(job, "Saving vectors to the search index…")
        with store_lock:  # add everything at once so a failed file leaves no partial data
            if state["index"] is None:
                state["index"] = faiss.IndexFlatIP(vectors.shape[1])
            state["index"].add(vectors)
            state["chunks"].extend({"text": c, "source": filename} for c in chunks)
            persist()
        job["status"] = "done"
        log(job, f"Finished: {len(chunks)} chunks are searchable")
    except Exception as e:
        job.update(status="error", error=str(e), batch_started=None)
        log(job, f"ERROR: {e}")


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    data = file.file.read()
    job_id = uuid.uuid4().hex[:8]
    t = time.time()
    jobs[job_id] = {"id": job_id, "file": file.filename, "status": "queued", "done": 0,
                    "total": 0, "error": None, "message": "Queued", "log": [],
                    "started_at": t, "updated_at": t, "batch_started": None, "embed_started": None}
    threading.Thread(target=run_ingest, args=(job_id, file.filename or "upload", data), daemon=True).start()
    return job_view(jobs[job_id])


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return job_view(jobs[job_id])


@app.post("/query")
def query(req: QueryRequest):
    if state["index"] is None or not state["chunks"]:
        raise HTTPException(400, "Index is empty. Ingest documents first.")

    qvec = embed([req.question], QUERY_PREFIX)
    with store_lock:
        scores, ids = state["index"].search(qvec, min(req.top_k, len(state["chunks"])))
    hits = [{**state["chunks"][i], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i != -1]

    context = "\n\n".join(f"[{n+1}] ({h['source']}) {h['text']}" for n, h in enumerate(hits))
    messages = [
        {"role": "system", "content": (
            "You answer questions using ONLY the provided context. "
            "If the answer is not in the context, say you don't know. "
            "Cite sources like [1], [2].")},
        # /no_think turns off Qwen3's long reasoning section
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question} /no_think"},
    ]
    gen_kwargs = {"max_new_tokens": req.max_tokens}
    if req.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=req.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    try:
        with llm_lock:
            result = models["llm"](messages, **gen_kwargs)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    answer = result[0]["generated_text"][-1]["content"]
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return {"answer": answer, "sources": hits}


@app.post("/reset")
def reset():
    with store_lock:
        state["index"], state["chunks"] = None, []
        for f in (INDEX_FILE, CHUNKS_FILE):
            if os.path.exists(f):
                os.remove(f)
    return {"status": "cleared"}


app.mount("/", NoCacheStatic(directory="static", html=True), name="static")