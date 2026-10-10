# Rag-App-using-local-model
Local RAG app: upload PDFs/text, ask questions, get answers with sources. Runs Qwen3 fully offline with Hugging Face Transformers and FAISS (CPU or GPU, no Ollama or llama.cpp).


# Local RAG with Qwen3

Upload documents (PDF or text) and ask questions about them. Answers come with sources. Everything runs on your own computer, with no internet or API keys needed after the models are downloaded.

## Features
- Chat with your PDF and text files
- Runs offline using Hugging Face Transformers (no Ollama, no llama.cpp)
- FAISS vector search
- Uses GPU automatically if available, otherwise CPU
- Simple web UI with live upload progress

## Requirements
- Python 3.10+
- About 3 GB of free disk space for the models

## Install
```
pip install fastapi uvicorn python-multipart faiss-cpu numpy pypdf transformers torch
```
For an NVIDIA GPU, install the CUDA build of PyTorch instead:
```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Download the models
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='Model')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='EmbedModel')"
```

## Run
```
uvicorn app:app --port 8000
```
Open http://127.0.0.1:8000, upload a file, and ask a question.

## Settings (optional environment variables)
| Variable | Default | Meaning |
| `LLM_DIR` | `Model` | Chat model folder |
| `EMBED_DIR` | `EmbedModel` | Embedding model folder |
| `EMBED_POOLING` | `last` | `last` for Qwen3-Embedding, `mean` for other models |

## Notes
- If you change the embedding model, reset the index first (`/reset`), since vectors from different models can't be mixed.
- Scanned PDFs need OCR first.
