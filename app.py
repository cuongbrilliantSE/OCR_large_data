import os
from pathlib import Path
from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse
from starlette.routing import Route

# Disable Gradio SSR mode so Python handles port 7860 directly
os.environ["GRADIO_SSR_MODE"] = "false"

# Ensure data and log directories exist
Path("data/input").mkdir(parents=True, exist_ok=True)
Path("data/output").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Required for ZeroGPU compatibility on Hugging Face Spaces
try:
    import spaces

    @spaces.GPU(duration=60)
    def gpu_task(prompt: str = "") -> str:
        """GPU function for ZeroGPU validation."""
        return "GPU ready"
except ImportError:
    def gpu_task(prompt: str = "") -> str:
        return "CPU ready"

import gradio as gr
from src.web.app import app as fastapi_app, INDEX_HTML

# Gradio Blocks registering the ZeroGPU function
with gr.Blocks(title="Book OCR Studio AI") as demo:
    gr.Markdown("# Book OCR Studio AI Backend")
    _inp = gr.Textbox(visible=False)
    _out = gr.Textbox(visible=False)
    _btn = gr.Button("Init GPU", visible=False)
    _btn.click(fn=gpu_task, inputs=[_inp], outputs=[_out])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # Launch Gradio server (triggers ZeroGPU validation)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=True,
        ssr_mode=False
    )

    # Mount our Web Studio Dashboard HTML at root / for both GET and HEAD
    async def studio_index(request: Request):
        if not INDEX_HTML.is_file():
            raise HTTPException(status_code=404, detail="Index HTML not found")
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

    demo.server_app.routes.insert(0, Route("/", studio_index, methods=["GET", "HEAD"]))

    # Insert all FastAPI backend routes (/library, /books/{slug}, /api/*, /covers, etc.)
    for r in fastapi_app.routes:
        if hasattr(r, "path") and r.path != "/":
            demo.server_app.routes.insert(0, r)

    # Sync local library and restore persistent records from HF Dataset
    try:
        from src.db.book_repository import BookRepository
        from src.db.hf_dataset_sync import HFDatasetSync
        _repo = BookRepository()
        _syncer = HFDatasetSync()
        _repo.sync_all_books()
        if _syncer.is_configured():
            _syncer.pull_manifest_and_restore(_repo)
    except Exception as e:
        print(f"[app.py] Startup library restore error: {e}")

    # Keep server running
    demo.block_thread()
