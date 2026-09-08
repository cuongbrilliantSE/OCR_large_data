import os
import uvicorn
from pathlib import Path

# Disable Gradio SSR mode to prevent port 7860 collisions on Hugging Face Spaces
os.environ["GRADIO_SSR_MODE"] = "false"

# Ensure runtime directories exist
Path("data/input").mkdir(parents=True, exist_ok=True)
Path("data/output").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Required for ZeroGPU compatibility on Hugging Face Spaces
try:
    import spaces

    @spaces.GPU(duration=1)
    def _noop():
        """ZeroGPU requirement for Hugging Face Spaces."""
        pass
except ImportError:
    pass

import gradio as gr
from src.web.app import app

# Minimal Gradio block for Hugging Face platform integration
demo = gr.Blocks(title="Book OCR Studio AI Backend")
with demo:
    gr.Markdown("# Book OCR Studio AI Backend")

# Mount Gradio at /gradio with ssr_mode=False so it doesn't hijack port 7860
# This leaves root / and /api/* intact on the FastAPI app to serve our Web Studio Dashboard
mounted_app = gr.mount_gradio_app(app, demo, path="/gradio", ssr_mode=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(mounted_app, host="0.0.0.0", port=port)
