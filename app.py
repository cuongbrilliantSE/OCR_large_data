import os
import uvicorn
from pathlib import Path

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
demo = gr.Blocks(title="Book OCR Studio API")
with demo:
    gr.Markdown("# Book OCR Studio AI Backend")

# Mount Gradio at /gradio, while root / serves our exact Web Studio Dashboard
mounted_app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(mounted_app, host="0.0.0.0", port=port)
