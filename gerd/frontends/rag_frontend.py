"""RAG Frontend for GERD.

This module implements a Gradio-based frontend
for the Retrieval-Augmented Generation (RAG)
system in GERD. It allows users to upload documents, ask questions,
and view relevant sources and answers.
"""

import logging
from pathlib import Path
from typing import Optional

import gradio as gr
from langchain_community.vectorstores import FAISS

from gerd.backends import TRANSPORTER
from gerd.config import CONFIG, load_qa_config
from gerd.rag import load_faiss
from gerd.transport import QAFileUpload, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())
qa_config = load_qa_config()
model_config = qa_config.model
store: FAISS | None = None


# -----------------------------
# Initialize FAISS Index
# -----------------------------
def initialize_store() -> None:
    """Vector store initialization.

    Sets the `store` variable to a FAISS index if it exists.
    otherwise None. This is called at startup and after file uploads to ensure the index
    is up-to-date.

    Parameter:
        none

    Returns:
        none
    """
    global store
    db_path = Path(qa_config.embedding.db_path)
    index_file = db_path / "index.faiss"
    if not index_file.exists():
        _LOGGER.warning(
            "FAISS index not found at %s. Please upload documents to create the index.",
            index_file,
        )
        store = None
        return
    try:
        store = load_faiss(db_path, qa_config.embedding.model.name, qa_config.device)
        _LOGGER.info("FAISS index loaded.")
        _LOGGER.info("Vectors in index: %d", store.index.ntotal)
    except Exception as e:
        _LOGGER.error("Error loading FAISS index: %s", e)
        store = None


# -----------------------------
# Model Selection
# -----------------------------
def change_model(name: str) -> None:
    """Change the LLM model used for QA.

    Updates the global `model_config` with the selected model name.

    Parameter:
        name (str): The name of the model to switch to.

    Returns:
        none
    """
    model_config.name = name
    _LOGGER.info("Model switched to: %s", name)


# -----------------------------
# Upload Files
# -----------------------------
def files_changed(files: Optional[list[str]]) -> str:
    """Handle file uploads.

    Parameters:
        files (Optional[list[str]]): List of uploaded file paths.

    Returns:
        str: Status message about the upload result.
    """
    global store
    if not files:
        return "No files uploaded."
    result = ""
    for file in files:
        with open(file.name, "rb") as f:
            data = f.read()
        qa_file = QAFileUpload(name=file.name, data=data)
        res = TRANSPORTER.add_file(qa_file)
        if res.status != 200:
            return f"Upload failed: {res.error_msg}"
        result += f"{file.name} indexed successfully\n"

    # Reload FAISS after indexing
    initialize_store()
    if store is not None:
        _LOGGER.info("Vectors after upload: %d", store.index.ntotal)
    return result


# -----------------------------
# Retrieve Sources
# -----------------------------
def return_relevant_sources(question: QAQuestion) -> str:
    """Retrieve relevant sources for a given question.

    Parameters:
        question (QAQuestion): The question object.

    Returns:
        str: the relevant Sources returned as str.
    """
    global store
    if store is None:
        return "Vector store not initialized. Upload documents first."

    # Use the same search method as QAService
    docs = store.search(
        question.question, search_type=question.search_strategy, k=question.max_sources
    )

    if not docs:
        return "No relevant sources found."

    context = "\n\n".join(
        f"📄 Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )
    return context


# -----------------------------
# Query LLM
# -----------------------------
def query(
    question: str, k_source: int, strategy: str, no_think: bool
) -> tuple[str, str]:
    """Handle the QA query.

    Parameters:
        question (str): The user's question.
        k_source (int): The number of sources to retrieve.
        strategy (str): The search strategy to use ("similarity" or "mmr").
        no_think (bool): Whether to disable the "thinking" step in the LLM.

    Returns:
        tuple[str, str]: A tuple containing the answer and the relevant sources.
    """
    _LOGGER.info("no_think: %s", no_think)
    q = QAQuestion(
        question=question,
        search_strategy=strategy,
        max_sources=k_source,
        no_think=no_think,
    )

    try:
        context = return_relevant_sources(q)
    except Exception as e:
        context = f"Source retrieval error: {e}"

    try:
        qa_res = TRANSPORTER.qa_query(q)

        if qa_res.status != 200:
            error_msg = f"Query failed: {qa_res.error_msg} (Code {qa_res.status})"
            raise gr.Error(error_msg) from None
        return qa_res.response, context

    except Exception as e:
        error_msg = f"QA Query failed: {str(e)}"
        raise gr.Error(error_msg) from e


# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.Blocks(title="GERD - RAG Frontend")

with demo:
    gr.Markdown("# GERD - RAG Frontend")
    gr.Markdown("Retrieval-Augmented Generation QA System")

    with gr.Row():
        with gr.Column(scale=2):
            file_upload = gr.Files(
                file_types=[".txt", ".pdf"], label="Upload Documents"
            )
        with gr.Column(scale=2):
            think_box = gr.Checkbox(value=False, label="no_think Mode")
            type_radio = gr.Radio(
                choices=["qwen2.5-0.5B-instruct", "qwen3-0.6B"],
                value="qwen3-0.6B",
                label="Model",
            )
            k_slider = gr.Slider(
                minimum=1, maximum=10, step=1, value=3, label="Number of Sources"
            )
            strategy_dropdown = gr.Dropdown(
                choices=["similarity", "mmr"],
                value="similarity",
                label="Search Strategy",
            )
    question_box = gr.Textbox(label="Question", placeholder="Ask a question...")

    with gr.Row():
        source_box = gr.Textbox(label="Relevant Sources", lines=10)
        answer_box = gr.Textbox(label="Answer", lines=10)

    upload_status = gr.Textbox(label="Upload Status")
    submit_btn = gr.Button("Submit", variant="primary")

    # Events
    file_upload.upload(fn=files_changed, inputs=file_upload, outputs=upload_status)
    type_radio.change(fn=change_model, inputs=type_radio)
    submit_btn.click(
        fn=query,
        inputs=[question_box, k_slider, strategy_dropdown, think_box],
        outputs=[answer_box, source_box],
    )
    question_box.submit(
        fn=query,
        inputs=[question_box, k_slider, strategy_dropdown, think_box],
        outputs=[answer_box, source_box],
    )
# -----------------------------
# Start App
# -----------------------------
if __name__ == "__main__":
    initialize_store()
    from gerd.config import CONFIG

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel(CONFIG.logging.level.value.upper())
    demo.launch()
