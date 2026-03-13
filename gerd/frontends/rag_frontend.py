"""RAG Frontend for GERD.

This module implements a Gradio-based frontend
for the Retrieval-Augmented Generation (RAG)
system in GERD. It allows users to upload documents, ask questions,
and view relevant sources and answers.
"""

import logging
import pathlib
from os import unlink
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional
from xml.dom.minidom import Document

import gradio as gr
from jinja2 import BaseLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gerd.backends import TRANSPORTER
from gerd.config import CONFIG, load_qa_config
from gerd.rag import create_faiss, load_faiss
from gerd.transport import DocumentSource, FileTypes, QAAnswer, QAFileUpload, QAQuestion

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
    if (
        qa_config.embedding.db_path
        and Path(qa_config.embedding.db_path, "index.faiss").exists()
    ):
        _LOGGER.info("faiss path does not exist %s", qa_config.embedding.db_path)
        store = load_faiss(
            qa_config.embedding.db_path,
            qa_config.embedding.model.name,
            qa_config.device,
        )
    else:
        _LOGGER.info(
            "FAISS index not found at %s. Please upload documents to create the index.",
            qa_config.embedding.db_path,
        )
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


store_set: set[str] = set()


# -----------------------------
# Upload Files
# -----------------------------
def files_changed(file_paths: Optional[list[str]]) -> None:
    """Check if the file upload element has changed.

    If so, upload the new files to the vectorstore and delete the one that
    have been removed.

    Parameters:
        file_paths: The file paths to upload
    """
    file_paths = file_paths or []
    progress = gr.Progress()
    new_set = set(file_paths)
    new_files = new_set - store_set
    delete_files = store_set - new_set
    for new_file in new_files:
        store_set.add(new_file)
        with pathlib.Path(new_file).open("rb") as file:
            data = QAFileUpload(
                data=file.read(),
                name=pathlib.Path(new_file).name,
            )
        res = TRANSPORTER.add_file(data)
        if res.status != 200:
            _LOGGER.warning(
                "Data upload failed with error code: %d\nReason: %s",
                res.status,
                res.error_msg,
            )
            msg = (
                f"Datei konnte nicht hochgeladen werden: {res.error_msg}"
                "(Error Code {res.status})"
            )
            raise gr.Error(msg)
    for delete_file in delete_files:
        store_set.remove(delete_file)
        res = TRANSPORTER.remove_file(pathlib.Path(delete_file).name)
    initialize_store()
    progress(100, desc="Fertig!")


def db_query(question: QAQuestion) -> List[DocumentSource]:
    """Queries the vector store with a question.

    The number of sources that are returned is defined by the max_sources parameter
    of the service's configuration.

    Parameters:
        question: The question to query the vector store with.

    Returns:
        A list of document sources
    """
    if not self._vectorstore:
        return []
    return [
        DocumentSource(
            query=question.question,
            content=doc.page_content,
            name=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", 1),
        )
        for doc in self._vectorstore.search(
            question.question,
            search_type=question.search_strategy,
            k=question.max_sources,
        )
    ]


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

    # start db search mode
    db_res = TRANSPORTER.db_query(q)
    if not db_res:
        msg = f"Database query returned empty!"
        raise gr.Error(msg)
    output = ""
    for doc in db_res:
        output += f"{doc.content}\n"
        output += f"({doc.name} / {doc.page})\n----------\n\n"
    return output


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
    file_upload.delete(fn=files_changed, inputs=file_upload, outputs=upload_status)
    file_upload.clear(fn=files_changed, inputs=file_upload, outputs=upload_status)
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
