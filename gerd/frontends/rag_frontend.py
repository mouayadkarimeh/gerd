"""RAG Frontend for GERD.

This module implements a Gradio-based frontend
for the Retrieval-Augmented Generation (RAG) system in GERD.
It allows users to upload documents, ask questions,
and view relevant sources and answers.
"""

import logging
import pathlib
import threading
from typing import Optional

import gradio as gr
from langchain_community.vectorstores import FAISS

from gerd.backends import TRANSPORTER
from gerd.config import CONFIG, load_qa_config
from gerd.models.model import ModelEndpoint
from gerd.transport import DocumentSource, QAFileUpload, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())
_MODEL_SWITCH_LOCK = threading.Lock()
_CURRENT_MODEL = load_qa_config().model.name
_CURRENT_CHUNK_SIZE = load_qa_config().embedding.chunk_size
_CURRENT_CHUNK_OVERLAP = load_qa_config().embedding.chunk_overlap


# -----------------------------
# Model Selection
# -----------------------------
def change_model(name: str) -> None:
    """Change the LLM model used for QA.

    Updates the QA service with the selected model name.

    Parameter:
        name (str): The name of the model to switch to.

    Returns:
        none
    """
    global _CURRENT_MODEL
    with _MODEL_SWITCH_LOCK:
        if name == _CURRENT_MODEL:
            return
        _LOGGER.info("Changing model to: %s", name)
        qa_config = load_qa_config()
        qa_config.model.name = name
        # If the selected model is Qwen3.5, use LM Studio's OpenAI-compatible endpoint.
        # For other models, ensure no remote endpoint is set so the loader will choose
        # the local transformers/llama.cpp path.
        if "qwen3.5" in name.lower():
            qa_config.model.endpoint = ModelEndpoint(
                url="http://localhost:8000",
                type="openai",
                key=None,
            )
        else:
            qa_config.model.endpoint = None
        TRANSPORTER.reinit_qa_service(qa_config)
        # _LOGGER.info("config: %s", qa_config)
        _CURRENT_MODEL = name
        # _LOGGER.info("Model successfully switched to: %s", name )


def change_embedding_param(chunk_size: int, chunk_overlap: int) -> None:
    """Change the embedding parameters for the vectorstore.

    This function is called when the chunk size or overlap sliders are changed.
    It updates the QA service with the new embedding parameters.

    Parameters:
        chunk_size (int): The new chunk size for text splitting.
        chunk_overlap (int): The new chunk overlap for text splitting.

    Returns:
        none
    """
    global _CURRENT_CHUNK_SIZE, _CURRENT_CHUNK_OVERLAP
    with _MODEL_SWITCH_LOCK:
        if (
            chunk_size == _CURRENT_CHUNK_SIZE
            and chunk_overlap == _CURRENT_CHUNK_OVERLAP
        ):
            return
        qa_config = load_qa_config()
        qa_config.model.name = _CURRENT_MODEL
        # If the selected model is Qwen3.5, use LM Studio's OpenAI-compatible endpoint.
        # For other models, ensure no remote endpoint is set so the loader will choose
        # the local transformers/llama.cpp path.
        if "qwen3.5" in _CURRENT_MODEL.lower():
            qa_config.model.endpoint = ModelEndpoint(
                url="http://localhost:8000",
                type="openai",
                key=None,
            )
        else:
            qa_config.model.endpoint = None
        qa_config.embedding.chunk_size = chunk_size
        qa_config.embedding.chunk_overlap = chunk_overlap
        TRANSPORTER.reinit_qa_service(qa_config)
        _CURRENT_CHUNK_SIZE = chunk_size
        _CURRENT_CHUNK_OVERLAP = chunk_overlap


store_set: set[str] = set()


# -----------------------------
# Upload Files
# -----------------------------
def files_changed(file_paths: Optional[list[str]]) -> None:
    """Check if the file upload element has changed.

    If so, upload the new files to the vectorstore and delete the one that
    have been removed.

    Parameters:
        file_paths (Optional[list[str]]): The file paths to upload
    Returns:
        none
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
                f"Datei konnte nicht hochgeladen werden: {res.error_msg} "
                f"(Error Code {res.status})"
            )
            raise gr.Error(msg)
    for delete_file in delete_files:
        store_set.remove(delete_file)
        res = TRANSPORTER.remove_file(pathlib.Path(delete_file).name)
    progress(100, desc="Fertig!")


# -----------------------------
# Query LLM
# -----------------------------
def query(
    question: str,
    k_source: int,
    strategy: str,
    thinking: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, str, str]:
    """Handle the QA query.

    Parameters:
        question (str): The user's question.
        k_source (int): The number of sources to retrieve.
        strategy (str): The search strategy to use ("similarity" or "mmr").
        thinking (str): The thinking mode ("Default", "Think", or "No Think").
        model_name (str): The name of the model to use.
        chunk_size (int): The size of the text chunks for searching.
        chunk_overlap (int): The overlap between text chunks for searching.


    Returns:
        tuple[str, str, str]: A tuple containing the answer, thoughts,
        and the relevant sources.
    """
    # Guarantees query and model selection stay in sync even if events race.
    if model_name != _CURRENT_MODEL:
        _LOGGER.warning(
            "Model mismatch detected before query (%s != %s). Applying switch first.",
            model_name,
            _CURRENT_MODEL,
        )
        change_model(model_name)
    change_embedding_param(chunk_size, chunk_overlap)
    think = None
    if thinking == "Think":
        think = True
    elif thinking == "No Think":
        think = False
    gesamt_context = ""
    _LOGGER.info("think: %s", think)
    q = QAQuestion(
        question=question,
        search_strategy=strategy,
        max_sources=k_source,
        think=think,
    )

    context: list[DocumentSource] = []
    try:
        context = TRANSPORTER.db_query(q)
    except Exception as e:
        _LOGGER.exception("Source retrieval error: %s", e)
    for cnt in context:
        # _LOGGER.info("Retrieved source: %s", cnt.content[:100])
        gesamt_context += cnt.content + "\n" + "******************************" + "\n\n"

    try:
        qa_res = TRANSPORTER.qa_query(q)

        if qa_res.status != 200:
            error_msg = f"Query failed: {qa_res.error_msg} (Code {qa_res.status})"
            raise gr.Error(error_msg) from None
        thoughts = qa_res.thoughts or ""
        return qa_res.response, thoughts, gesamt_context

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
            think_radio = gr.Radio(
                choices=["Default", "Think", "No Think"],
                value="Default",
                label="Thinking Mode",
                info="Thinking-Verhalten für Reasoning Modelle",
            )
            type_radio = gr.Radio(
                choices=[
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen3-0.6B",
                    "Qwen/Qwen3.5-0.8B",
                    "Qwen/Qwen3.5-9B",
                ],
                value=_CURRENT_MODEL,
                label="Model",
            )
            slider_returned_sources = gr.Slider(
                minimum=1, maximum=10, step=1, value=3, label="Number of Sources"
            )
            slider_chunk_size = gr.Slider(
                minimum=64,
                maximum=512,
                step=64,
                value=256,
                label="Chunk Size",
                info="Größe der Textstücke für die Suche",
            )
            slider_chunk_overlap = gr.Slider(
                minimum=0,
                maximum=50,
                step=5,
                value=25,
                label="Chunk Overlap",
                info="Überlappung der Textstücke.",
            )
            strategy_dropdown = gr.Dropdown(
                choices=["similarity", "mmr"],
                value="similarity",
                label="Search Strategy",
                info=(
                    "Suchstrategie anhand der die relevanten Quellen abgerufen werden. "
                    "'similarity' ruft die ähnlichsten Quellen ab, während "
                    "'mmr' (Maximal Marginal Relevance) eine abwechslungsreiche Menge "
                    "an Quellen abruft, die für die Abfrage relevant sind."
                ),
            )
    question_box = gr.Textbox(label="Question", placeholder="Ask a question...")

    with gr.Row():
        source_box = gr.Textbox(
            label="Relevant Sources", lines=10, info="relevante Quellen für RAG Antwort"
        )
        thoughts_box = gr.Textbox(
            label="Thoughts",
            lines=10,
            info="gedankliche Prozesse dzrch Reasining Modell",
        )
        answer_box = gr.Textbox(
            label="Answer", lines=10, info="saubere Antwort ohne Kontext oder Gedanken"
        )

    upload_status = gr.Textbox(label="Upload Status")
    submit_btn = gr.Button("Submit", variant="primary")

    # Events
    file_upload.upload(fn=files_changed, inputs=file_upload, outputs=upload_status)
    file_upload.delete(fn=files_changed, inputs=file_upload, outputs=upload_status)
    file_upload.clear(fn=files_changed, inputs=file_upload, outputs=upload_status)
    type_radio.change(fn=change_model, inputs=type_radio)
    slider_chunk_size.change(
        fn=change_embedding_param, inputs=[slider_chunk_size, slider_chunk_overlap]
    )
    slider_chunk_overlap.change(
        fn=change_embedding_param, inputs=[slider_chunk_size, slider_chunk_overlap]
    )
    submit_btn.click(
        fn=query,
        inputs=[
            question_box,
            slider_returned_sources,
            strategy_dropdown,
            think_radio,
            type_radio,
            slider_chunk_size,
            slider_chunk_overlap,
        ],
        outputs=[answer_box, thoughts_box, source_box],
    )
    question_box.submit(
        fn=query,
        inputs=[
            question_box,
            slider_returned_sources,
            strategy_dropdown,
            think_radio,
            type_radio,
            slider_chunk_size,
            slider_chunk_overlap,
        ],
        outputs=[answer_box, thoughts_box, source_box],
    )
# -----------------------------
# Start App
# -----------------------------
if __name__ == "__main__":
    from gerd.config import CONFIG

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gerd").setLevel(CONFIG.logging.level.value.upper())
    # Show detailed exceptions in the browser UI to make 500 root causes visible.
    demo.launch(show_error=True)
