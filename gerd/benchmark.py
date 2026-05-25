"""Benchmarking.py: benchmark between diffrent llms and aproaches.

for GRASCCO dataset with RAG and non-RAG approaches.
gool is to evaluate the performance of the QA system
on specific questions
with RAG and without RAG and
with reasining and non-reasining model
related to patient information extraction from medical documents.
The results are logged into CSV files for further analysis.
"""

import argparse
import csv
import json
import logging
import re
from datetime import date, datetime
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from gerd.backends import TRANSPORTER
from gerd.config import load_qa_config
from gerd.loader import load_model_from_config
from gerd.transport import QAFileUpload, QAQuestion

# Basic logging configuration
_LOGGER = logging.getLogger("gerd.benchmark")
project_dir = Path(__file__).parent.parent


QUESTIONS = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburtstag?",
    "Wann wurde der Patient bei uns aufgenommen?",
    "Wann wurde der Patient bei uns entlassen?",
]
LABEL_MAPPING = {
    "Wie heißt der Patient?": "PatientName",
    "Wann hat der Patient Geburtstag?": "PatientGeburtsdatum",
    "Wann wurde der Patient bei uns aufgenommen?": "AufnahmeDatum",
    "Wann wurde der Patient bei uns entlassen?": "EntlassungsDatum",
}

FUZZY_THRESHOLD = 0.95
RAW = project_dir / "tests/data/grascco/raw"
RESULTS_DIR = project_dir / "results"
RESULT_CSV_RAG = RESULTS_DIR / "grascco_benchmark_rag.csv"
RESULT_CSV_RAG_ONE_FILE = RESULTS_DIR / "grascco_benchmark_rag_one_file.csv"
RESULT_CSV_WITHOUT_RAG = RESULTS_DIR / "grascco_benchmark_no_rag.csv"
RESULT_CSV_WITHOUT_RAG_ONE_FILE = RESULTS_DIR / "grascco_benchmark_no_rag_one_file.csv"

# load llm
"""'
model_config = load_qa_config().model
# Log which model fields are present and what backend will be chosen
backend_choice = (
    "remote"
    if getattr(model_config, "endpoint", None)
    else ("llama.cpp" if getattr(model_config, "file", None) else "transformers")
)
_LOGGER.info(
    "QA model config: name=%s file=%s endpoint=%s -> backend=%s",
    getattr(model_config, "name", None),
    getattr(model_config, "file", None),
    getattr(model_config, "endpoint", None),
    backend_choice,
)
"""
model_config = load_qa_config().model
llm = load_model_from_config(model_config)


# Prefix-rules for each  Label to handle common patterns in predictions
# (e.g. "Patient: Anna Müller" for PatientName)

LABEL_PREFIXES = {
    "PatientName": [
        "patient:",
        "patientin:",
        "der patient heißt",
        "der patient heisst",
        "die patientin:",
        "patientname:",
        "hr.",
        "fr.",
        "herr",
        "frau",
    ],
    "PatientGeburtsdatum": [
        "geburtsdatum:",
        "patient geburtsdatum:",
        "der patient hat geburtsdatum:",
        "dob:",
        "geb.",
        "geburtsdatum",
    ],
    "AufnahmeDatum": [
        "aufnahmedatum:",
        "patient wurde aufgenommen am",
        "aufgenommen am",
        "aufgenommen:",
    ],
    "EntlassungsDatum": [
        "entlassungsdatum:",
        "patient wurde entlassen am",
        "entlassen am",
        "entlassen:",
    ],
}

# Helpers
BAD_PREFIXES = (
    "okay",
    "let's see",
    "the user is asking",
    "based on",
    "from the context",
    "wurde am",
    "entließ",
)

BAD_PREFIX_RE = re.compile(
    rf"^\s*(?:{'|'.join(map(re.escape, BAD_PREFIXES))})[\s,.:;-]*", re.IGNORECASE
)


THINK_BLOCK_RE = re.compile(
    r"<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>", re.IGNORECASE | re.DOTALL
)


# 1. Regex-Definitionen für die Extraktion von Namen,
# Geburtstagen, Aufnahmedaten und Entlassungsdaten


# trigger for patient name: look for "Patient: Anna Müller" or
# "Der Patient heißt Anna Müller" etc.
NAME_TRIGGER = re.compile(
    r"\bpatient\b\s*[:\-]?\s*" r"(?:is|ist|named)?\s*" r"(?:herrn|herr|frau)?\s*",
    re.IGNORECASE,
)


# Name-Regex: 1–3 Words, Unicode, starting with capital letter,
#  allowing common name characters
NAME_RE = re.compile(
    r"\b([A-ZÄÖÜA-Za-z][a-zäöüßà-öø-ÿ]{2,}"
    r"(?:\s+[A-ZÄÖÜA-Za-z][a-zäöüßà-öø-ÿ]{2,}){0,2})\b",
    re.UNICODE,
)

# These Titels should be excluded from names)
TITLE_RE = re.compile(r"\b(dr|doctor|prof|professor|mr|mrs|ms|md|phd)\b", re.IGNORECASE)

# clinics / Organisation keywords, to exclude cases like
# "Patientin wurde in der Charité aufgenommen"
ORG_KEYWORDS = re.compile(
    r"\b(hospital|clinic|klinik|medical|center|centre|university|charité|health|care)\b",
    re.IGNORECASE,
)

# eliminate some non-usefull words
NON_NAME_PREFIX_RE = re.compile(
    r"\b(it|the|mentions|looking|okay|this|that|these|those|user|wie|was|wo|wann)\b",
    re.IGNORECASE,
)

# elliminate Words after dr-Titele
DOCTOR_CUTOFF_RE = re.compile(r"\b(dr|doctor|prof|professor|md)\b", re.IGNORECASE)


# 2. Preprocessing: Think-Tags entfernen
def remove_think_tags(text: str) -> str:
    """Remove entire think/tool_call blocks from text.

    Removes blocks like <think>...</think> and <tool_call>...</tool_call>,
    including their contents, to avoid leaving internal artifacts.
    """
    block_re = re.compile(
        r"<\s*(?:think|tool_call)\s*>.*?<\s*/\s*(?:think|tool_call)\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    return block_re.sub("", text)


# 3. Validation funktion


def is_valid_person_name(name: str) -> bool:
    """Validate if a candidate string is a valid person name.

    Parameter:
        name (str): The candidate name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    return (
        not TITLE_RE.search(name)
        and not ORG_KEYWORDS.search(name)
        and not NON_NAME_PREFIX_RE.match(name)
        and name.lower() != "think"  # Sicherheit gegen Artefakte
    )


def extract_name_from_text(text: str) -> str:
    """Extract patient name from text.

    Parameter:
        text (str): The input text containing the patient name.

    Returns:
        str: The extracted patient name or a default message.
    """
    # eliminate Think-Tags
    cleaned_text = remove_think_tags(text)

    # eliminate words after  Arzt-Titeln
    cutoff = DOCTOR_CUTOFF_RE.search(cleaned_text)
    if cutoff:
        cleaned_text = cleaned_text[: cutoff.start()]

    #  Trigger-Search
    trigger_match = NAME_TRIGGER.search(cleaned_text)
    if trigger_match:
        after_trigger = cleaned_text[trigger_match.end() :]
        candidates = NAME_RE.findall(after_trigger)
        for name in candidates:
            if is_valid_person_name(name):
                return str(name).strip()

    # Fallback: global search
    candidates = NAME_RE.findall(cleaned_text)
    for name in candidates:
        if is_valid_person_name(name):
            return str(name).strip()

    return "Nicht angegeben"


BIRTHDAY_TRIGGERS = re.compile(
    r"(?:"
    r"geburtsdatum|geb\.|geboren|geburtstag||"
    r"born on|date of birth|dob|birthday"
    r")"
    r"(?:\s+(?:am|im|on))?"
    r"[:\s]*",
    re.IGNORECASE,
)


MONTHS = (
    "januar|februar|märz|maerz|april|mai|juni|juli|"
    "august|september|oktober|november|dezember|"
    "january|february|march|april|may|june|july|"
    "august|september|october|november|december"
)


BIRTHDAY_RE = re.compile(
    rf"\b("
    # 15. März 1980 / 15 März 1980
    rf"\d{{1,2}}\.?\s+(?:{MONTHS})\s+\d{{4}}"
    rf"|"
    # March 23, 1968
    rf"(?:{MONTHS})\s+\d{{1,2}},\s*\d{{4}}"
    rf"|"
    # 23.03.1968 / 23-03-68
    rf"\d{{1,2}}[.\-/]\d{{1,2}}[.\-/]\d{{2,4}}"
    rf"|"
    # 1968-03-23
    rf"\d{{4}}[.\-/]\d{{1,2}}[.\-/]\d{{1,2}}"
    rf")\b",
    re.IGNORECASE,
)


def extract_birthday_from_text(text: str) -> str:
    """Extract patient birthday from text.

    Parameter:
        text (str): The input text containing the patient birthday.

    Returns:
        str: The extracted patient birthday or a default message.
    """
    # <tool_call>-Block observe
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    #  Trigger-Search
    trigger_match = BIRTHDAY_TRIGGERS.search(combined_text)
    # _LOGGER.info("birthday trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end() :]
        before_trigger = combined_text[: trigger_match.start()]
        # _LOGGER.info("after birthday trigger:", after_trigger)
        # _LOGGER.info("before birthday trigger:", before_trigger)

        candidates = BIRTHDAY_RE.findall(after_trigger)
        # _LOGGER.info("candidates after trigger:", candidates)
        if not candidates:
            candidates = BIRTHDAY_RE.findall(before_trigger)
            # _LOGGER.info("candidates before trigger:", candidates)

        if candidates:
            return str(candidates[0]).strip()

    # No Fallback anymore
    return "Nicht angegeben"


RECORDING_DATE_TRIGGER = re.compile(
    r"\b(?:aufnahmedatum|aufnahme|admission date|treated|to|wurde|was checked from)"
    r"(?:\s+(?:am|on))?"
    r"[:\s]*",
    re.IGNORECASE,
)


def extract_recording_release_date_from_text(text: str) -> str:
    """Extract patient recording and release date from text.

    Parameter:
        text (str): The input text.

    Returns:
        str: The extracted patient recording and release date.
    """
    # <tool_call>-Block berücksichtigen
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    # Nach Triggern suchen
    trigger_match = RECORDING_DATE_TRIGGER.search(combined_text)
    # _LOGGER.info("recording date trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end() :]
        # _LOGGER.info("after recording date trigger:", after_trigger)

        candidates = RECORDING_RELEASE_DATE_RE.findall(after_trigger)
        # _LOGGER.info("candidates after trigger:", candidates)

        if candidates:
            return str(candidates[0]).strip()

    # Kein Fallback mehr
    return "Nicht angegeben"


RELEASE_DATE_TRIGGER = re.compile(
    r"\b(?:entlassungsdatum|entlassung|entlassen|entließ|release date"
    r"|discharge date|released|discharged|to|bis zum)"
    r"(?:\s+(?:am|on))?"
    r"[:\s]*",
    re.IGNORECASE,
)

RECORDING_RELEASE_DATE_RE = re.compile(
    r"\b("
    # 15. März 1980 / 15 März 1980
    r"\d{1,2}\.?\s+"
    r"(?:januar|februar|märz|maerz|april|mai|juni|juli|"
    r"august|september|oktober|november|dezember)\s+\d{4}"
    r"|"
    # 23 March 1968
    r"\d{1,2}\s+"
    r"(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\s+\d{4}"
    r"|"
    # March 23, 1968 / Jan 24, 2028 / Sep 3, 2021
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},\s+\d{4}"
    r"|"
    # 23.03.1968 / 23-03-68 / 23/03/1968
    r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"
    r"|"
    # 1968-03-23
    r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}"
    r")\b",
    re.IGNORECASE,
)


def extract_date_for_question(text: str, question: str) -> str:
    """Return either the recording or release date based on the question.

    Parameter:
        text (str): The input text.
        question (str): The question to determine which date to return.

    Returns:
        str: The extracted date or a default message.
    """
    cleaned = remove_think_tags(text)

    recording = "Nicht angegeben"
    release = "Nicht angegeben"

    rm = RECORDING_DATE_TRIGGER.search(cleaned)
    if rm:
        after = cleaned[rm.end() :]
        c = RECORDING_RELEASE_DATE_RE.findall(after)
        if c:
            recording = c[0].strip()

    relm = RELEASE_DATE_TRIGGER.search(cleaned)
    if relm:
        after = cleaned[relm.end() :]
        c = RECORDING_RELEASE_DATE_RE.findall(after)
        if c:
            release = c[0].strip()

    q = (question or "").lower()
    if any(k in q for k in ("aufgenommen", "aufnahme", "aufnahmedatum", "admission")):
        return recording
    if any(
        k in q
        for k in ("entlassen", "entlass", "entlassungsdatum", "discharge", "release")
    ):
        return release

    return recording if recording != "Nicht angegeben" else release


def clean_answer_strict(value: str) -> str:
    """Clean answer string strictly.

    Parameter:
        value (str): The input string to clean.

    Returns:
        str: The cleaned string or a default message.
    """
    if not value:
        return "Nicht angegeben"

    # 1. <tool_call>...<tool_call> eliminate
    value = THINK_BLOCK_RE.sub("", value)

    # 2. BAD_PREFIXES eliminate at Begin
    value = BAD_PREFIX_RE.sub("", value)

    # 3.just the first non-empty line
    for line in value.splitlines():
        line = line.strip()
        if line:
            return line

    return "Nicht angegeben"


def normalize(s: str) -> str:
    """Normalize a string.

    Parameter:
        s(str): string to be normalized.

    Returns:
        str
    """
    return " ".join((s or "").lower().strip().split())


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate the Levenshtein distance between two strings.

    Parameter:
        a (str): The first string.
        b (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    a, b = (a or "").lower(), (b or "").lower()
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Calculate the Levenshtein similarity ratio between two strings.

    Parameter:
        a (str): The first string.
        b (str): The second string.

    Returns:
        float: The Levenshtein similarity ratio between the two strings.
    """
    a, b = a or "", b or ""
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return (max_len - dist) / max_len


def safe_slice_text(text: Optional[str], start: int, end: int) -> str:
    """Safely slice text between start and end indices, handling edge cases.

    Parameter:
        text (Optional[str]): The text to slice.
        start (int): The starting index.
        end (int): The ending index.

    Returns:
        str: The sliced text or an empty string.
    """
    if not text:
        return ""
    start = max(0, int(start))
    end = min(len(text), int(end))
    if start >= end:
        return ""
    return text[start:end]


def normalize_date(s: Optional[str]) -> Optional[date]:
    """Normalize date string to date object.

    Parameter:
        s (Optional[str]): The date string to normalize.

    Returns:
        Optional[date]: The normalized date object or None.
    """
    if not s:
        return None

    def _try_parse(value: str, fmt: str) -> Optional[date]:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            return None

    try:
        value = s.strip()
        for fmt in (
            "%d.%m.%Y",
            "%d.%m.%y",
            "%d-%m-%Y",
            "%d-%m-%y",
            "%d/%m/%Y",
            "%d/%m/%y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%d. %B %Y",
            "%d %B %Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ):
            parsed = _try_parse(value, fmt)
            if parsed is not None:
                return parsed

        month_map = {
            "januar": 1,
            "februar": 2,
            "märz": 3,
            "maerz": 3,
            "april": 4,
            "mai": 5,
            "juni": 6,
            "juli": 7,
            "august": 8,
            "september": 9,
            "oktober": 10,
            "november": 11,
            "dezember": 12,
            "january": 1,
            "february": 2,
            "march": 3,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        month_match = re.match(r"^(\d{1,2})\.??\s+([A-Za-zÄÖÜäöüß]+)\s+(\d{4})$", value)
        if month_match:
            day = int(month_match.group(1))
            month_name = month_match.group(2).lower()
            year = int(month_match.group(3))
            month = month_map.get(month_name)
            if month:
                return date(year, month, day)
    except Exception:
        return None
    return None


DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
DATE_DAYMONTH_RE = re.compile(r"\b\d{1,2}\.\s*[A-Za-zäöüÄÖÜß]+\s*\d{4}\b")


def extract_date(text: Optional[str]) -> Optional[date]:
    """Extract date from text using multiple strategies.

    Parameter:
        text (Optional[str]): The text from which to extract the date.

    Returns:
        Optional[date]: The extracted date or None.
    """
    if not text:
        return None
    t = str(text)
    # 1) numerics like 21.01.2023 or 21/01/2023
    m = DATE_NUMERIC_RE.search(t)
    if m:
        return normalize_date(m.group(0))
    # 2) day + monthname + year (z.B. 21. Januar 2023)
    m2 = DATE_DAYMONTH_RE.search(t)
    if m2:
        return normalize_date(m2.group(0))
    # 3) attempt fuzzy parse
    return normalize_date(t)


def clean_gt_value(label: str, value: Optional[str]) -> str:
    """Clean the ground truth value based on the label.

    Parameter:
        label (str): The label for which to clean the value.
        value (Optional[str]): The ground truth value to clean.

    Returns:
        str: The cleaned ground truth value.
    """
    v = (value or "").strip()
    if not v:
        return ""
    if label == "PatientName":
        v = re.sub(r"^(tr\.|hr\.|fr\.|herr|frau)\s*:??\s*", "", v, flags=re.I).strip()
        return v
    if label == "PatientGeburtsdatum":
        v = re.sub(r"^(geb\.|geburtsdatum:)\s*", "", v, flags=re.I).strip()
        return v
    return v


DATE_LABELS = {"PatientGeburtsdatum", "AufnahmeDatum", "EntlassungsDatum"}


def _matches_label_prefix(
    pred_norm: str, label: str, gt_raw: str, pred_raw: str
) -> bool:
    """Check whether a prediction matches a label-specific prefix rule."""
    prefixes = LABEL_PREFIXES.get(label, [])
    gt_norm = normalize(gt_raw)
    for prefix in prefixes:
        prefix_norm = prefix.lower()
        if not pred_norm.startswith(prefix_norm):
            continue

        rest = pred_norm[len(prefix_norm) :].strip()
        if label == "PatientName":
            if normalize(remove_titles(rest)) == normalize(remove_titles(gt_norm)):
                return True
            continue

        if label in DATE_LABELS:
            d1 = extract_date(gt_raw)
            d2 = extract_date(pred_raw)
            if d1 and d2 and d1 == d2:
                return True

    return False


def _evaluate_date_prediction(
    gt_raw: str, pred_raw: str, gt_norm: str, pred_norm: str
) -> Tuple[float, bool]:
    """Evaluate date labels.

    Parameters:
        gt_raw (str): The raw ground truth value.
        pred_raw (str): The raw predicted value.
        gt_norm (str): The normalized ground truth value.
        pred_norm (str): The normalized predicted value.

    Returns:
        Tuple[float, bool]: The similarity score and whether it is considered correct.

    """
    d1 = extract_date(gt_raw)
    d2 = extract_date(pred_raw)
    if d1 and d2 and d1 == d2:
        return 1.0, True
    if gt_norm and pred_norm and gt_norm == pred_norm:
        return 1.0, True
    score = levenshtein_ratio(gt_norm, pred_norm)
    return score, score >= FUZZY_THRESHOLD


def _evaluate_name_prediction(gt_norm: str, pred_norm: str) -> Tuple[float, bool]:
    """Evaluate name labels.

    Parameters:
        gt_norm (str): The normalized ground truth value.
        pred_norm (str): The normalized predicted value.

    Returns:
        Tuple[float, bool]: The similarity score.
    """
    gt_clean = normalize(remove_titles(gt_norm))
    pred_clean = normalize(remove_titles(pred_norm))

    if gt_clean and pred_clean and (gt_clean in pred_clean or pred_clean in gt_clean):
        return 1.0, True

    if gt_clean and pred_clean and set(gt_clean.split()) == set(pred_clean.split()):
        return 1.0, True

    score = levenshtein_ratio(gt_clean, pred_clean)
    return score, score >= FUZZY_THRESHOLD


def _evaluate_generic_prediction(gt_norm: str, pred_norm: str) -> Tuple[float, bool]:
    """Evaluate remaining labels.

    Parameters:
        gt_norm (str): The normalized ground truth value.
        pred_norm (str): The normalized predicted value.

    Returns:
        Tuple[float, bool]: The similarity score and whether it is considered correct.
    """
    score = levenshtein_ratio(gt_norm, pred_norm)
    return score, score >= FUZZY_THRESHOLD


def remove_titles(s: Optional[str]) -> str:
    """Remove common titles like "Herr", "Frau", etc.

    Parameter:
        s (Optional[str]): Der Text, aus dem Titel entfernt werden sollen.

    Returns:
        str: Der Text ohne Titel.
    """
    if not s:
        return ""
    return re.sub(r"\b(?:herr|frau|hr\.|fr\.|dr\.?|dr)\b", "", s, flags=re.I).strip()


# robust extraction from GRASCCO annotations


def load_grascco_annotations() -> Dict[str, dict]:
    """Load GRASCCO annotations from JSON file and create a mapping.

    Returns:
        Dict[str, dict]: The created mapping.
    """
    json_path = project_dir / "tests/data/grascco/grascco.json"
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, dict] = {}
    for entry in data:
        file_upload = entry.get("file_upload", "") or ""
        name = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload
        # normalize simple umlauts for file matching
        name_norm = name.replace("ö", "o").replace("ä", "a").replace("ü", "u")
        mapping[name_norm] = entry
    return dict(sorted(mapping.items()))


def _first_label_result(annotation_entry: dict, label: str) -> dict | None:
    """Return the first annotation result that matches a label.

    Parameter:
        annotation_entry (dict): Der Annotationseintrag.
        label (str): Das Label, für das der Text extrahiert werden soll.

    Returns:
        dict | None: Das erste passende Annotationsergebnis oder None.
    """
    if not annotation_entry or "annotations" not in annotation_entry:
        return None

    annotations = annotation_entry["annotations"]
    if not annotations:
        return None

    results = cast("list[dict[str, Any]]", annotations[0].get("result", []))
    for result in results:
        labels = result.get("value", {}).get("labels") or []
        if labels and labels[0] == label:
            return result
    return None


def _slice_annotation_candidate(text: str, start: int | None, end: int | None) -> str:
    """Extract a candidate snippet from text around a label annotation.

    Parameter:
        text (str): The text from which to extract the candidate.
        start (int | None): The starting position of the annotation.
        end (int | None): The ending position of the annotation.

    Returns:
        str: The extracted candidate snippet or an empty string.
    """
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    ):
        candidate = safe_slice_text(text, start, end).strip()
        if candidate:
            return candidate

    text_norm = text.replace("\r\n", "\n")
    for delta in range(-12, 13):
        slice_start = max(0, (start or 0) + delta)
        slice_end = min(len(text_norm), (end or 0) + delta)
        if slice_start < slice_end:
            part = text_norm[slice_start:slice_end].strip()
            if part and len(part) <= 200:
                return part

    tokens = re.findall(r"\S+", text_norm)
    if tokens and isinstance(start, int):
        char_pos = 0
        for idx, token in enumerate(tokens):
            token_start = char_pos
            token_end = char_pos + len(token)
            if token_start <= start <= token_end:
                snippet = " ".join(tokens[max(0, idx - 2) : min(len(tokens), idx + 3)])
                return snippet.strip()
            char_pos = token_end + 1

    return ""


def extract_label_text(annotation_entry: dict, label: str, text: str) -> str:
    """Extract the text for a given label from the annotation entry.

    Parameter:
        annotation_entry (dict): Der Annotationseintrag.
        label (str): Das Label, für das der Text extrahiert werden soll.
        text (str): Der Text, aus dem der Wert extrahiert werden soll.

    Returns:
        str: Der extrahierte Text oder ein leerer String.
    """
    if not text:
        return ""

    result = _first_label_result(annotation_entry, label)
    if result is None:
        return ""

    value = result.get("value", {})
    start = value.get("start")
    end = value.get("end")

    try:
        candidate = _slice_annotation_candidate(text, start, end)
        if candidate:
            return candidate
    except Exception:
        _LOGGER.exception("Fehler beim Extrahieren der Annotation")

    return ""


# Prediction cleaning


def clean_pred(answer: Optional[str]) -> str:
    """Clean the prediction answer.

    Parameter:
        answer (Optional[str]): The raw prediction answer to clean.

    Returns:
        str: The cleaned prediction answer.
    """
    if not answer:
        return ""
    answer = answer.strip()
    lower = answer.lower()

    # try to extract common numeric date patterns first
    m = DATE_NUMERIC_RE.search(answer)
    if m:
        return m.group(0)
    m2 = DATE_DAYMONTH_RE.search(answer)
    if m2:
        return m2.group(0)

    # common declarative prefixes
    bad_prefixes = [
        "der patient wurde aufgenommen am",
        "der patient wurde entlassen am",
        "der patient heißt",
        "der patient heisst",
        "die patientin heißt",
        "patientin:",
        "patient:",
        "antwort:",
        "unbekannt",
        "nicht angegeben",
    ]

    for prefix in bad_prefixes:
        if lower.startswith(prefix):
            return answer[len(prefix) :].strip(" :.,")

    # if nothing matched, return stripped answer
    return answer


# Evaluation / Matching


def evaluate_prediction(gt: str, pred: str, label: str) -> Tuple[float, bool]:
    """Evaluates the prediction against the ground truth with heuristics.

    Parameter:
        gt (str): The ground truth value.
        pred (str): The prediction value.
        label (str): The label for the evaluation.

    Returns:
        Tuple[float, bool]: A tuple containing the evaluation score
        and a boolean indicating if the prediction is correct.
    """
    gt_raw = clean_gt_value(label, gt or "")
    pred_raw = pred or ""

    gt_norm = normalize(gt_raw)
    pred_norm = normalize(pred_raw)

    # 1) empty GT => no match
    if not gt_norm:
        return 0.0, False

    # 2) Prefix-basiertes Exact-Match (falls prediction "patient: Anna")
    if _matches_label_prefix(pred_norm, label, gt_raw, pred_raw):
        return 1.0, True

    # 3) Datum-Labels: extrahiere und vergleiche robust
    if label in DATE_LABELS:
        return _evaluate_date_prediction(gt_raw, pred_raw, gt_norm, pred_norm)

    # 4) Name-Label: heuristiken + levensthein
    if label == "PatientName":
        return _evaluate_name_prediction(gt_norm, pred_norm)

    # 5) Allgemeiner Fallback
    return _evaluate_generic_prediction(gt_norm, pred_norm)


def make_prompt_german(text: str, question: str, no_think: bool) -> str:
    """Create german prompts.

    Parameter:
        text (str): The input text for the prompt.
        question (str): The question to be answered based on the text.
        no_think (bool): Whether to activate the '/no_think' tag.

    Returns:
        str: The generated prompt string.
    """
    question = f"/no_think {question}" if no_think else question

    return f"""Gib ausschließlich den exakten Wortlaut zurück, wie er im Text vorkommt.
Keine vollständigen Sätze. Keine Erklärungen. Keine zusätzlichen Wörter.
Gib nur den wörtlich relevanten Text zurück.
Antworte mit 'Unbekannt', falls die Information im Text nicht enthalten ist.


Text:
{text}

Frage:
{question}
""".strip()


def make_prompt_englich(text: str, question: str, no_think: bool) -> str:
    """Create english prompts.

    Parameter:
        text (str): The input text for the prompt.
        question (str): The question to be answered based on the text.
        no_think (bool): Whether to activate the '/no_think' tag.

    Returns:
        str: The generated prompt string.
    """
    question = f"/no_think {question}" if no_think else question

    return f"""Return only the exact wording as it appears in the text.
No full sentences. No explanations. No additional words.
Return the literal relevant text only.
Answer with 'Unknown' if the information is not present in the text.


Text:
{text}

Question:
{question}
""".strip()


def _extract_prediction_text(qa_res: object) -> str:
    """Extract the prediction text from different response shapes.

    Parameter:
        qa_res (object): The raw response from the LLM or RAG query.

    Returns:
        str: The extracted prediction text or an empty string if it cannot be extracted.
    """
    if isinstance(qa_res, tuple) and len(qa_res) == 2:
        return str(qa_res[1] or "")
    if hasattr(qa_res, "response"):
        return str(getattr(qa_res, "response", "") or "")
    if isinstance(qa_res, dict) and "response" in qa_res:
        return str(qa_res.get("response") or "")
    return ""


def _process_benchmark_questions(
    fname: str,
    annotation_entry: dict,
    text: str,
    use_rag: bool,
    no_think_option: bool,
    question_key: str,
) -> tuple[list[dict], int, int]:
    """Run the benchmark questions for a single file.

    Parameter:
        fname (str): The name of the file being processed.
        annotation_entry (dict): The annotation entry for the file.
        text (str): The text content of the file.
        use_rag (bool): Whether to use the RAG approach.
        no_think_option (bool): Whether to activate the '/no_think' option in prompts.
        question_key (str): The key to use for the question in the results.

    Returns:
        tuple[list[dict], int, int]: A tuple containing the list of result rows,
    """
    rows: list[dict] = []
    total = 0
    correct = 0

    for question in QUESTIONS:
        total += 1
        label = LABEL_MAPPING[question]
        truth_answer = extract_label_text(annotation_entry, label, text)
        prompt = make_prompt_englich(
            text=text, question=question, no_think=no_think_option
        )

        if use_rag:
            qa_result: object = TRANSPORTER.qa_query(
                QAQuestion(
                    question=prompt,
                    search_strategy="similarity",
                    max_sources=3,
                )
            )
        else:
            try:
                qa_result = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}]
                )
            except Exception as exc:
                _LOGGER.warning(
                    "WARNUNG: LLM-Aufruf für %s ist fehlgeschlagen: %s", fname, exc
                )
                qa_result = None

        pred_answer = _extract_prediction_text(qa_result)
        score, match = evaluate_prediction(truth_answer, pred_answer, label)
        if match:
            correct += 1

        _LOGGER.info(
            "%s | %s | Q: %s | GT: %s | PRED: %s | score=%.2f | match=%s",
            fname,
            label,
            question,
            truth_answer,
            pred_answer,
            score,
            match,
        )

        rows.append(
            {
                "file": fname,
                question_key: question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred_answer,
                "match_score": f"{score:.3f}",
                "match": match,
            }
        )

    return rows, total, correct


# BENCHMARK-Funktions


def run_bmk(max_files: int, use_rag: bool, no_think_option: bool) -> None:
    """Run benchmark for multiple files.

    run benchmark with or without RAG approach for multiple files.
    run benchmark with or without 'no_think' option for multiple files.
    test multiple files against the provided questions .
    truth answers are extracted from the grascco annotations.

    Parameters:
        max_files (int): The maximum number of files to process.
        use_rag (bool): Whether to use the RAG approach.
        no_think_option (bool): Whether to use the 'no_think' option.

    Returns:
        None
    """
    _LOGGER.info("**************Benchmark___Rag*********")
    _LOGGER.info("Model: %s", model_config.name)
    _LOGGER.info("'no_think': %s", no_think_option)
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    files = islice(sorted(RAW.glob("*.txt")), max_files)

    for file_path in files:
        fname = file_path.name.replace("ö", "o")

        if fname not in ann:
            _LOGGER.warning("No annotation for %s", fname)
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text(encoding="utf-8")

        # ---
        # Reset + Upload Vectorstore
        # ---
        if use_rag:
            clear_fn = getattr(TRANSPORTER, "clear_vectorstore", None)
            if callable(clear_fn):
                try:
                    clear_fn()
                except Exception:
                    _LOGGER.exception("Failed to reset vectorstore")
                    continue
            upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
            res = TRANSPORTER.add_file(upload)
            if getattr(res, "status", None) != 200:
                _LOGGER.error("Upload failed for %s", fname)
                continue

        batch_rows, batch_total, batch_correct = _process_benchmark_questions(
            fname=fname,
            annotation_entry=annotation_entry,
            text=text,
            use_rag=use_rag,
            no_think_option=no_think_option,
            question_key="question_used",
        )
        rows.extend(batch_rows)
        total += batch_total
        correct += batch_correct

    # ---
    # Final metrics
    # ---
    acc = correct / total if total else 0.0
    _LOGGER.info("================================================")
    _LOGGER.info("FINAL SCORE: %d / %d accuracy=%.3f", correct, total, acc)
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        _LOGGER.info("Results saved in: %s", RESULT_CSV_RAG)


def profile(file_path: Path, use_rag: bool, no_think: bool) -> None:
    """Profile a single file and extract key information.

    This function processes a single file to extract key information
    such as patient name, birth date, recording date, and release date.
    It uses either a RAG approach or direct LLM querying
    based on the parameters provided.
    The extracted information is then formatted into a profile
    and saved to a text file.

    Parameter:
        file_path (Path): The path to the file to be profiled.
        use_rag (bool): Whether to use the RAG approach.
        no_think (bool): Whether to use the 'no_think' option.

    Returns:
        None
    """
    _LOGGER.info("********Profile*********")
    text = file_path.read_text(encoding="utf-8")
    _LOGGER.info("Datei: %s", file_path.name)
    _LOGGER.info("Model: %s", model_config.name)
    _LOGGER.info("RAG: %s", use_rag)
    _LOGGER.info("no_think: %s", no_think)
    answers: dict[str, str] = {}

    # =========================
    # RAG-SETUP
    # =========================
    if use_rag:
        clear_fn = getattr(TRANSPORTER, "clear_vectorstore", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception:
                _LOGGER.exception("Fehler beim Zurücksetzen des Vectorstores")

        upload = QAFileUpload(data=text.encode("utf-8"), name=file_path.name)
        upload_res = TRANSPORTER.add_file(upload)
        if getattr(upload_res, "status", None) != 200:
            msg = "Upload in Vectorstore fehlgeschlagen"
            raise RuntimeError(msg)

    # =========================
    # FRAGEN-SCHLEIFE
    # =========================
    for question in QUESTIONS:
        if use_rag:
            prompt = make_prompt_englich(text, question, no_think)
            q = QAQuestion(
                question=prompt,
                search_strategy="similarity",
                max_sources=3,
            )

            qa_res: object = TRANSPORTER.qa_query(q)
            raw_answer = _extract_prediction_text(qa_res)

        else:
            prompt = make_prompt_englich(text, question, no_think)
            llm_res: object = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )

            raw_answer = _extract_prediction_text(llm_res)

        answers[question] = raw_answer
        _LOGGER.info("Frage: %s | Antwort: %r", question, raw_answer)

    # =========================
    # SORTIERTE TEXTAUSGABE
    # =========================
    """
    _LOGGER.info("Wie heißt der Patient?:",
        answers.get("Wie heißt der Patient?", "Nicht angegeben"))
    _LOGGER.info("******************************")
    _LOGGER.info("Wann hat der Patient Geburtstag?",
        answers.get("Wann hat der Patient Geburtstag?", "Nicht angegeben"))
    _LOGGER.info("******************************")
    _LOGGER.info("Wann wurde der Patient bei uns aufgenommen?",
        answers.get("Wann wurde der Patient bei uns aufgenommen?", "Nicht angegeben"))
    _LOGGER.info("******************************")
    _LOGGER.info("Wann wurde der Patient bei uns entlassen?",
        answers.get("Wann wurde der Patient bei uns entlassen?", "Nicht angegeben"))
    _LOGGER.info("******************************")
    """

    lines = [
        "Kurzprofil Entlassungsbrief\n",
        f"Patient: {extract_name_from_text(
            answers.get('Wie heißt der Patient?'
                        , 'Nicht angegeben'))}",
        f"Geburtsdatum: {extract_birthday_from_text(
            answers.get('Wann hat der Patient Geburtstag?'
                        , 'Nicht angegeben'))}",
        f"Aufnahme: {extract_recording_release_date_from_text(
            answers.get('Wann wurde der Patient bei uns aufgenommen?'
                        , 'Nicht angegeben'))}",
        f"Entlassung: {extract_recording_release_date_from_text(
            answers.get('Wann wurde der Patient bei uns entlassen?'
                        , 'Nicht angegeben'))}",
    ]

    summary = "\n".join(lines).strip()

    # =========================
    # SPEICHERN
    # =========================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{file_path.stem}_profile.txt"
    out_path.write_text(summary, encoding="utf-8")
    _LOGGER.info("Profil gespeichert unter: %s", out_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GRASCCO benchmark/profile.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    p = argparse.ArgumentParser(description="Run GRASCCO benchmark/profile")
    p.add_argument(
        "--file-path",
        dest="file_path",
        help=f"Path to input text file (default: {RAW / 'Cajal.txt'})",
    )
    p.add_argument(
        "--use-rag",
        dest="use_rag",
        action="store_true",
        help="Use RAG approach (default: False unless --use-rag is provided)",
    )
    p.add_argument(
        "--no-think",
        dest="no_think",
        action="store_true",
        help="Set no_think flag for prompts",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    args = parse_args()

    # determine file path
    if args.file_path:
        val = args.file_path
        # allow shorthand RAW/filename
        if val.startswith("RAW"):
            rel = val.split("RAW", 1)[1].lstrip("/\\")
            fp = RAW / rel
        else:
            fp = Path(val)
    else:
        fp = RAW / "Cajal.txt"
    fp = fp.expanduser()

    # default for use_rag and no_think is False unless provided
    # examples how to run profile with different options in CLI:
    # run uv benchmark.py --file-path RAW/Cajal.txt --use-rag --no-think
    # run uv benchmark.py --file-path RAW/Cajal.txt --use-rag
    # run uv benchmark.py --file-path RAW/Cajal.txt --no-think
    # run uv benchmark.py --file-path RAW/Cajal.txt
    profile(file_path=fp, use_rag=bool(args.use_rag), no_think=bool(args.no_think))
