"""
NER-based de-identification (runs after 02_deidentify.sql).
Pulls chatbot_messages_deidentified, applies Presidio + spaCy NER for names/locations
regex misses, then writes back.

Usage:
    python deidentify.py
    python deidentify.py --batch-size 500 --skip-ner

Prerequisites:
    gcloud auth application-default login
    python -m spacy download en_core_web_lg
    python -m spacy download xx_core_web_sm
"""

import argparse
import logging
import re
from typing import Optional

import pandas as pd
from google.cloud import bigquery

try:
    from langdetect import detect as _lang_detect, LangDetectException as _LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gcp_config import PROJECT_ID, bq_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SOURCE_TABLE = bq_table("chatbot_messages_deidentified")
DEST_TABLE   = bq_table("chatbot_messages_deidentified")

NER_ENTITY_TYPES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "IP_ADDRESS",
    "LOCATION", "CREDIT_CARD", "IBAN_CODE", "US_SSN",
]

PRESIDIO_PLACEHOLDERS: dict[str, str] = {
    "PERSON":        "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER":  "[PHONE]",
    "URL":           "[URL]",
    "IP_ADDRESS":    "[IP]",
    "LOCATION":      "[LOCATION]",
    "CREDIT_CARD":   "[PAYMENT]",
    "IBAN_CODE":     "[PAYMENT]",
    "US_SSN":        "[PERSONAL_ID]",
    "DOMAIN_NAME":   "[DOMAIN]",
    "ACCOUNT_ID":    "[ACCOUNT_ID]",
}

_NUMBERED_ENTITY_TYPES = frozenset({
    "PERSON", "EMAIL_ADDRESS", "LOCATION", "ORG", "ACCOUNT_ID",
})

# matches [TAG] placeholders in already-deidentified text
_PLACEHOLDER_RE = re.compile(
    r'\[(?P<tag>PERSON|EMAIL|PHONE|URL|IP|LOCATION|ORG|DOMAIN|ACCOUNT_ID|API_KEY|CREDENTIAL|PAYMENT|PERSONAL_ID|OTP)\]'
)


def normalize_placeholders(text: str) -> str:
    """Number repeated placeholders by first-appearance order: [PERSON] → [PERSON_1], [PERSON_2], etc."""
    counters: dict[str, int] = {}
    mapping:  dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        tag = m.group("tag")
        key = f"[{tag}]"
        if key not in mapping:
            counters[tag] = counters.get(tag, 0) + 1
            mapping[key] = f"[{tag}_{counters[tag]}]"
        return mapping[key]

    return _PLACEHOLDER_RE.sub(_replace, text)


def number_entities_in_text(
    text: str,
    entity_registry: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Wraps normalize_placeholders; entity_registry kept for API compatibility."""
    return normalize_placeholders(text), entity_registry

# Regex pre-pass: high-precision patterns that don't require NER
_REGEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', re.I), '[EMAIL]'),
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
    (re.compile(r'https?://[^\s"\'<>()\]]+'), '[URL]'),
    (re.compile(r'\bwww\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'), '[URL]'),
    (re.compile(r'(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*'), '[API_KEY]'),
    (re.compile(r'\b(?:sk|pk|rk|whsec|xoxb|xoxa|xoxp|xoxs|ghp|gho|ghs|ghr)[-_][A-Za-z0-9][A-Za-z0-9_-]{9,}'), '[API_KEY]'),
    (re.compile(r'\+\d[\d\s\-().]{7,20}\d'), '[PHONE]'),
    (re.compile(r'(?i)(?:password|passwd|pwd)\s*(?:is|:|=|was)\s*\S+'), '[CREDENTIAL]'),
    # OTP / verification codes: matches digits preceded by a disclosure phrase
    (re.compile(r'(?i)(?:verification|OTP|one[\s\-]time)\s+(?:code\s+)?(?:is\s+)?\b\d{4,8}\b'), '[OTP]'),
]


def _is_english(text: str) -> bool:
    """Return True if text is detected as English (or detection is inconclusive)."""
    if not _HAS_LANGDETECT or not isinstance(text, str) or len(text.strip()) < 10:
        return True  # Default to applying NER when unsure (more privacy-conservative)
    try:
        return _lang_detect(text) == "en"
    except _LangDetectException:
        return True


def apply_regex(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    for pattern, placeholder in _REGEX_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def _build_presidio(lang_code: str, model_name: str) -> tuple[AnalyzerEngine, AnonymizerEngine]:
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang_code, "model_name": model_name}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[lang_code])

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="DOMAIN_NAME",
        patterns=[Pattern("domain", r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|co|info|biz|dev|app|site|online|cloud|ai|me|us|uk|ca|de|fr|nl|eu|am)\b", 0.6)],
    ))
    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="ACCOUNT_ID",
        patterns=[
            Pattern("saas_id", r"\b(?:acct|site|client|account|user|cust|inv|order|sub|ticket|ref)[-_#]?[0-9]{4,12}\b", 0.85),
            Pattern("uuid",    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", 0.95),
        ],
    ))
    return analyzer, AnonymizerEngine()


_ANALYZER_EN: Optional[AnalyzerEngine] = None
_ANONYMIZER_EN: Optional[AnonymizerEngine] = None

# Multilingual NER: prefer xx_core_web_sm (spaCy 3.8+), else xx_ent_wiki_sm (spaCy 3.7 stacks).
_XX_MODEL_CANDIDATES = ("xx_ent_wiki_sm", "xx_core_web_sm")
_ANALYZER_XX: Optional[AnalyzerEngine] = None
_ANONYMIZER_XX: Optional[AnonymizerEngine] = None
_HAS_XX_MODEL = True


def _get_presidio_en() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    global _ANALYZER_EN, _ANONYMIZER_EN
    if _ANALYZER_EN is None:
        log.info("Initializing Presidio English analyzer (spaCy en_core_web_lg)…")
        _ANALYZER_EN, _ANONYMIZER_EN = _build_presidio("en", "en_core_web_lg")
        log.info("English Presidio ready.")
    return _ANALYZER_EN, _ANONYMIZER_EN


def _get_presidio_xx() -> tuple[Optional[AnalyzerEngine], Optional[AnonymizerEngine]]:
    global _ANALYZER_XX, _ANONYMIZER_XX, _HAS_XX_MODEL
    if not _HAS_XX_MODEL:
        return None, None
    if _ANALYZER_XX is None:
        last_exc: Optional[Exception] = None
        for model_name in _XX_MODEL_CANDIDATES:
            try:
                log.info("Initializing Presidio multilingual analyzer (spaCy %s)…", model_name)
                _ANALYZER_XX, _ANONYMIZER_XX = _build_presidio("xx", model_name)
                log.info("Multilingual Presidio ready (%s).", model_name)
                break
            except Exception as exc:
                last_exc = exc
                continue
        else:
            log.warning(
                "No multilingual spaCy model worked (%s). Non-English rows will be regex-only.\n"
                "Install one of: python -m spacy download xx_core_web_sm  (spaCy 3.8+)\n"
                "             or: python -m spacy download xx_ent_wiki_sm (spaCy 3.7.x)",
                last_exc,
            )
            _HAS_XX_MODEL = False
            return None, None
    return _ANALYZER_XX, _ANONYMIZER_XX


def apply_ner(text: str, language: str = "en") -> str:
    """Run Presidio NER — en_core_web_lg for English, xx_core_web_sm for everything else."""
    if not isinstance(text, str) or not text.strip():
        return text

    if language == "en":
        analyzer, anonymizer = _get_presidio_en()
        lang_code = "en"
    else:
        analyzer, anonymizer = _get_presidio_xx()
        if analyzer is None:
            return text
        lang_code = "xx"

    results: list[RecognizerResult] = analyzer.analyze(
        text=text,
        entities=NER_ENTITY_TYPES + ["DOMAIN_NAME", "ACCOUNT_ID"],
        language=lang_code,
    )
    if not results:
        return text
    operators = {
        r.entity_type: OperatorConfig(
            "replace",
            {"new_value": PRESIDIO_PLACEHOLDERS.get(r.entity_type, f"[{r.entity_type}]")},
        )
        for r in results
    }
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


def deidentify_text(text: str, run_ner: bool = True, language: str = "en") -> str:
    text = apply_regex(text)
    if run_ner:
        text = apply_ner(text, language=language)
    return text


def run(batch_size: int, skip_ner: bool, number_placeholders: bool = False) -> None:
    client = bigquery.Client(project=PROJECT_ID)

    if not _HAS_LANGDETECT:
        log.warning(
            "langdetect not installed — NER will be applied to ALL client messages "
            "regardless of language. Install with: pip install langdetect"
        )

    log.info("Pulling content from %s …", SOURCE_TABLE)
    df = client.query(f"SELECT * FROM `{SOURCE_TABLE}`").to_dataframe(
        progress_bar_type="tqdm"
    )
    log.info("Loaded %d rows.", len(df))

    log.info("Detecting language per row…")
    def _detect_lang(text) -> str:
        if not _HAS_LANGDETECT or not isinstance(text, str) or len(text.strip()) < 10:
            return "en"
        try:
            return _lang_detect(text)
        except _LangDetectException:
            return "en"

    df["_lang"] = df["content"].apply(_detect_lang)
    en_mask  = df["_lang"] == "en"
    non_en_mask = ~en_mask
    log.info(
        "Language split: %d English rows, %d non-English rows (of %d total).",
        int(en_mask.sum()), int(non_en_mask.sum()), len(df),
    )

    if not skip_ner:
        en_indices = df.index[en_mask].tolist()
        n_batches = max(1, (len(en_indices) + batch_size - 1) // batch_size)
        log.info("Running English NER on %d rows (%d batches)…", len(en_indices), n_batches)
        for i, start in enumerate(range(0, len(en_indices), batch_size)):
            batch_idx = en_indices[start : start + batch_size]
            df.loc[batch_idx, "content"] = df.loc[batch_idx, "content"].apply(
                lambda t: deidentify_text(t, run_ner=True, language="en")
            )
            log.info("  English NER batch %d/%d complete (%d rows)", i + 1, n_batches, len(batch_idx))

        non_en_indices = df.index[non_en_mask].tolist()
        n_batches_xx = max(1, (len(non_en_indices) + batch_size - 1) // batch_size)
        log.info("Running multilingual NER on %d rows (%d batches)…", len(non_en_indices), n_batches_xx)
        for i, start in enumerate(range(0, len(non_en_indices), batch_size)):
            batch_idx = non_en_indices[start : start + batch_size]
            df.loc[batch_idx, "content"] = df.loc[batch_idx, "content"].apply(
                lambda t: deidentify_text(t, run_ner=True, language="xx")
            )
            log.info("  Multilingual NER batch %d/%d complete (%d rows)", i + 1, n_batches_xx, len(batch_idx))
    else:
        log.info("NER pass disabled — running regex-only on all rows.")
        df["content"] = df["content"].apply(lambda t: deidentify_text(t, run_ner=False))

    df.drop(columns=["_lang"], inplace=True)

    if number_placeholders:
        log.info("Applying per-conversation numbered placeholder normalization…")
        # Group by conversation_id so numbering is consistent within each thread.
        if "conversation_id" in df.columns:
            def _number_conv(grp: pd.DataFrame) -> pd.DataFrame:
                counters: dict[str, int] = {}
                mapping:  dict[str, str] = {}

                def _replacer(m: re.Match) -> str:
                    tag = m.group("tag")
                    key = f"[{tag}]"
                    if key not in mapping:
                        counters[tag] = counters.get(tag, 0) + 1
                        mapping[key] = f"[{tag}_{counters[tag]}]"
                    return mapping[key]

                grp = grp.copy()
                grp["content"] = grp["content"].apply(
                    lambda t: _PLACEHOLDER_RE.sub(_replacer, t) if isinstance(t, str) else t
                )
                return grp

            df = df.groupby("conversation_id", group_keys=False).apply(_number_conv)
            log.info("Numbered placeholder normalization complete.")
        else:
            log.warning(
                "conversation_id column not found — cannot apply per-conversation "
                "numbered placeholders. Falling back to per-message numbering."
            )
            df["content"] = df["content"].apply(
                lambda t: normalize_placeholders(t) if isinstance(t, str) else t
            )

    log.info("Writing cleaned table back to %s …", DEST_TABLE)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, DEST_TABLE, job_config=job_config)
    job.result()
    log.info("Done. %d rows written.", len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NER-based de-identification pass on chatbot_messages_deidentified."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Rows processed per Presidio batch (default: 1000).",
    )
    parser.add_argument(
        "--skip-ner", action="store_true", default=False,
        help="Run regex-only, skip Presidio NER. Faster but misses person names.",
    )
    parser.add_argument(
        "--number-placeholders", action="store_true", default=False,
        help=(
            "After NER, normalize repeated placeholders to numbered variants "
            "within each conversation: [PERSON] → [PERSON_1], [PERSON_2], etc. "
            "Requires a conversation_id column in the source table."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.batch_size, args.skip_ner, args.number_placeholders)
