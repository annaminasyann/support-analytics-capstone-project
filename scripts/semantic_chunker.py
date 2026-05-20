"""
Semantic chunking of multilingual AI-human support conversations.

Splits conversations into 200-700-token chunks via sliding window + optional cosine
similarity boundary detection. Writes results to BigQuery (conversation_chunks table).

Note: chunk_level_translation.sql (future_work/) was not executed —
translated_chunk_text = original_chunk_text.

Pipeline:
  03_prepare_docs.sql → semantic_chunker.py → 04_chunk_embeddings.sql → chunk_clustering.py

Usage:
    python semantic_chunker.py
    python semantic_chunker.py --sem-threshold 0.60
    python semantic_chunker.py --skip-bq
    python semantic_chunker.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import re
import sys
from typing import Iterator

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from gcp_config import PROJECT_ID, bq_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SRC_MESSAGES  = bq_table("chatbot_messages_deidentified")
SRC_DOCS      = bq_table("conversation_docs")
DEST_CHUNKS   = bq_table("conversation_chunks")

MIN_TOKENS_DEFAULT  = 200
MAX_TOKENS_DEFAULT  = 700
OVERLAP_MESSAGES    = 1      # messages to carry over from previous chunk
SEM_THRESHOLD       = 0.55   # cosine sim below which a new chunk starts
MIN_MSG_TOKENS      = 8      # messages shorter than this are noise-attached

_BOILERPLATE_EXACT: frozenset[str] = frozenset({
    "hi", "hello", "hey", "yes", "no", "ok", "okay", "sure", "thanks",
    "bye", "goodbye", "please", "alright", "right", "great", "perfect",
    "awesome", "cool", "fine", "noted", "understood", "agree", "agreed",
    "welcome", "yep", "nope", "np", "ty", "thx", "yw", "k", "hmm",
    "oh", "ah", "well", "nice", "good", "wow",
    "sorry", "correct", "true", "wait", "idk",
    # Filler wait/hold phrases (AI buying time)
    "one moment", "please wait", "hold on", "stand by", "just a moment",
    "one sec", "one second", "brb",
    "i'm sorry", "i am sorry", "my apologies", "apologies",
    "that's right", "you're right", "you are right", "i agree", "sure thing",
    "not really", "i don't know", "i do not know",
    "thank you", "thank you!", "thanks!", "good morning", "good afternoon",
    "good evening", "got it", "i see", "sounds good", "no problem",
    "no worries", "my pleasure", "of course", "absolutely", "certainly",
    "will do", "sounds great", "works for me", "makes sense", "fair enough",
    "i know", "i understand", "take care", "good luck", "same to you",
    "no rush", "that works", "you are welcome", "you're welcome",
    "it was my pleasure", "thank you very much", "many thanks",
    "much appreciated", "have a great day", "have a nice day",
    "have a good day", "have a great week", "have a good week",
    "good to know", "talk soon", "see you", "best regards", "kind regards",
    "warm regards", "best wishes", "sincerely", "cheers", "regards",
    "ok thanks", "okay thanks", "ok thank you", "got it thanks",
    "thank you for your help", "thanks for your help", "thanks for helping",
    "you're welcome!", "you are welcome!",
})

_BOILERPLATE_PREFIXES: tuple[str, ...] = (
    "as an ai",
    "i am an ai",
    "i'm an ai",
    "i am a virtual assistant",
    "i'm a virtual assistant",
    "i am a chatbot",
    "i'm a chatbot",
    "please note that i am",
    "i do not have access to",
    "i cannot access your account",
    "i don't have access to",
    "is there anything else i can help",
    "let me know if you need anything else",
    "let me know if you have any",
    "feel free to reach out",
    "feel free to ask",
    "feel free to contact",
    "please don't hesitate",
    "please feel free",
    "please let me know",
    "hope that helps",
    "hope this helps",
    "have a great day",
    "have a nice day",
    "have a good day",
    "have a wonderful day",
    "have a great week",
    "if you have any other",
    "if you have any questions",
    "if you need anything else",
    "if there's anything else",
    "thank you for reaching out",
    "thank you for contacting",
    "thank you for your message",
    "thank you for your inquiry",
    "thank you for your patience",
    "thank you for using",
    "thank you for choosing",
    "thank you for your time",
    "i'm here to help",
    "i am here to help",
    "i'd be happy to",
    "i would be happy to",
    "i'll be happy to",
    "i will be happy to",
    "i'm happy to assist",
    "i am happy to assist",
    "of course, i",
    "certainly, i",
    "absolutely, i",
    # Apology fillers (short ones — "i apologize for the inconvenience" alone is noise)
    "i apologize for any inconvenience",
    "i'm sorry for any inconvenience",
    "i apologize for the inconvenience",
    "i'm sorry for the inconvenience",
    "i understand your frustration",
    "i understand how frustrating",
    "i can understand your",
    "i can see that you",
    "i see that you are",
    "one moment please",
    "just a moment please",
    "please wait while",
    "please hold on",
    "please give me a moment",
    # Generic offer-to-help openers (no topic content)
    "let me help you",
    "i'll help you with that",
    "i will help you with that",
    "i can help you with that",
    "i'd be glad to",
    "i would be glad to",
    "i'll be glad to",
    "i will be glad to",
    "happy to help",
    "glad to help",
    "great question",
    "good question",
    "that's a great question",
    "let me explain",
    "allow me to explain",
    "as i mentioned",
    "as mentioned earlier",
    "as stated above",
    "to summarize",
    "in summary",
    "to recap",
)


def _is_boilerplate(text: str) -> bool:
    """Return True if the (translated) message adds no topical signal."""
    t = text.strip().lower().rstrip("!.,?")
    if not t:
        return True
    if t in _BOILERPLATE_EXACT:
        return True
    return any(t.startswith(p) for p in _BOILERPLATE_PREFIXES)


try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _token_count(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))

except ImportError:
    log.warning(
        "tiktoken not installed — using whitespace word count as token proxy.\n"
        "  Install with: pip install tiktoken"
    )

    def _token_count(text: str) -> int:  # type: ignore[misc]
        return len(text.split())


_SIM_MODEL = None


def _get_sim_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading lightweight embedding model for semantic boundary detection…")
            _SIM_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            log.info("Semantic model ready.")
        except Exception as exc:
            log.warning(
                "Could not load SentenceTransformer (%s).\n"
                "  Falling back to window-only chunking (no semantic boundary detection).\n"
                "  Install with: pip install sentence-transformers",
                exc,
            )
            _SIM_MODEL = None
    return _SIM_MODEL


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 1.0
    return float(np.dot(a, b) / denom)


def _chunk_conversation(
    messages: list[dict],
    *,
    min_tokens: int,
    max_tokens: int,
    sem_threshold: float,
    use_semantic: bool,
) -> list[dict]:
    """Split one conversation's messages into semantic chunks. Returns a list of chunk dicts."""
    if not messages:
        return []

    model = _get_sim_model() if use_semantic else None

    # non-English: translated_text is a conversation-level blob, use original for token counting
    enriched: list[dict] = []
    for m in messages:
        orig = m.get("original_text") or ""
        is_conv_level = m.get("translation_granularity") == "conversation_level"
        if is_conv_level:
            # original_text is chunk-specific; conversation_text_translated is not
            trans = orig
        else:
            trans = m.get("translated_text") or orig
        if _is_boilerplate(trans):
            continue
        tc = _token_count(trans)
        if tc < 2:
            continue
        enriched.append({**m, "_trans": trans, "_orig": orig, "_tc": tc,
                         "_is_conv_level": is_conv_level})

    if not enriched:
        return []

    chunks: list[dict] = []
    current: list[dict] = []
    current_tokens = 0
    current_embedding: np.ndarray | None = None

    def _flush(overlap: int = OVERLAP_MESSAGES) -> list[dict]:
        """Save current window as a chunk and return overlap messages."""
        nonlocal current, current_tokens, current_embedding
        if not current:
            return []
        chunks.append(_build_chunk(current, len(chunks)))
        # Carry last `overlap` messages into next chunk
        tail = current[-overlap:] if overlap > 0 else []
        current = list(tail)
        current_tokens = sum(m["_tc"] for m in current)
        current_embedding = None
        return tail

    for msg in enriched:
        if use_semantic and model is not None and current and current_tokens >= min_tokens:
            try:
                chunk_text = " ".join(m["_trans"] for m in current)
                emb_chunk = model.encode(chunk_text, normalize_embeddings=True)
                emb_msg   = model.encode(msg["_trans"], normalize_embeddings=True)
                sim = _cosine_sim(emb_chunk, emb_msg)
                if sim < sem_threshold:
                    _flush()
            except Exception:
                pass  # if encoding fails, fall back to token-size splitting

        if current_tokens + msg["_tc"] > max_tokens and current_tokens >= min_tokens:
            _flush()

        current.append(msg)
        current_tokens += msg["_tc"]

    if current:
        _flush(overlap=0)  # last chunk — no overlap needed

    return chunks


def _build_chunk(messages: list[dict], chunk_seq: int) -> dict:
    """Assemble a chunk record from a window of messages."""
    conv_id = messages[0]["conversation_id"]
    orig_parts  = []
    trans_parts = []
    is_conv_level = any(m.get("_is_conv_level") for m in messages)
    for m in messages:
        role_tag = "[USER]" if (m.get("role") or "").lower() in ("client", "user") else "[AI]"
        orig_parts.append(f"{role_tag}: {m['_orig']}")
        trans_parts.append(f"{role_tag}: {m['_trans']}")

    orig_text  = "\n".join(orig_parts)
    trans_text = "\n".join(trans_parts)

    return {
        "conversation_id":        conv_id,
        "chunk_id":               f"{conv_id}__c{chunk_seq:04d}",
        "chunk_seq":              chunk_seq,
        "message_start_idx":      messages[0].get("message_id"),
        "message_end_idx":        messages[-1].get("message_id"),
        "num_messages":           len(messages),
        "chunk_char_count":       len(orig_text),   # used by 04_chunk_embeddings.sql
        "original_chunk_text":    orig_text,
        # chunk_level_translation.sql not run — translated == original for non-English chunks
        "translated_chunk_text":  trans_text,
        # message_level → English (chunk-specific); conversation_level → non-English (translated == original)
        "translation_granularity": "conversation_level" if is_conv_level else "message_level",
        "languages":              ",".join(sorted({m.get("language", "en") for m in messages})),
        "min_message_date":       min(
            m["message_date"] for m in messages if m.get("message_date") is not None
        ),
    }


def _pull_messages(client, limit: int | None) -> pd.DataFrame:
    """Pull messages from BigQuery joined with translations from conversation_docs."""
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT
        m.conversation_id,
        m.message_date,
        m.agent_or_bot                               AS role,
        m.content                                    AS original_text,
        -- Use per-message original for English, full-conv translation for others
        CASE
            WHEN COALESCE(d.detected_language, 'en') = 'en'
                THEN m.content
            ELSE d.conversation_text_translated
        END                                          AS translated_text,
        COALESCE(d.detected_language, 'en')          AS language,
        CASE
            WHEN COALESCE(d.detected_language, 'en') = 'en'
                THEN 'message_level'
            ELSE 'conversation_level'
        END                                          AS translation_granularity
    FROM `{SRC_MESSAGES}` m
    LEFT JOIN `{SRC_DOCS}` d USING (conversation_id)
    WHERE m.content IS NOT NULL
      AND TRIM(m.content) != ''
    ORDER BY m.conversation_id, m.message_date
    {limit_clause}
    """
    log.info("Pulling messages from BigQuery…")
    df = client.query(query).to_dataframe(progress_bar_type="tqdm")
    # synthetic positional message_id
    df["message_id"] = df.groupby("conversation_id").cumcount()
    log.info("  %d messages across %d conversations.", len(df), df["conversation_id"].nunique())
    return df


def _write_chunks(client, chunk_df: pd.DataFrame, dest: str) -> None:
    from google.cloud import bigquery as bq
    job_config = bq.LoadJobConfig(
        write_disposition=bq.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    log.info("Writing %d chunks → %s…", len(chunk_df), dest)
    job = client.load_table_from_dataframe(chunk_df, dest, job_config=job_config)
    job.result()
    log.info("Done.")


def run(args: argparse.Namespace) -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)
    df = _pull_messages(client, limit=args.limit)

    all_chunks: list[dict] = []
    grouped = df.groupby("conversation_id", sort=False)
    n_convs = len(grouped)
    log.info("Chunking %d conversations…", n_convs)

    for i, (conv_id, grp) in enumerate(grouped):
        messages = grp.to_dict("records")
        chunks = _chunk_conversation(
            messages,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            sem_threshold=args.sem_threshold,
            use_semantic=not args.no_semantic_split,
        )
        all_chunks.extend(chunks)
        if (i + 1) % 1000 == 0:
            log.info("  %d / %d conversations processed (%d chunks so far)…",
                     i + 1, n_convs, len(all_chunks))

    chunk_df = pd.DataFrame(all_chunks)
    if chunk_df.empty:
        log.warning("No chunks produced — check that source tables are populated.")
        return chunk_df

    log.info(
        "Chunking complete. %d chunks from %d conversations. "
        "Avg %.1f chunks/conv. Avg %.1f messages/chunk.",
        len(chunk_df),
        chunk_df["conversation_id"].nunique(),
        len(chunk_df) / max(chunk_df["conversation_id"].nunique(), 1),
        chunk_df["num_messages"].mean(),
    )

    OUT_DIR = REPO_ROOT / "data"
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "conversation_chunks.csv"
    chunk_df.to_csv(csv_path, index=False)
    log.info("Local copy → %s", csv_path)

    if args.dry_run:
        log.info("--dry-run: skipping BigQuery write.")
        print(chunk_df.head(5).to_string())
        return chunk_df

    if not args.skip_bq:
        _write_chunks(client, chunk_df, DEST_CHUNKS)
    else:
        log.info("--skip-bq: BigQuery write skipped.")

    return chunk_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic chunking of support conversations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--min-tokens",  type=int,   default=MIN_TOKENS_DEFAULT,
                        help="Minimum translated-token count per chunk.")
    parser.add_argument("--max-tokens",  type=int,   default=MAX_TOKENS_DEFAULT,
                        help="Maximum translated-token count per chunk (hard cap).")
    parser.add_argument("--sem-threshold", type=float, default=SEM_THRESHOLD,
                        help="Cosine similarity below which a semantic boundary is inserted.")
    parser.add_argument("--no-semantic-split", action="store_true",
                        help="Disable cosine-similarity boundary detection; use window only.")
    parser.add_argument("--skip-bq",   action="store_true",
                        help="Do not write to BigQuery; save CSV only.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Compute chunks but do not write anything.")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Pull at most N messages from BigQuery (dev mode).")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
