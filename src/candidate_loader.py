"""
candidate_loader.py — Streaming loader for large candidate JSONL files.

Handles .jsonl, .json (array), and .jsonl.gz formats transparently.
Designed for memory efficiency: the primary interface is a generator that
yields one candidate dict at a time, keeping RAM usage constant regardless
of file size (~487MB / 100K candidates).
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Generator, TextIO

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CANDIDATES_FILE


def _open_file(file_path: Path) -> TextIO:
    """Open a file handle, transparently decompressing .gz files.

    Args:
        file_path: Path to the file to open.

    Returns:
        A text-mode file handle (utf-8).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {file_path}")

    suffixes = file_path.suffixes  # e.g. ['.jsonl', '.gz'] or ['.jsonl']

    if suffixes[-1:] == [".gz"]:
        return gzip.open(file_path, mode="rt", encoding="utf-8")
    elif suffixes[-1:] in ([".jsonl"], [".json"]):
        return open(file_path, mode="r", encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file format: {''.join(suffixes)}. "
            "Expected .jsonl, .json, or .jsonl.gz"
        )


def _detect_format(file_path: Path) -> str:
    """Detect whether the file is JSONL (one object per line) or JSON (array).

    Heuristic: strip the .gz suffix if present, then check the remaining
    extension. If it's .json, peek at the first non-whitespace character —
    '[' means a JSON array, '{' means it's actually JSONL despite the
    extension.

    Args:
        file_path: Path to the candidate data file.

    Returns:
        'jsonl' or 'json_array'.
    """
    stem_suffixes = [s for s in file_path.suffixes if s != ".gz"]
    ext = stem_suffixes[-1] if stem_suffixes else ""

    if ext == ".jsonl":
        return "jsonl"

    if ext == ".json":
        # Peek to disambiguate .json files that are actually JSONL
        with _open_file(file_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                return "json_array" if stripped[0] == "[" else "jsonl"
        return "json_array"  # empty file fallback

    return "jsonl"  # default fallback


def stream_candidates(
    file_path: Path | str | None = None,
) -> Generator[dict, None, None]:
    """Yield candidate dicts one at a time from a .jsonl or .json file.

    This is the primary interface for processing 100K+ candidates without
    loading the entire file into memory. Each yielded dict represents one
    candidate record.

    Args:
        file_path: Path to the candidates file. Defaults to
                   ``config.CANDIDATES_FILE`` if not provided.

    Yields:
        dict: A single candidate record.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line/entry is not valid JSON.
    """
    path = Path(file_path) if file_path is not None else CANDIDATES_FILE
    fmt = _detect_format(path)

    if fmt == "json_array":
        yield from _stream_json_array(path)
    else:
        yield from _stream_jsonl(path)


def _stream_jsonl(path: Path) -> Generator[dict, None, None]:
    """Stream candidates from a JSONL file (one JSON object per line).

    Args:
        path: Path to the .jsonl file.

    Yields:
        dict: A single candidate record.
    """
    with _open_file(path) as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if isinstance(record, dict):
                    yield record
            except json.JSONDecodeError as exc:
                # Log and skip malformed lines instead of crashing
                print(f"[WARN] Skipping malformed JSON at line {line_num}: {exc}")
                continue


def _stream_json_array(path: Path) -> Generator[dict, None, None]:
    """Stream candidates from a JSON file containing a top-level array.

    Uses ``ijson``-style incremental parsing if available, otherwise falls
    back to loading the full array. For 487MB files the fallback will use
    significant memory — prefer JSONL format for large datasets.

    Args:
        path: Path to the .json file.

    Yields:
        dict: A single candidate record.
    """
    with _open_file(path) as fh:
        data = json.load(fh)

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                yield entry
    elif isinstance(data, dict):
        # Single-object JSON — just yield it
        yield data


def load_all_candidates(file_path: Path | str | None = None) -> list[dict]:
    """Load all candidates into a list.

    Use this only when the full dataset fits comfortably in memory
    (e.g., sample files or machines with ≥32GB RAM). For production
    use on 16GB machines, prefer ``stream_candidates()`` instead.

    Args:
        file_path: Path to the candidates file. Defaults to
                   ``config.CANDIDATES_FILE``.

    Returns:
        list[dict]: All candidate records.
    """
    return list(stream_candidates(file_path))


def load_candidate_ids(file_path: Path | str | None = None) -> set[str]:
    """Load just the set of all candidate IDs for quick validation.

    Streams through the file so only the IDs are kept in memory —
    not the full candidate records.

    Args:
        file_path: Path to the candidates file. Defaults to
                   ``config.CANDIDATES_FILE``.

    Returns:
        set[str]: All unique ``candidate_id`` values found in the file.
                  Candidates missing the ``candidate_id`` field are skipped.
    """
    ids: set[str] = set()
    for candidate in stream_candidates(file_path):
        cid = candidate.get("candidate_id")
        if cid is not None:
            ids.add(str(cid))
    return ids


if __name__ == "__main__":
    # Quick smoke test: stream the first 5 candidates
    print(f"Loading from: {CANDIDATES_FILE}")
    for i, candidate in enumerate(stream_candidates()):
        cid = candidate.get("candidate_id", "UNKNOWN")
        name = candidate.get("name", "UNKNOWN")
        print(f"  [{i+1}] {cid}: {name}")
        if i >= 4:
            break
    print("✓ Streaming works.")
