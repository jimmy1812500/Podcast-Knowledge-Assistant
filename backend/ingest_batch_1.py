"""
Batch ETL ingestion for Huberman Lab 2025-2026 episodes.

Fetches every episode published between --since and --until, downloads the MP3,
runs Whisper ASR, splits transcript, embeds with sentence-transformers, and
upserts into ChromaDB.

Resume-safe: episodes already in ChromaDB are skipped automatically.

Usage:
    python ingest_batch.py                             # 2025-01-01 → today
    python ingest_batch.py --since 2025-06-01          # from a specific date
    python ingest_batch.py --until 2025-12-31          # cap at end of 2025
    python ingest_batch.py --model small               # use a larger Whisper model
    python ingest_batch.py --dry-run                   # list episodes without processing

Estimated time (base model, Apple Silicon M-series, CPU):
    ~5-10 min per hour of audio → 157 episodes × ~90 min avg ≈ 100-200 hrs total.
    Run overnight / across multiple sessions. Each session resumes from where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

# Allow `python backend/ingest_batch_1.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

# ── Validate env early ───────────────────────────────────────────────────────


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on torch availability."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _check_env(diarize: bool) -> None:
    if not diarize:
        print("  Diarize: OFF (pass --diarize to enable speaker labels, adds ~7 min/episode)")
    elif not os.environ.get("HF_TOKEN"):
        print("WARNING: --diarize requested but HF_TOKEN is not set — diarization will be skipped.")
        print("  Add HF_TOKEN to backend/.env\n")


# ── Lazy imports (after env check) ───────────────────────────────────────────


def _import_etl():
    from backend.etl.embeddings import COLLECTION_NAME, get_chroma
    from backend.etl.pipeline import run_etl
    from backend.etl.podcast_rss import HUBERMAN_LAB_RSS, download_episodes, fetch_episodes

    return fetch_episodes, download_episodes, run_etl, get_chroma, COLLECTION_NAME, HUBERMAN_LAB_RSS


# ── Already-indexed check ────────────────────────────────────────────────────


def _trim_audio(src: Path, fraction: float = 0.25) -> tuple[Path, float]:
    """Trim audio to the first `fraction` of its total duration using ffmpeg.
    Outputs WAV (PCM) to avoid MP3 frame-boundary sample-count mismatches in pyannote.
    Returns (trimmed_path, trimmed_duration_seconds)."""
    import av

    with av.open(str(src)) as container:
        total_secs = container.duration / 1_000_000  # microseconds → seconds
    trim_secs = total_secs * fraction
    dest = src.with_stem(src.stem + "_quarter").with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(src),
            "-t",
            str(trim_secs),
            "-ar",
            "16000",  # 16 kHz mono — what pyannote expects
            "-ac",
            "1",
            str(dest),
            "-y",
            "-loglevel",
            "error",
        ],
        check=True,
    )
    return dest, trim_secs


def _already_indexed(chroma, collection_name: str, title: str) -> bool:
    """Return True if any chunk with source_file == title exists in ChromaDB."""
    try:
        coll = chroma.get_or_create_collection(collection_name)
        result = coll.get(
            where={"source_file": {"$eq": title}},
            limit=1,
            include=[],
        )
        return len(result["ids"]) > 0
    except Exception:
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _log(msg: str) -> None:
    """Print a line above the tqdm bar without breaking it."""
    tqdm.write(msg)


class _Ticker:
    """Updates the tqdm bar postfix every 2 s with elapsed time — no new lines printed."""

    def __init__(self, label: str, pbar: tqdm):
        self._label = label
        self._pbar = pbar
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        start = time.monotonic()
        while not self._stop.wait(2):
            elapsed = time.monotonic() - start
            self._pbar.set_postfix_str(f"{self._label} {_fmt_duration(elapsed)}")

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


# ── Main pipeline ─────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    fetch_episodes, download_episodes, run_etl, get_chroma, COLLECTION_NAME, HUBERMAN_LAB_RSS = (
        _import_etl()
    )

    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    until = datetime.fromisoformat(args.until).replace(tzinfo=UTC) if args.until else None
    dest = Path(args.dest)
    device = _resolve_device(args.device)

    print("\nHuberman Lab batch ingest")
    print(f"  Range  : {since.date()} → {until.date() if until else 'today'}")
    print(f"  Model  : whisper-{args.model}")
    print(f"  Device : {device}")
    print(f"  Dest   : {dest}")
    if args.dry_run:
        print("  Mode   : DRY RUN (no download / ETL)")
    print("─" * 70)

    # ── Fetch episode list ───────────────────────────────────────────────────
    print("Fetching RSS feed …")
    episodes = await fetch_episodes(HUBERMAN_LAB_RSS, since=since)
    if until:
        episodes = [ep for ep in episodes if ep.published <= until]
    TARGET = "Essentials: The Science of Eating for Health, Fat Loss & Lean Muscle"
    episodes = [ep for ep in episodes if ep.title.startswith(TARGET[:50])]
    print(f"Found {len(episodes)} episodes in range.\n")

    if args.dry_run:
        for i, ep in enumerate(episodes, 1):
            print(f"  [{i:3d}] {ep.published.strftime('%Y-%m-%d')}  {ep.title[:65]}")
        print(f"\nTotal: {len(episodes)} episodes (dry run — nothing downloaded)")
        return

    # ── Connect to ChromaDB ──────────────────────────────────────────────────
    chroma = get_chroma()

    # ── Per-episode processing ───────────────────────────────────────────────
    total = len(episodes)
    done = skipped = failed = 0
    wall_start = time.monotonic()

    with tqdm(
        total=total,
        unit="ep",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n}/{total} [elapsed {elapsed}] {postfix}",
    ) as pbar:
        for ep in episodes:
            pbar.set_description(ep.title[:45])
            pbar.set_postfix(done=done, skip=skipped, fail=failed)

            _log(f"\n{'─' * 70}")
            _log(f"  {ep.published.strftime('%Y-%m-%d')}  {ep.title}")

            # ── Skip already-indexed ─────────────────────────────────────────
            if _already_indexed(chroma, COLLECTION_NAME, ep.title):
                _log("  ↳ SKIP — already in ChromaDB")
                skipped += 1
                pbar.set_postfix(done=done, skip=skipped, fail=failed)
                pbar.update(1)
                continue

            ep_start = time.monotonic()

            # ── Step 1: Download ─────────────────────────────────────────────
            pbar.set_description(f"[1/3 ↓] {ep.title[:38]}")
            try:
                paths = await download_episodes([ep], dest_dir=dest)
                path = paths[0]
                size_mb = path.stat().st_size / 1_000_000
                _log(f"  ↳ Download    {size_mb:.0f} MB ✓")
            except Exception as exc:
                _log(f"  ↳ Download    FAILED: {exc}")
                failed += 1
                pbar.set_postfix(done=done, skip=skipped, fail=failed)
                pbar.update(1)
                continue

            # ── Step 2 & 3: ETL (ASR + diarize + embed + store) ─────────────
            pbar.set_description(f"[2/3 ◎] {ep.title[:38]}")
            try:
                with _Ticker("ETL", pbar):
                    result = await run_etl(
                        path,
                        source_name=ep.title,
                        whisper_model=args.model,
                        chunk_size=500,
                        overlap=100,
                        device=device,
                        diarize=args.diarize,
                        user_id=args.user_id,
                    )
                pbar.set_description(f"[3/3 ✦] {ep.title[:38]}")
                elapsed = time.monotonic() - ep_start
                speakers = ", ".join(result["speakers_found"]) or "UNKNOWN"
                _log(f"  ↳ Transcribe  lang={result['language']}  segs={result['segments']} ✓")
                _log(f"  ↳ Embed       {result['chunks_stored']} chunks stored ✓")
                _log(f"  ↳ Speakers    {speakers}")
                _log(f"  ↳ Time        {_fmt_duration(elapsed)}")
                done += 1
            except Exception as exc:
                _log(f"  ↳ ETL         FAILED: {exc}")
                failed += 1
            finally:
                path.unlink(missing_ok=True)

            pbar.set_postfix(done=done, skip=skipped, fail=failed)
            pbar.update(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(
        f"Done in {_fmt_duration(time.monotonic() - wall_start)}\n"
        f"  Ingested : {done}\n"
        f"  Skipped  : {skipped}  (already in ChromaDB)\n"
        f"  Failed   : {failed}\n"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-ingest Huberman Lab podcast episodes into ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--since",
        default="2025-01-01",
        help="Ingest episodes published on or after this date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--until",
        default=None,
        help="Ingest episodes published on or before this date (YYYY-MM-DD). Default: today.",
    )
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size. Larger = more accurate but slower.",
    )
    p.add_argument(
        "--dest",
        default="/tmp/huberman_podcasts",
        help="Directory for temporary MP3 downloads (deleted after ETL).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print episode list without downloading or running ETL.",
    )
    p.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization via pyannote (requires HF_TOKEN). Adds ~7 min per episode on CPU.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for Whisper and pyannote. 'auto' picks CUDA if available, else CPU.",
    )
    p.add_argument(
        "--user-id",
        default="public",
        help="user_id tag written to every ChromaDB chunk. Must match the X-User-ID used when querying /chat.",
    )
    return p.parse_args()


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass

    _args = _parse_args()
    _check_env(_args.diarize)
    asyncio.run(run(_args))
