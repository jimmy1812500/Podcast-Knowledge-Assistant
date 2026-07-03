# Phase 6 — Audio Timestamp Playback in the UI

## Background

Every chunk stored in ChromaDB already carries `timestamp_start` and `timestamp_end` (float seconds from the Whisper transcription). These timestamps are returned in every `/chat` and `/chat/stream` response as part of `sources`. However, the current `index.html` only displays them as numbers — there is no way to actually hear the spoken moment.

This phase wires source timestamps to an HTML5 `<audio>` player so users can click "▶ Play" on any source card and jump directly to the relevant passage in the episode audio.

**Scope:** new `GET /audio/{filename}` endpoint (in `backend/routers/`), modifications to `backend/static/index.html`. Prerequisite: MP3 files must be kept on disk after ingestion.

---

## Tasks

### 1. Verify MP3 retention in `backend/etl/podcast_rss.py`
- [ ] Read `podcast_rss.py` and check whether downloaded MP3 files are deleted after transcription
- [ ] If they are deleted, change the pipeline to retain them in `./audio_cache/` instead
- [ ] Add `audio_cache/` to `.gitignore` if not already present

### 2. Create a new router `backend/routers/audio.py`
- [ ] Define `AUDIO_DIR` as the path where MP3s are stored (e.g. `Path("audio_cache")`)
- [ ] Implement `GET /audio/{filename}`:
  - Resolve `AUDIO_DIR / filename`
  - Return HTTP 404 if the file does not exist
  - Return `FileResponse(path, media_type="audio/mpeg")` if it does
- [ ] Register the router in `backend/main.py`: `app.include_router(audio.router)`

### 3. Modify `backend/static/index.html`

#### 3a. Add a persistent audio player
- [ ] Add `<audio id="player" controls style="width:100%"></audio>` in a fixed/sticky bar at the bottom of the page
- [ ] The player should be hidden (`display:none`) until the first "▶ Play" is clicked

#### 3b. Add "▶ Play" button to each source card
- [ ] In the JavaScript that renders source cards, add a `<button>▶ Play</button>` element per card
- [ ] On click, set `audio.src` to `/audio/<encoded filename>` and `audio.currentTime` to `source.timestamp_start`, then call `audio.play()`
- [ ] Format and display `timestamp_start` as `mm:ss` in the card UI for readability

---

## Acceptance Criteria

- [ ] `GET /audio/{filename}` returns the MP3 file with `Content-Type: audio/mpeg`
- [ ] `GET /audio/nonexistent.mp3` returns HTTP 404
- [ ] The route is registered in `main.py` and appears in the FastAPI `/docs` page
- [ ] MP3 files are retained in `./audio_cache/` after ingestion completes (not deleted)
- [ ] `audio_cache/` is listed in `.gitignore`
- [ ] Each source card in the UI shows a "▶ Play" button and a `mm:ss` timestamp label
- [ ] Clicking "▶ Play" on a source card causes the audio player to appear, load the correct episode, and begin playback from `timestamp_start` (±2 seconds tolerance)
- [ ] The audio player persists on screen between chat messages — it does not reset when a new answer arrives
- [ ] No JavaScript errors appear in the browser console during playback

---

## Notes

- The `filename` passed to `/audio/{filename}` should match the `source_file` value stored in ChromaDB metadata — verify these are the actual MP3 filenames on disk, not episode titles.
- If `source_file` stores an episode title (not a filename), add a `audio_file` metadata field during ingestion that records the actual on-disk filename.
- For episodes not yet downloaded, consider showing the "▶ Play" button as disabled with a tooltip "Audio not cached."
- MP3 files are ~100 MB each — do not serve them from a memory buffer; `FileResponse` streams them from disk correctly.
