"""
Podcast registry — single source of truth for supported podcast IDs and RSS URLs.

Adding a new podcast is a one-liner: add an entry to PODCASTS.
No other file needs to change for ingestion; only routers that expose the
podcast list need to know about this module.
"""

from __future__ import annotations

PODCASTS: dict[str, dict] = {
    "huberman": {
        "name": "Huberman Lab",
        "rss": "https://feeds.megaphone.fm/hubermanlab",
    },
    "lex_fridman": {
        "name": "Lex Fridman Podcast",
        "rss": "https://lexfridman.com/feed/podcast/",
    },
}


def get_podcast(podcast_id: str) -> dict:
    """Return the registry entry for *podcast_id*.

    Raises KeyError with a descriptive message if the ID is not registered.
    """
    if podcast_id not in PODCASTS:
        valid = list(PODCASTS)
        raise KeyError(f"Unknown podcast_id {podcast_id!r}. Valid IDs: {valid}")
    return PODCASTS[podcast_id]
