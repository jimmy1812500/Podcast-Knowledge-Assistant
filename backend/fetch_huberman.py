"""
Standalone script — fetch & display recent Huberman Lab episodes.

Run:
    python fetch_huberman.py                   # list 5 latest episodes
    python fetch_huberman.py --filter sleep    # filter by keyword
    python fetch_huberman.py --download --n 2  # download 2 episodes to /tmp/podcasts
"""

import argparse
import asyncio
from pathlib import Path

from etl.podcast_registry import get_podcast
from etl.podcast_rss import download_episodes, fetch_episodes


async def main(n: int, title_filter: str | None, do_download: bool) -> None:
    rss_url = get_podcast("huberman")["rss"]
    print(f"Fetching RSS: {rss_url}\n")

    episodes = await fetch_episodes(
        rss_url,
        max_episodes=n,
        title_filter=title_filter,
    )

    if not episodes:
        print("No episodes matched.")
        return

    for i, ep in enumerate(episodes, 1):
        print(f"[{i}] {ep.title}")
        print(f"     Published : {ep.published.strftime('%Y-%m-%d')}")
        print(f"     Duration  : {ep.duration}")
        print(f"     ID        : {ep.episode_id}")
        print(f"     URL       : {ep.audio_url}")
        print()

    if do_download:
        dest = Path("/tmp/podcasts")
        print(f"Downloading {len(episodes)} episode(s) → {dest} …")
        paths = await download_episodes(episodes, dest_dir=dest)
        for path in paths:
            size_mb = path.stat().st_size / 1_000_000
            print(f"  ✓  {path.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huberman Lab RSS fetcher")
    parser.add_argument("--n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--filter", dest="title_filter", default=None, help="Title keyword filter")
    parser.add_argument("--download", action="store_true", help="Actually download MP3 files")
    args = parser.parse_args()

    asyncio.run(main(args.n, args.title_filter, args.download))
