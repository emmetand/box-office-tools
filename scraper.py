"""Build a table of movie data from the TMDB API for films released in US theaters in 2021 or later.

How it works:
  1. Query /discover/movie one month at a time for US theatrical releases (limited or wide)
     from the start year through today. Monthly chunks stay under TMDB's 500-page limit.
  2. Fetch /movie/{id} (with release dates appended) for each film.
  3. Keep films whose *earliest* US theatrical date is on or after Jan 1 of the start year.
     That drops re-releases of older films, which TMDB stores as extra dates on the original.

Requires TMDB_TOKEN (API Read Access Token) or TMDB_API_KEY in a .env file next to this script.

Usage:
  python scraper.py                           # writes box_office_2021_plus.csv
  python scraper.py --only-with-revenue -o movies.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.themoviedb.org/3"
# TMDB release types: 1 premiere, 2 limited theatrical, 3 theatrical, 4 digital, 5 physical, 6 TV.
THEATRICAL_TYPES = {2, 3}
COLUMNS = [
    "Movie", "Release Date", "Production Companies", "Opening", "Budget", "Rating",
    "Genres", "Domestic", "International", "Worldwide", "TMDB ID",
]


class TMDB:
    """TMDB client with retries (including 429 rate limits) and an on-disk JSON cache."""

    def __init__(self, token: str | None, api_key: str | None, cache_dir: Path, max_age_hours: float):
        self.token = token
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.max_age = max_age_hours * 3600
        self._local = threading.local()
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            if self.token:
                session.headers["Authorization"] = f"Bearer {self.token}"
            retry = Retry(
                total=6, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                respect_retry_after_header=True,
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
            self._local.session = session
        return session

    def get(self, path: str, use_cache: bool = True, **params) -> dict | None:
        key = path + "?" + json.dumps(params, sort_keys=True)
        cache_file = self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")
        if use_cache and cache_file.exists() and time.time() - cache_file.stat().st_mtime < self.max_age:
            return json.loads(cache_file.read_text(encoding="utf-8"))

        if not self.token:
            params["api_key"] = self.api_key
        resp = self._session().get(API_URL + path, params=params, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if use_cache:
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data


def month_ranges(start: date, end: date):
    current = start
    while current <= end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current, min(next_month - timedelta(days=1), end)
        current = next_month


def discover_movie_ids(tmdb: TMDB, start_year: int) -> list[int]:
    ids: dict[int, None] = {}  # ordered set
    for first, last in month_ranges(date(start_year, 1, 1), date.today()):
        page, total_pages = 1, 1
        while page <= total_pages:
            data = tmdb.get(
                "/discover/movie",
                region="US",
                with_release_type="2|3",
                **{"release_date.gte": first.isoformat(), "release_date.lte": last.isoformat()},
                sort_by="primary_release_date.asc",
                include_adult="false",
                page=page,
            )
            total_pages = min(data["total_pages"], 500)
            for movie in data["results"]:
                ids.setdefault(movie["id"])
            page += 1
        print(f"  {first:%b %Y}: {len(ids)} films so far")
    return list(ids)


def us_theatrical_info(movie: dict) -> tuple[date | None, str | None]:
    """Earliest US theatrical date and the US rating (MPA certification)."""
    us = next(
        (c for c in movie.get("release_dates", {}).get("results", []) if c["iso_3166_1"] == "US"),
        None,
    )
    if not us:
        return None, None
    theatrical = sorted(
        (r for r in us["release_dates"] if r["type"] in THEATRICAL_TYPES),
        key=lambda r: r["release_date"],
    )
    first = date.fromisoformat(theatrical[0]["release_date"][:10]) if theatrical else None
    # Prefer the rating on the theatrical release; fall back to any US rating.
    rating = next((r["certification"] for r in theatrical if r["certification"]), None)
    if rating is None:
        rating = next((r["certification"] for r in us["release_dates"] if r["certification"]), None)
    return first, rating


def parse_movie(movie: dict) -> dict:
    release_date, rating = us_theatrical_info(movie)
    return {
        "Movie": movie["title"],
        "Release Date": release_date,
        "Production Companies": ", ".join(c["name"] for c in movie.get("production_companies", [])) or None,
        "Opening": None,        # not available from TMDB
        "Budget": movie.get("budget") or None,  # TMDB uses 0 for unknown
        "Rating": rating,
        "Genres": ", ".join(g["name"] for g in movie.get("genres", [])) or None,
        "Domestic": None,       # not available from TMDB
        "International": None,  # not available from TMDB
        "Worldwide": movie.get("revenue") or None,
        "TMDB ID": movie["id"],
        "_primary_release": movie.get("release_date") or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=2021, help="earliest US release year to keep (default: 2021)")
    parser.add_argument("-o", "--output", default="box_office_2021_plus.csv", help="output file (.csv or .xlsx)")
    parser.add_argument("--only-with-revenue", action="store_true", help="drop films with no worldwide revenue on TMDB")
    parser.add_argument("--workers", type=int, default=8, help="parallel requests (default: 8)")
    parser.add_argument("--cache-dir", default=".cache/tmdb", help="where API responses are cached")
    parser.add_argument("--cache-hours", type=float, default=24, help="re-download cached responses older than this")
    args = parser.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    token = os.getenv("TMDB_TOKEN", "").strip() or None
    api_key = os.getenv("TMDB_API_KEY", "").strip() or None
    if not token and not api_key:
        print("Set TMDB_TOKEN or TMDB_API_KEY in .env (see .env.example).", file=sys.stderr)
        return 1

    tmdb = TMDB(token, api_key, Path(args.cache_dir), args.cache_hours)
    try:
        tmdb.get("/authentication", use_cache=False)
    except requests.HTTPError as exc:
        print(f"TMDB rejected the credentials in .env: {exc}", file=sys.stderr)
        return 1

    print(f"Finding US theatrical releases since {args.start_year}...")
    movie_ids = discover_movie_ids(tmdb, args.start_year)

    print(f"Fetching details for {len(movie_ids)} films...")
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(tmdb.get, f"/movie/{mid}", append_to_response="release_dates"): mid
            for mid in movie_ids
        }
        for i, future in enumerate(as_completed(futures), start=1):
            mid = futures[future]
            try:
                movie = future.result()
                if movie:
                    rows.append(parse_movie(movie))
            except Exception as exc:  # keep going; report at the end
                failures.append((mid, exc))
            if i % 500 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}")

    start = date(args.start_year, 1, 1)
    df = pd.DataFrame(rows)
    df = df[df["Release Date"].notna() & (df["Release Date"] >= start)]
    # Guard against older films getting a first US release now (restorations, imports):
    # allow a year of slack for festival premieres before the US theatrical run.
    primary = pd.to_datetime(df["_primary_release"], errors="coerce").dt.date
    df = df[primary.isna() | (primary >= date(args.start_year - 1, 1, 1))]
    if args.only_with_revenue:
        df = df[df["Worldwide"].notna()]
    df = df.sort_values(["Release Date", "Movie"])[COLUMNS].reset_index(drop=True)

    money_cols = ["Opening", "Budget", "Domestic", "International", "Worldwide"]
    df[money_cols] = df[money_cols].astype("Int64")

    if args.output.lower().endswith(".xlsx"):
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"\nWrote {len(df)} films to {args.output}")
    print(f"  with budget: {df['Budget'].notna().sum()}, with worldwide revenue: {df['Worldwide'].notna().sum()}")
    if failures:
        print(f"{len(failures)} films failed to download (re-run to retry):", file=sys.stderr)
        for mid, exc in failures[:20]:
            print(f"  {mid}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
