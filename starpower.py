"""Add a "Starpower Rating" (A / B / C) column to a movie CSV.

Source: The Numbers' "Top Grossing Leading Stars" ranking - actors ranked by
lifetime worldwide box office of films they led:
https://www.the-numbers.com/box-office-star-records/worldwide/lifetime-acting/top-grossing-leading-stars

An actor's tier comes from their rank on that list:

    A = rank 1-100, B = rank 101-500, C = everyone else (including unranked)

A movie's rating is the tier of its highest-ranked top-billed lead. Casts come
from TMDB's credits for the film, and are matched to The Numbers' list by
name (ignoring accents, punctuation and case).

Overrides: star_overrides.csv (columns name,tier,note) is the final word.
Any actor listed there gets that tier regardless of their ranking. Names are
matched the same forgiving way, and overridden tiers are marked with * in
the progress output.

The ranking is saved to a snapshot CSV with the date it was scraped, so reruns
are reproducible. Pass --refresh to scrape a new snapshot.

Setup: put your TMDB key in a .env file next to this script:

    TMDB_API_KEY=your_key_here

Either the v3 "API Key" or the v4 "API Read Access Token" works.

Usage:
    python starpower.py movies_sample.csv
    python starpower.py box_office_filtered.csv -o box_office_starpower.csv
"""

import argparse
import datetime
import io
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

RATING_COLUMN = "Starpower Rating"
TMDB_URL = "https://api.themoviedb.org/3"
NUMBERS_URL = "https://www.the-numbers.com/box-office-star-records/worldwide/lifetime-acting/top-grossing-leading-stars"
USER_AGENT = "box-office-tools/1.0 (personal research script)"
TIER_ORDER = {"A": 0, "B": 1, "C": 2}

TITLE_COLUMNS = ["title", "movie", "film", "name"]
DATE_COLUMNS = ["release_date", "release date", "date", "year"]
TMDB_ID_COLUMNS = ["tmdb id", "tmdb_id", "tmdbid"]

HERE = Path(__file__).resolve().parent


def load_dotenv(path):
    """Read KEY=value lines from a .env file into os.environ (existing values win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class TMDB:
    def __init__(self, key):
        self.session = requests.Session()
        # v4 read access tokens are JWTs (start with "eyJ"); v3 keys go in the query string.
        if key.startswith("eyJ"):
            self.session.headers["Authorization"] = f"Bearer {key}"
            self.params = {}
        else:
            self.params = {"api_key": key}

    def get(self, path, **params):
        response = self.session.get(f"{TMDB_URL}{path}", params={**self.params, **params}, timeout=30)
        if response.status_code == 401:
            sys.exit("TMDB rejected the API key. Check TMDB_API_KEY in .env.")
        response.raise_for_status()
        return response.json()

    def find_movie(self, title, year):
        params = {"query": title}
        if year:
            params["primary_release_year"] = year
        results = self.get("/search/movie", **params)["results"]
        if not results and year:  # release year can be off by one between sources
            results = self.get("/search/movie", query=title)["results"]
        if not results:
            return None
        exact = [r for r in results if r["title"].casefold() == title.casefold()]
        return (exact or results)[0]["id"]

    def leads(self, movie_id, max_leads):
        cast = self.get(f"/movie/{movie_id}/credits")["cast"]
        return sorted(cast, key=lambda c: c["order"])[:max_leads]


def normalize_name(name):
    """'Robert Downey, Jr.' and 'Robert Downey Jr.' -> 'robert downey jr'; 'Saldaña' -> 'saldana'."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", name.casefold()).split())


def scrape_numbers_ranking(size):
    """Scrape The Numbers' leading-stars ranking, 100 rows per page, until `size` rows."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    pages = []
    for start in range(1, size + 1, 100):
        url = NUMBERS_URL if start == 1 else f"{NUMBERS_URL}/{start}"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        pages.append(pd.read_html(io.StringIO(response.text))[0])
        time.sleep(1)  # be polite between pages
    table = pd.concat(pages).head(size)
    return pd.DataFrame({
        "rank": table["Rank"].astype(int),
        "name": table["Name"],
        "worldwide_box_office": table["Worldwide Box Office"],
        "movies": table["Movies"],
        "scraped": datetime.date.today().isoformat(),
    })


def load_ranking(path, size, refresh):
    """Return {normalized name: rank}, from the snapshot file unless it's missing or a refresh is asked for."""
    ranking = pd.read_csv(path) if path.exists() and not refresh else None
    if ranking is not None and len(ranking) >= size:
        print(f"Using ranking snapshot {path.name} (scraped {ranking['scraped'].iloc[0]})", file=sys.stderr)
    else:
        print(f"Scraping top {size} leading stars from The Numbers...", file=sys.stderr)
        ranking = scrape_numbers_ranking(size)
        ranking.to_csv(path, index=False)
    # If two stars normalize to the same name, keep the higher-ranked one.
    return {name: rank for name, rank in zip(ranking["name"].map(normalize_name)[::-1], ranking["rank"][::-1])}


def load_overrides(path):
    """Return {normalized name: tier} from the overrides CSV, or {} if the file doesn't exist."""
    if not path.exists():
        return {}
    overrides = pd.read_csv(path, dtype=str).dropna(subset=["name", "tier"])
    overrides["tier"] = overrides["tier"].str.strip().str.upper()
    bad = overrides[~overrides["tier"].isin(TIER_ORDER)]
    if len(bad):
        sys.exit(f"{path.name}: tier must be A, B or C. Fix: {bad[['name', 'tier']].values.tolist()}")
    return dict(zip(overrides["name"].map(normalize_name), overrides["tier"]))


def tier_for_rank(rank, a_cutoff, b_cutoff):
    if rank is not None and rank <= a_cutoff:
        return "A"
    if rank is not None and rank <= b_cutoff:
        return "B"
    return "C"


def find_column(df, candidates, override=None):
    if override:
        if override not in df.columns:
            sys.exit(f"Column {override!r} not found. Columns: {list(df.columns)}")
        return override
    lowered = {c.lower(): c for c in df.columns}
    return next((lowered[c] for c in candidates if c in lowered), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="input CSV")
    parser.add_argument("-o", "--output", help="output CSV (default: <input>_starpower.csv)")
    parser.add_argument("--a-cutoff", type=int, default=100, help="ranks 1..N are A-list")
    parser.add_argument("--b-cutoff", type=int, default=500, help="ranks above the A cutoff up to N are B-list")
    parser.add_argument("--max-leads", type=int, default=3, help="number of top-billed leads to consider")
    parser.add_argument("--ranking", default=str(HERE / "numbers_star_ranking.csv"), help="ranking snapshot file")
    parser.add_argument("--overrides", default=str(HERE / "star_overrides.csv"), help="actor tiers that override the ranking")
    parser.add_argument("--refresh", action="store_true", help="scrape a new ranking snapshot from The Numbers")
    parser.add_argument("--title-col", help="movie title column (auto-detected by default)")
    parser.add_argument("--date-col", help="release date/year column (auto-detected by default)")
    parser.add_argument("--tmdb-id-col", help="TMDB movie ID column (auto-detected; skips the title search)")
    parser.add_argument("--workers", type=int, default=8, help="parallel TMDB requests")
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        sys.exit(f"No TMDB key found. Add a line like this to {HERE / '.env'}:\n  TMDB_API_KEY=your_key_here")

    df = pd.read_csv(args.input)
    title_col = find_column(df, TITLE_COLUMNS, args.title_col)
    if not title_col:
        sys.exit(f"Couldn't find a title column; pass --title-col. Columns: {list(df.columns)}")
    date_col = find_column(df, DATE_COLUMNS, args.date_col)
    id_col = find_column(df, TMDB_ID_COLUMNS, args.tmdb_id_col)

    tmdb = TMDB(key)
    ranking = load_ranking(Path(args.ranking), args.b_cutoff, args.refresh)
    overrides = load_overrides(Path(args.overrides))
    if overrides:
        print(f"Using {len(overrides)} overrides from {Path(args.overrides).name}", file=sys.stderr)

    def actor_tier(name):
        """(tier, overridden?) - the overrides file wins over the ranking."""
        key = normalize_name(name)
        if key in overrides:
            return overrides[key], True
        return tier_for_rank(ranking.get(key), args.a_cutoff, args.b_cutoff), False

    def rate(row):
        title = row[title_col]
        try:
            movie_id = row[id_col] if id_col and pd.notna(row[id_col]) else None
            if movie_id is None:
                year = str(row[date_col])[:4] if date_col and pd.notna(row[date_col]) else None
                movie_id = tmdb.find_movie(title, year)
            if movie_id is None:
                print(f"  ! not found on TMDB: {title}", file=sys.stderr)
                return None
            leads = tmdb.leads(int(movie_id), args.max_leads)
        except requests.RequestException as e:
            print(f"  ! TMDB request failed for {title}: {e}", file=sys.stderr)
            return None
        tiers = [actor_tier(lead["name"]) for lead in leads]
        rating = min((t for t, _ in tiers), key=TIER_ORDER.get, default="C")
        detail = ", ".join(f"{l['name']} {t}{'*' if o else ''}" for l, (t, o) in zip(leads, tiers))
        print(f"  {title}: {rating}  ({detail})", file=sys.stderr)
        return rating

    print(f"Rating {len(df)} movies...", file=sys.stderr)
    with ThreadPoolExecutor(args.workers) as pool:
        df[RATING_COLUMN] = list(pool.map(rate, (row for _, row in df.iterrows())))

    output = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_starpower.csv"))
    df.to_csv(output, index=False)
    print(f"Wrote {output} ({df[RATING_COLUMN].notna().sum()} of {len(df)} rated).", file=sys.stderr)


if __name__ == "__main__":
    main()
