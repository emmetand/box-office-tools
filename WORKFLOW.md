# Box Office Tools — Workflow

The pipeline that turns TMDB data into a box office prediction model. There are four steps, and each one reads the previous step's output file.

```
scraper.py                  TMDB API  ──►  reference_docs/box_office_2021_plus.csv          (~23k films)
        │
        ▼
dataset_refiner.ipynb       filter + flag Netflix / anime / Chinese films
                            TMDB API, Wikipedia, Wikidata ──►  reference_docs/reference_tables/*.csv
                                                         ──►  reference_docs/box_office_refined.csv  (931 films)
        │
        ▼
starpower.py                The Numbers ranking + TMDB credits
                                                         ──►  reference_docs/box_office_refined_starpower.csv
        │
        ▼
ml_explore.ipynb            clean → features → train/test → linear regression vs gradient boosting
```

## Setup

- Python packages: `pandas`, `numpy`, `requests`, `python-dotenv`, `scikit-learn`, `scipy`, `matplotlib`, `lxml` (for `pd.read_html`)
- A `.env` file in the project root (gitignored) with your TMDB credentials:
  ```
  TMDB_TOKEN=eyJ...        # used by scraper.py and dataset_refiner.ipynb
  TMDB_API_KEY=...         # used by starpower.py (accepts either the v3 key or the v4 token)
  ```

## Step 1: Scrape TMDB (`scraper.py`)

```
python scraper.py -o reference_docs/box_office_2021_plus.csv
```

- Finds every US theatrical release (limited or wide) since 2021 using TMDB's `/discover/movie`, one month at a time.
- Fetches the details for each film and keeps those whose *earliest* US theatrical date is in range, which drops re-releases.
- Columns: `Movie, Release Date, Production Companies, Opening, Budget, Rating, Genres, Domestic, International, Worldwide, TMDB ID`. `Opening`, `Domestic` and `International` are always empty because TMDB doesn't provide them.
- Raw API responses are cached in `.cache/tmdb/` for 24 hours (`--cache-hours`).

## Step 2: Refine the dataset (`dataset_refiner.ipynb`)

Run all cells. Every threshold lives in the **Config** cell.

### 2a. Filter
This replaces the manual filtering that used to happen in `data_explore.ipynb`. It reproduces the original 931-film `box_office_filtered.csv` exactly:
- `Budget` > $1,000,000. TMDB has many placeholder budgets like $5.
- `Worldwide`, `Rating`, `Genres` and `Production Companies` must all be present.

### 2b. Reference tables (`reference_docs/reference_tables/`)
| File | Source | Grain | Contents |
|---|---|---|---|
| `tmdb_movie_facts.csv` | TMDB `/movie/{id}` | 1 row per film | original language, origin country, keywords, current US streaming providers |
| `tmdb_us_release_dates.csv` | TMDB `/movie/{id}` release_dates | 1 row per US release event | type (premiere / limited / wide / digital / physical / TV), date, note (e.g. "Netflix") |
| `wikipedia_netflix_films.csv` | Wikipedia "List of Netflix original films (year)" pages + "List of Netflix exclusive international distribution films", resolved to TMDB IDs via Wikidata (P4947) | 1 row per list entry | title, article, Netflix region, `tmdb_id`, `us_netflix` |

The tables are **snapshots**:
- The TMDB tables are filled in incrementally, so a rerun only fetches films that aren't in them yet. To rebuild them, set `REFRESH_REFERENCE_TABLES = True`.
- The Wikipedia table is reused until `REFRESH_WIKIPEDIA = True`. Wikipedia rate-limits hard (HTTP 429), so the client pauses about 1 second between calls and retries.

### 2c. Netflix flag (`Is Netflix`)
A film is flagged only when **both** deterministic sources agree:
1. **TMDB:** a US digital release whose note mentions Netflix within 60 days of the first US theatrical date (`NETFLIX_MAX_WINDOW_DAYS`).
2. **Wikipedia:** the film is on a Netflix originals list, or on the international-distribution list with a region that includes the US.

Each source alone gets things wrong:
- TMDB alone flags Indian theatrical hits (*RRR*, *Animal*, *Dhurandhar*), whose normal 4–8 week window ends on Netflix.
- Wikipedia alone flags late acquisitions (*Black Box*, released in theatres in 2022 and on Netflix in 2026).

Current result: **19 films**, including *Frankenstein*, *Wake Up Dead Man*, *Glass Onion*, *KPop Demon Hunters* and *Society of the Snow*. The notebook's review table lists every disagreement.

### 2d. Anime flag (`Is Anime`, plus `Anime` appended to `Genres`)
TMDB genre **Animation** AND (TMDB keyword `anime` OR original language Japanese OR origin country Japan). Current result: **9 films**.

### 2e. Chinese-film flag (`Is Chinese`)
Original language Mandarin/Cantonese (`zh` / `cn`) OR origin country China / Hong Kong / Taiwan. Current result: **9 films**, e.g. *Ne Zha 2*, *The Wandering Earth II* and *Dead to Rights*.

The refiner only **flags** films; it never drops them for these reasons. Whether a flagged group is used is decided by the toggles in `ml_explore.ipynb`.

### 2f. Overrides
`netflix_overrides.csv`, `anime_overrides.csv` and `chinese_overrides.csv` (columns `tmdb_id,movie,value,note`, where `value` is 1 or 0) always have the final word, like `star_overrides.csv`. All three are currently empty. The `Flag Source` column records whether each flag came from the rule or an override.

### 2g. Output: `reference_docs/box_office_refined.csv`
The scrape's columns plus `Is Netflix`, `Is Anime`, `Is Chinese`, `Original Language`, `US Wide Release` (1 = TMDB has a wide US theatrical release, 0 = limited only) and `Flag Source`.

## Step 3: Add star power (`starpower.py`)

```
python starpower.py reference_docs/box_office_refined.csv
```

- Writes `reference_docs/box_office_refined_starpower.csv` with a `Starpower Rating` column (A / B / C).
- An actor's tier comes from their rank on The Numbers' "Top Grossing Leading Stars" list: A = rank 1–100, B = 101–500, C = everyone else. Each film gets the best tier among its top 3 billed leads, using TMDB credits.
- The ranking is snapshotted to `numbers_star_ranking.csv`. Use `--refresh` to re-scrape it.
- `star_overrides.csv` overrides any actor's tier.

## Step 4: Model (`ml_explore.ipynb`)

Run all cells. Each section depends only on the sections above it.

1. **Clean** (each filter switchable in Config):
   - drop empty columns
   - drop flagged groups, one toggle each: `EXCLUDE_NETFLIX`, `EXCLUDE_ANIME` and `EXCLUDE_CHINESE` (all `True` by default). These groups are out of scope for US-focused predictions, so they're removed from both training and testing. Setting a toggle to `False` brings the group back and automatically adds its feature (`is_netflix`, `genre_Anime`, `is_chinese`).
   - drop grosses under $100k
   - drop films released in the last 8 weeks, which are still earning
2. **Time-based split:** the newest 20% of releases form the test set. The split happens *before* feature engineering.
3. **Features:** one `make_*_features` function per group:
   - log budget
   - star power dummies
   - rating dummies
   - multi-hot genres (rare genres go to `Other`; `Anime` is always kept when anime films are included)
   - season + COVID-era flag
   - major-studio flag
   - US wide-release flag
4. **Models:** median baseline, linear regression, **Huber regression** and gradient boosting, all defined in `MODEL_FACTORIES`. They're compared with time-series cross-validation, then scored on the held-out test set.
   - Huber is the same linear model with an outlier-resistant loss. Training films more than `HUBER_EPSILON` x the residual scale off the line get capped influence instead of squared influence. Nothing is deleted, and the test set is scored on every film as-is.
   - `HUBER_EPSILON = 2.5` caps the most extreme ~2.5% of training films. The textbook 1.35 capped 37% here and scored worse. Section 11 has an ε sweep.
5. **Inspection:**
   - OLS coefficient table (p-values, % effects)
   - Huber vs OLS coefficients, plus the list of films Huber down-weighted
   - permutation importance
   - biggest misses
6. **Decision report:** refit on all data, then describe films in `films_to_predict` and run `show_prediction_report()`. The budget can be exact or a `(low, high)` range. For each film the report gives:
   - the predicted gross (median outcome)
   - a ±1 SD likely range
   - the **chance of breaking even**: gross ≥ `BREAKEVEN_MULTIPLE` x budget, default 2.5x
   - a table of chances at 1x–5x budget
   - break-even odds across the budget range
   - a major-studio what-if

   The spread comes from the primary model's out-of-sample misses, measured separately for each segment (wide/limited x major studio or not). Major-studio wide releases miss by about 2.7x, independents by about 4.3x. A calibration table confirms that the stated probabilities match how often films actually broke even.

Current results (845 films after cleaning: 676 train, 169 test):

| Model | CV R² (log) | Test R² (log) | Test: within 2x of actual |
|---|---|---|---|
| Median baseline | -0.01 | -0.01 | 27% |
| Linear regression | 0.44 | 0.47 | 42% |
| Huber regression (ε = 2.5) | 0.44 | 0.47 | 44% |
| Gradient boosting | 0.53 | 0.43 | 40% |

## Refreshing everything

| What changed | Rerun |
|---|---|
| New releases / updated grosses | `scraper.py` → refiner → `starpower.py` → `ml_explore` |
| Refiner rule or threshold, or an override CSV | refiner → `starpower.py` → `ml_explore` |
| Star tiers or `star_overrides.csv` | `starpower.py` → `ml_explore` |
| Feature or model tweaks | `ml_explore` only |

## Known limitations

- **Films without a TMDB budget are dropped.** That removes many big anime and international releases (*Jujutsu Kaisen 0*, *The First Slam Dunk*, *One Piece Film Red*), so only 9 anime films survive.
- **Some TMDB grosses are wrong.** *Army of the Dead* shows $190M, though it is excluded as a Netflix release anyway.
- **Indian-language films (51 in the data) also earn mostly outside the US** and could be another candidate toggle. *Saiyaara* is the largest positive outlier left in training.
- **TMDB notes and Wikipedia lists are community-maintained.** A film missing from either one won't be flagged as Netflix. Check the review table in the refiner and use `netflix_overrides.csv`.
- **Wikipedia list entries without an article can't be resolved to a TMDB ID.** There are 629 such rows, mostly small direct-to-streaming titles that don't reach our dataset.

## Files

| Path | Role |
|---|---|
| `scraper.py` | Step 1 |
| `dataset_refiner.ipynb` | Step 2 |
| `starpower.py` | Step 3 |
| `ml_explore.ipynb` | Step 4 |
| `netflix_overrides.csv`, `anime_overrides.csv`, `chinese_overrides.csv`, `star_overrides.csv` | Hand-maintained overrides |
| `numbers_star_ranking.csv` | Star ranking snapshot |
| `reference_docs/reference_tables/` | Refiner reference-table snapshots |
| `data_explore.ipynb` | Scratch notebook for ad-hoc exploration (no longer part of the pipeline) |
| `reference_docs/box_office_filtered*.csv`, `box_office_data*.csv`, `box_office_with_stars.csv` | Legacy outputs from before the refiner; superseded by `box_office_refined*.csv` |
