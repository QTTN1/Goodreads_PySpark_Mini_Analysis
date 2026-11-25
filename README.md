# Goodreads Analysis Pipeline — IDS 706 Mini Assignment

A distributed PySpark pipeline over the **UCSD Goodreads Book Graph** dataset (~7.5 GB gzipped JSON → 6.9 GB Delta Lake, 23.9M rows), built on Databricks (Unity Catalog + serverless compute).

## Dataset

Source: [Goodreads Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (UCSD). Downloaded as gzipped JSON into a Unity Catalog volume and parsed into four Delta tables:

| Table | Rows | Size | Key columns |
| --- | --- | --- | --- |
| `workspace.goodreads.books` | 2,360,655 | 1.56 GB | title, author(s), average_rating, ratings_count, num_pages, publication_year/month/day, format, is_ebook, work_id |
| `workspace.goodreads.authors` | 829,529 | 0.01 GB | name, author_id, average_rating, ratings_count |
| `workspace.goodreads.book_genres` | 5,042,542 | 0.02 GB | book_id, genre (shelf-derived), shelf_count |
| `workspace.goodreads.reviews` | 15,739,967 | 5.31 GB | review_id, book_id, user_id, rating, review_text, n_votes, n_comments, date_added |

Total on-disk: **6.91 GB** (1 GB+ requirement met; verified in the ingestion notebook's final cell). Source numerics arrive as JSON strings and are TRY_CAST to double/int/bigint during ingestion.
 
 Note: the books table is one row per *edition*; `work_id` groups editions of the same title (2,360,655 editions → 1,521,963 distinct works, ~1.55 editions per work).

## Project structure

* `Goodreads Books Dataset Ingestion` notebook — downloads the 4 files to a UC volume, parses/flattens them, writes Delta tables.
* `Goodreads Analysis Pipeline (IDS 706 Mini Assignment)` notebook — this pipeline: 12 research questions, 2+ SQL queries, optimization analysis, caching demo, actions-vs-transformations demo, Parquet output.
* `README.md` — this file.
* Screenshots folder — Spark UI / query details (see below).

## Pipeline

**Transformations used**: `filter` (early, pre-join), `join` (books↔authors, books↔genres), `groupBy` + agg (`avg`, `count`, `stddev`, `collect_set`, `min/max`), `withColumn` (derived: review length, popularity buckets, series position via `regexp_extract`), window functions (`row_number` for per-genre peak years), and struct-ordering tricks (`min(struct(year, genre))` for earliest/latest genre per author).

**SQL**: two `spark.sql` queries over temp views — genre rankings, and highest-rated prolific authors.

**MLlib**: two linear regressions (rating and popularity targets) — full results in the MLlib section below.

**Output (Parquet)**: written to `/Volumes/workspace/goodreads/processed`:
* `genre_year_stats` — 1,202 rows, **partitioned by publication_year** (downstream year-range reads prune files)
* `author_stats` — 150,063 rows

## Performance analysis

* **`.explain()`** (see notebook + screenshot): the genre-year aggregation plan shows filters pushed into the Delta scans (`RequiredDataFilters` / `DictionaryFilters` — `ratings_count >= 50`, year range applied at the scan), a **broadcast hash join** against the small genres table (no shuffle), and two-stage partial → final hash aggregation. Photon accelerates the whole plan.
* **Early filtering**: all `ratings_count >= 50` and year-range filters are applied before joins/aggregations, shrinking shuffled data.
* **Caching**: the reused genre aggregation ran 1.10s cold, 0.48s cached — **2.3x speedup**.
* **Partitioned Parquet**: `partitionBy(publication_year)` writes one directory per year; a single-year read touches only that partition's files.
* **Actions vs transformations**: building 3 transformations + `groupBy` took 0.0006s (no Spark job); the first action (`count`) took 1.10s to scan 15.7M reviews — demonstrating lazy evaluation.

## MLlib

Two linear regressions benchmarked against a predict-the-mean baseline (80/20 splits, MLlib `VectorAssembler` + `LinearRegression`):

**Part 5a — predict a book's average rating** from log-popularity, log-review-count, page count, and book age (413,752 train / 103,546 test editions):

| Model | RMSE | Baseline RMSE (predict the mean 3.877) | R² |
| --- | --- | --- | --- |
| Linear regression | 0.3201 | 0.3258 | 0.035 |

Ratings are nearly unlearnable from book metadata — the model explains ~3.5% of variance, consistent with ratings clustering in the narrow 3.8-3.9 band. The largest coefficient is log-popularity (+0.063 per tenfold ratings increase).

**Part 5b — predict popularity** (`log10(ratings_count)`) from the author's track record — prior popularity and prior average rating of their *other* works, computed leave-one-out to avoid leakage — plus page count and book age (246,379 train / 61,334 test works):

| Model | RMSE | Baseline RMSE (predict the mean 2.477) | R² |
| --- | --- | --- | --- |
| Linear regression | 0.5133 | 0.6305 | 0.337 |

Popularity is predictable, and the author's track record dominates (prior-author-popularity coefficient +0.63): an author's past audience is the best single predictor of their next book's audience.

**Stability check**: rerunning both models over 10 re-randomized 80/20 splits confirms both results — 5a R² stays in 0.034-0.035, 5b in 0.331-0.338 (RMSE 0.5160 ± 0.0017 vs baseline 0.6323). The ~10x gap between targets holds across every seed.

## Key findings

Highlights from the pipeline's 12 questions — one headline result per analysis type (full results in the notebook):

1. **Attention is winner-take-all** (window functions + cumulative share): across 2.35M rated books and 959M ratings, the top 0.1% of books capture **42.3% of all ratings**, the top 1% capture 68.8%, and the bottom half of books share under 1%. Champions: The Hunger Games (4.9M ratings), Harry Potter #1 (4.77M).
2. **Review length vs rating** (withColumn + correlation): no correlation (r = 0.007). Length matters for *engagement* instead: the median review gets zero votes, but 3000+ char reviews average 5.3 votes vs 0.4 for <100 chars (13x).
3. **Series trajectory** (regexp_extract position parsing + windows): the first book is always the weakest in terms of popularity — first/middle/last = 3.89 / 4.03 / 4.05 — and the first→last gap grows with series length (+0.16 for trilogies, +0.19 for 5-book series), flattening only in 10+ book series.
4. **Genre combinations over time** (self-join pairs + decade windows): the most popular pairing is history/historical-fiction + non-fiction (91.0K authors); its formation peaked in the 2000s (35.1K new authors) before fantasy/paranormal + fiction took over the 2010s (33.6K) — genre-fiction crossovers displaced the literary pairing.
5. **Prolific authors** (join + distinct counts): 42,045 authors have 3+ distinct well-rated works (avg 3.88) — counted by `work_id`, since the books table is one row per *edition* (Stephen King: 1,873 editions but 244 works).
6. **Digital formats** (trend + share analysis): digital publications rose from ~0.5% (1990) to ~35% (2013); among multi-format works, 73.7% of 2010s releases have an ebook edition vs 38.2% for 2000s — digital availability grows with newer releases.
7. **Publication booms** (window peak ranking): 2013 is the biggest year (163,752 books) and every genre peaked in 2012-2015 — a platform-wide boom, not genre-specific waves.

Also explored in the notebook: page counts by format/genre/decade, review-vote distributions and top-upvoted reviews, genre publication counts/rating variability, publication months, and popularity-vs-rating buckets.

## Reproducing

1. Run the ingestion notebook (downloads ~7.5 GB, writes 4 Delta tables — needs ~15 min on serverless).
2. Run the analysis notebook top to bottom (~5 min on serverless).