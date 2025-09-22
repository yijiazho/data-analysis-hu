#!/usr/bin/env python3
"""
Make denormalized CSV summaries from a TMDB-like movies CSV.

Inputs:
  - A CSV with headers like your sample (id,title,vote_average,...,production_countries,genres,...)

Outputs (written to --outdir):
  - movies_by_year.csv
  - movies_by_decade.csv
  - movies_by_rating_bin.csv
  - movies_by_revenue_bin.csv
  - movies_by_runtime_bin.csv
  - movies_by_region.csv
  - movies_by_genre.csv
  - movies_by_language.csv
  - movies_facts_denorm.csv   (per-movie skinny fact table w/ derived fields and bin labels)

Usage:
  python make_movie_denorms.py movies.csv --outdir output \
    --rating-bin-width 1.0 \
    --revenue-bins 0 1e5 1e6 1e7 1e8 1e9 inf \
    --runtime-bins 0 80 100 120 140 160 200 inf
"""
import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description="Create denormalized CSV rollups from a movies CSV.")
    p.add_argument("csv_path", default="sample.csv", help="Path to input movies CSV")
    p.add_argument("--outdir", default="denorm_out", help="Directory to write outputs")
    p.add_argument("--rating-bin-width", type=float, default=1.0,
                   help="Width of rating windows in points (e.g., 1.0 makes 7.00–7.99)")
    p.add_argument("--min-rating", type=float, default=0.0)
    p.add_argument("--max-rating", type=float, default=10.0)
    p.add_argument("--revenue-bins", type=float, nargs="+",
                   default=[0, 1e5, 1e6, 1e7, 1e8, 5e8, 1e9, 5e9, math.inf],
                   help="Edges for revenue bins (USD). Use 'inf' for open-ended upper bound.")
    p.add_argument("--runtime-bins", type=float, nargs="+",
                   default=[0, 80, 100, 120, 140, 160, 200, math.inf],
                   help="Edges for runtime bins (minutes). Use 'inf' for open-ended upper bound.")
    return p.parse_args()


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    try:
        return float(str(s).strip())
    except Exception:
        return np.nan


def _safe_int(s: Optional[str]) -> Optional[int]:
    f = _safe_float(s)
    if pd.isna(f):
        return np.nan
    try:
        return int(f)
    except Exception:
        return np.nan


def _split_list_field(s: Optional[str]) -> List[str]:
    """Split a comma-separated string field into list of trimmed items."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    # The CSV itself is quoted and commas inside fields are literal separators,
    # so here we can split on commas safely and strip whitespace.
    return [item.strip() for item in str(s).split(",") if item.strip()]


def _format_left_closed_interval(edges: List[float], decimals: int = 2) -> List[str]:
    """
    Build human-friendly labels for pd.cut left-closed, right-open bins [a, b).
    Last bin is [a, +∞) if top edge is inf.
    """
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        if math.isinf(b):
            labels.append(f"{a:,.{decimals}f}–∞")
        else:
            # For windows like 7.00–7.99, subtract a small epsilon for display
            upper_display = b - (10 ** -decimals) * 1.0
            labels.append(f"{a:,.{decimals}f}–{upper_display:,.{decimals}f}")
    return labels


def _format_left_closed_interval_int(edges: List[float]) -> List[str]:
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        if math.isinf(b):
            labels.append(f"{int(a)}–∞")
        else:
            labels.append(f"{int(a)}–{int(b-1)}")
    return labels

def to_int64_safe(s: pd.Series) -> pd.Series:
    # try to coerce everything numeric
    out = pd.to_numeric(s, errors="coerce")
    # scrub inf/-inf
    out = out.replace([np.inf, -np.inf], np.nan)
    # keep only true integers (no decimals)
    mask = out.notna() & (out == np.floor(out))
    out = out.where(mask)
    # final cast
    return out.astype("Int64")

def load_and_clean(csv_path: str) -> pd.DataFrame:
    # --- load CSV ---

    # Parse with Python engine so \" inside quoted fields is honored.
    df = pd.read_csv(
        csv_path,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",     # accept \" within quoted text
        doublequote=True,    # use standard CSV escaping with ""
        on_bad_lines="skip", # skip truly malformed lines instead of crashing
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",    # use 'utf-8-sig' if your file has a BOM
        # lineterminator=...  # <-- REMOVE THIS; not supported by python engine
    )

    # --- type coercions & derived cols ---
    for col in ["vote_average", "vote_count", "revenue", "runtime", "budget", "popularity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "id" in df.columns:
        before = len(df)
        df["id"] = to_int64_safe(df["id"])
        df = df.dropna(subset=["id"])  # drop bad IDs outright
        print(f"Dropped {before - len(df)} rows with invalid ids")


    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year

        # Filter out unrealistic release years (before cinema invention and too far in future)
        # Keep years from 1888 (motion pictures invented) to current year + 10
        current_year = pd.Timestamp.now().year
        valid_year_mask = (df["release_year"] >= 1888) & (df["release_year"] <= current_year + 10)
        df.loc[~valid_year_mask, "release_year"] = pd.NA

        df["release_decade"] = (df["release_year"] // 10 * 10).astype("Int64")

    if "adult" in df.columns:
        df["adult"] = df["adult"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    def _split_list_field(s):
        if s is None or s == "":
            return []
        return [item.strip() for item in str(s).split(",") if item.strip()]

    for field in ["genres", "production_countries", "spoken_languages", "keywords", "production_companies"]:
        if field in df.columns:
            df[field] = df[field].apply(_split_list_field)

    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].str.strip().str.lower().replace({"": np.nan})

    df["has_revenue"] = df.get("revenue", pd.Series(dtype=float)).notna()
    df["has_runtime"] = df.get("runtime", pd.Series(dtype=float)).notna()
    df["has_vote"] = df.get("vote_average", pd.Series(dtype=float)).notna()
    return df

 

def add_bins(df: pd.DataFrame, rating_bin_width: float, min_rating: float, max_rating: float,
             revenue_bins: List[float], runtime_bins: List[float]) -> pd.DataFrame:
    # Rating bins (e.g., [7.00, 8.00) displayed as 7.00–7.99)
    rating_edges = list(np.arange(min_rating, max_rating + rating_bin_width, rating_bin_width))
    if rating_edges[-1] < max_rating:
        rating_edges.append(max_rating)
    # Add a tiny bit to the last edge to include max_rating (e.g., 10.0) in the last bin
    if rating_edges[-1] == max_rating:
        rating_edges[-1] = max_rating + 0.001
    rating_labels = _format_left_closed_interval(rating_edges, decimals=2)
    df["rating_bin"] = pd.cut(
        df["vote_average"], bins=rating_edges, right=False, include_lowest=True, labels=rating_labels
    )

    # Revenue bins
    rev_edges = revenue_bins
    rev_labels = _format_left_closed_interval(rev_edges, decimals=0)
    df["revenue_bin"] = pd.cut(
        df["revenue"], bins=rev_edges, right=False, include_lowest=True, labels=rev_labels
    )

    # Runtime bins
    run_edges = runtime_bins
    run_labels = _format_left_closed_interval_int(run_edges)
    df["runtime_bin"] = pd.cut(
        df["runtime"], bins=run_edges, right=False, include_lowest=True, labels=run_labels
    )
    return df


def write_rollups(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    def write_count(series: pd.Series, fname: str):
        out = series.value_counts(dropna=False).rename_axis(series.name).reset_index(name="movie_count")
        # Move NaN (if any) to the bottom
        out["is_na"] = out[series.name].isna()
        out = out.sort_values(by=["is_na", "movie_count"], ascending=[True, False]).drop(columns=["is_na"])
        out.to_csv(outdir / fname, index=False)

    # By year & decade
    if "release_year" in df.columns:
        write_count(df["release_year"], "movies_by_year.csv")
    if "release_decade" in df.columns:
        write_count(df["release_decade"], "movies_by_decade.csv")

    # By rating bin
    if "rating_bin" in df.columns:
        # Keep categorical order
        out = (
            df["rating_bin"]
            .value_counts(dropna=False)
            .rename_axis("rating_bin")
            .reset_index(name="movie_count")
        )
        if isinstance(df["rating_bin"].dtype, pd.CategoricalDtype):
            out["order"] = out["rating_bin"].cat.codes
            out = out.sort_values("order").drop(columns=["order"])
        out.to_csv(outdir / "movies_by_rating_bin.csv", index=False)

    # By revenue bin
    if "revenue_bin" in df.columns:
        out = (
            df["revenue_bin"]
            .value_counts(dropna=False)
            .rename_axis("revenue_bin")
            .reset_index(name="movie_count")
        )
        if isinstance(df["revenue_bin"].dtype, pd.CategoricalDtype):
            out["order"] = out["revenue_bin"].cat.codes
            out = out.sort_values("order").drop(columns=["order"])
        out.to_csv(outdir / "movies_by_revenue_bin.csv", index=False)

    # By runtime bin
    if "runtime_bin" in df.columns:
        out = (
            df["runtime_bin"]
            .value_counts(dropna=False)
            .rename_axis("runtime_bin")
            .reset_index(name="movie_count")
        )
        if isinstance(df["runtime_bin"].dtype, pd.CategoricalDtype):
            out["order"] = out["runtime_bin"].cat.codes
            out = out.sort_values("order").drop(columns=["order"])
        out.to_csv(outdir / "movies_by_runtime_bin.csv", index=False)

    # By original language
    if "original_language" in df.columns:
        write_count(df["original_language"], "movies_by_language.csv")

    # Exploded by region (production_countries)
    if "production_countries" in df.columns:
        expl = df.explode("production_countries")
        out = (
            expl["production_countries"]
            .dropna()
            .value_counts()
            .rename_axis("country")
            .reset_index(name="movie_count")
            .sort_values("movie_count", ascending=False)
        )
        out.to_csv(outdir / "movies_by_region.csv", index=False)

    # Exploded by genre
    if "genres" in df.columns:
        expl = df.explode("genres")
        out = (
            expl["genres"]
            .dropna()
            .value_counts()
            .rename_axis("genre")
            .reset_index(name="movie_count")
            .sort_values("movie_count", ascending=False)
        )
        out.to_csv(outdir / "movies_by_genre.csv", index=False)

    # Skinny per-movie facts denorm (useful for dashboards/joins)
    keep_cols = [
        c for c in [
            "id", "title", "original_title", "original_language", "release_date",
            "release_year", "release_decade", "vote_average", "vote_count",
            "revenue", "budget", "runtime", "adult", "popularity",
            "rating_bin", "revenue_bin", "runtime_bin"
        ] if c in df.columns
    ]
    df[keep_cols].to_csv(outdir / "movies_facts_denorm.csv", index=False)


def main():
    args = _parse_args()
    outdir = Path(args.outdir)

    df = load_and_clean(args.csv_path)
    df = add_bins(
        df,
        rating_bin_width=args.rating_bin_width,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        revenue_bins=args.revenue_bins,
        runtime_bins=args.runtime_bins,
    )
    write_rollups(df, outdir)

    print(f"Wrote denormalized CSVs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
