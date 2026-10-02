"""
This file pulls the play-by-play and caching it locally to avoid having to re-download every run.
"""

import os
import pandas as pd
import nfl_data_py as nfl
 
CACHE_DIR = "data_cache"
 
 
def load_pbp_data(years, force_refresh=False):
    """
    Load play-by-play data for the given years, using a local parquet
    cache to avoid re-downloading from nflverse on every run.
 
    Parameters
    ----------
    years : list[int]
        Seasons to load, e.g. list(range(2015, 2024))
    force_refresh : bool
        If True, ignore the cache and re-download.
 
    Returns
    -------
    pd.DataFrame
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f"pbp_{min(years)}_{max(years)}.parquet"
    )
 
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
 
    print(f"Downloading play-by-play data for {min(years)}-{max(years)} from nflverse...")
    pbp = nfl.import_pbp_data(years)
 
    pbp.to_parquet(cache_path)
    print(f"Cached to {cache_path} ({pbp.shape[0]} rows, {pbp.shape[1]} cols)")
 
    return pbp
 
 
if __name__ == "__main__":
    # Quick manual test: python data_loader.py
    years = list(range(2015, 2024))
    df = load_pbp_data(years)
    print(df.shape)
    print(df["play_type"].value_counts())
 