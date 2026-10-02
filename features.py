"""
This file will transform play-byplay data into a encoded feature matrix.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "shotgun",
    "no_huddle",
    "posteam",
    "defteam",
]

CATEGORICAL_COLS = ["posteam", "defteam"]

NUMERIC_COLS = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
]

def filter_plays(pbp):
    """Exclude invalid plays (special teams and bad data)"""
    df = pbp[pbp["play_type"].isin(["run", "pass"])].copy()
    df = df[df["down"].notna()]
    return df

def season_split(df, test_seasons_start):
    """Split data by seasons for training and testing"""
    train_df = df[df["season"] < test_seasons_start].copy()
    test_df = df[df["season"] >= test_seasons_start].copy()
    return train_df, test_df

def build_features(train_df, test_df):
    """We need to encode our categorical data, scale numerical data, and align training/testing columns. 
        Returns:
        x_train, x_test, y_train, y_test, scaler, feature_columns
    """
    cols_needed = FEATURE_COLS + ["play_type"]
    train_df = train_df[cols_needed].dropna()
    test_df = test_df[cols_needed].dropna()


    #pass=1 and run=0
    y_train = (train_df["play_type"] == "pass").astype(int)
    y_test = (test_df["play_type"] == "pass").astype(int)
    train_enc = pd.get_dummies(train_df.drop(columns=["play_type"]), columns=CATEGORICAL_COLS)
    test_enc = pd.get_dummies(test_df.drop(columns=["play_type"]), columns=CATEGORICAL_COLS)

    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)
    scaler = StandardScaler()
    train_enc[NUMERIC_COLS] = scaler.fit_transform(train_enc[NUMERIC_COLS])
    test_enc[NUMERIC_COLS] = scaler.transform(test_enc[NUMERIC_COLS])
    return train_enc, test_enc, y_train, y_test, scaler, list(train_enc.columns)