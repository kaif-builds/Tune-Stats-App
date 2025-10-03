import json
import pandas as pd
import numpy as np
from logger import logger

from flask_caching import Cache

cache = Cache()

@cache.memoize()
def get_data(data_path='data/data.csv'):
    # Define explicit type mappings for all columns
    type_mapping = {
        # String columns
        "genres": str,
        "artists": str,
        
        # Numeric columns
        "acousticness": "Float64",
        "danceability": "Float64", 
        "duration_ms": "Float64",
        "energy": "Float64",
        "instrumentalness": "Float64",
        "liveness": "Float64",
        "loudness": "Float64",
        "speechiness": "Float64",
        "tempo": "Float64",
        "valence": "Float64",
        "popularity": "Float64",
        "key": "Int64",
        "mode": "Int64",
        "count": "Int64",
        
        # Artist average columns
        "mode_artist_avg": "Int64",
        "count_artist_avg": "Int64",
        "acousticness_artist_avg": "Float64",
        "danceability_artist_avg": "Float64",
        "duration_ms_artist_avg": "Float64",
        "energy_artist_avg": "Float64",
        "instrumentalness_artist_avg": "Float64",
        "liveness_artist_avg": "Float64",
        "loudness_artist_avg": "Float64",
        "speechiness_artist_avg": "Float64",
        "tempo_artist_avg": "Float64",
        "valence_artist_avg": "Float64",
        "popularity_artist_avg": "Float64",
        "key_artist_avg": "Int64",
        
        # Genre average columns (mixed types - mostly numeric with empty strings)
        "mode_genre_avg": str,
        "acousticness_genre_avg": str,
        "danceability_genre_avg": str,
        "duration_ms_genre_avg": str,
        "energy_genre_avg": str,
        "instrumentalness_genre_avg": str,
        "liveness_genre_avg": str,
        "loudness_genre_avg": str,
        "speechiness_genre_avg": str,
        "tempo_genre_avg": str,
        "valence_genre_avg": str,
        "popularity_genre_avg": str,
        "key_genre_avg": str,
    }

    # Define column-specific values to treat as NaN (for mixed type columns)
    na_values_mapping = {
        # Genre average columns have empty strings that should be treated as NaN
        "mode_genre_avg": [""],
        "acousticness_genre_avg": [""],
        "danceability_genre_avg": [""],
        "duration_ms_genre_avg": [""],
        "energy_genre_avg": [""],
        "instrumentalness_genre_avg": [""],
        "liveness_genre_avg": [""],
        "loudness_genre_avg": [""],
        "speechiness_genre_avg": [""],
        "tempo_genre_avg": [""],
        "valence_genre_avg": [""],
        "popularity_genre_avg": [""],
        "key_genre_avg": [""],
    }
    
    # Load data based on file extension
    if data_path.endswith('.parquet'):
        # Read Parquet files
        df = pd.read_parquet(data_path, engine='pyarrow')
    else:
        # Read CSV with explicit type mapping and column-specific na_values and automatic separator detection
        df = pd.read_csv(data_path, dtype=type_mapping, na_values=na_values_mapping, sep=None, engine="python", encoding="utf-8-sig")
    
    logger.debug("Data loaded. Shape: %s", df.shape)
    logger.debug("Sample data:\n%s", df.head())

    # Convert genre average columns to numeric (they were loaded as strings to handle empty values)
    genre_avg_columns = [
        "mode_genre_avg", "acousticness_genre_avg", "danceability_genre_avg", 
        "duration_ms_genre_avg", "energy_genre_avg", "instrumentalness_genre_avg",
        "liveness_genre_avg", "loudness_genre_avg", "speechiness_genre_avg",
        "tempo_genre_avg", "valence_genre_avg", "popularity_genre_avg", "key_genre_avg"
    ]
    
    for col in genre_avg_columns:
        if col in df.columns:
            if col == "mode_genre_avg" or col == "key_genre_avg":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Float64")

    # Filter out rows where all values are null
    df = df.dropna(how='all')

    logger.debug("Data processed. Final shape: %s", df.shape)
    logger.debug("Data types:\n%s", df.dtypes)

    return df