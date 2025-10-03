import os
import sys
from datetime import datetime, time
from typing import TypedDict, Any

from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import numpy as np
import pandas as pd

from data import get_data, cache
from logger import logger

class FILTER_COMPONENT_IDS:
    '''
    A map of all component IDs used in the filter.
    These should all be column names of columns that will be filtered.
    IMPORTANT - Use underscores not hyphens
    '''
    genre_multiselect_filter = "genre_multiselect_filter"
    artist_multiselect_filter = "artist_multiselect_filter"
    popularity_min = "popularity_min"
    popularity_max = "popularity_max"
    energy_min = "energy_min"
    energy_max = "energy_max"
    danceability_min = "danceability_min"
    danceability_max = "danceability_max"
    valence_min = "valence_min"
    valence_max = "valence_max"
    tempo_min = "tempo_min"
    tempo_max = "tempo_max"
    duration_min = "duration_min"
    duration_max = "duration_max"


FILTER_CALLBACK_INPUTS = {
    "genre_multiselect_filter": Input(FILTER_COMPONENT_IDS.genre_multiselect_filter, "value"),
    "artist_multiselect_filter": Input(FILTER_COMPONENT_IDS.artist_multiselect_filter, "value"),
    "popularity_min": Input(FILTER_COMPONENT_IDS.popularity_min, "value"),
    "popularity_max": Input(FILTER_COMPONENT_IDS.popularity_max, "value"),
    "energy_min": Input(FILTER_COMPONENT_IDS.energy_min, "value"),
    "energy_max": Input(FILTER_COMPONENT_IDS.energy_max, "value"),
    "danceability_min": Input(FILTER_COMPONENT_IDS.danceability_min, "value"),
    "danceability_max": Input(FILTER_COMPONENT_IDS.danceability_max, "value"),
    "valence_min": Input(FILTER_COMPONENT_IDS.valence_min, "value"),
    "valence_max": Input(FILTER_COMPONENT_IDS.valence_max, "value"),
    "tempo_min": Input(FILTER_COMPONENT_IDS.tempo_min, "value"),
    "tempo_max": Input(FILTER_COMPONENT_IDS.tempo_max, "value"),
    "duration_min": Input(FILTER_COMPONENT_IDS.duration_min, "value"),
    "duration_max": Input(FILTER_COMPONENT_IDS.duration_max, "value"),
}

class TestInput(TypedDict):
    options: list[Any]
    default: Any

class ComponentResponse(TypedDict):
    layout: html.Div
    test_inputs: dict[str, TestInput]


def component() -> ComponentResponse:
    df = get_data()

    logger.debug("Filter component data loaded. Shape: %s", df.shape)
    logger.debug("Filter component sample data:\n%s", df.head())

    # Extract unique genres from the genres column
    all_genres = []
    for genre_list_str in df["genres"].dropna():
        if genre_list_str and genre_list_str != "[]":
            try:
                # Parse the string representation of the list
                genre_list_str = genre_list_str.strip("[]")
                if genre_list_str:
                    genres = [g.strip().strip("'\"") for g in genre_list_str.split(",")]
                    all_genres.extend([g for g in genres if g])
            except:
                continue
    
    unique_genres = sorted(list(set(all_genres)))
    
    # Get unique artists
    unique_artists = sorted(df["artists"].dropna().unique().tolist())

    # Compute ranges
    popularity_min = 0
    popularity_max = 100
    energy_min = 0.0
    energy_max = 1.0
    danceability_min = 0.0
    danceability_max = 1.0
    valence_min = 0.0
    valence_max = 1.0
    tempo_min = 0
    tempo_max = 220
    duration_min = 18
    duration_max = 5400

    layout = html.Div([ddk._ControlPanel(
        position="top",
        default_open=True,
        control_groups=[
            {
                "title": "Filters",
                "id": "filter_control_group",
                "description": "",
                "children": [
                    html.Div(
                        children=dcc.Dropdown(
                            id=FILTER_COMPONENT_IDS.genre_multiselect_filter,
                            options=[{"label": "All", "value": "all"}] + [{"label": g, "value": g} for g in unique_genres],
                            multi=True,
                            value=["all"]
                        ),
                        id=FILTER_COMPONENT_IDS.genre_multiselect_filter + "_parent",
                        title="Genre",
                        style={"minWidth": "200px"}
                    ),

                    html.Div(
                        children=dcc.Dropdown(
                            id=FILTER_COMPONENT_IDS.artist_multiselect_filter,
                            options=[{"label": "All", "value": "all"}] + [{"label": a, "value": a} for a in unique_artists],
                            multi=True,
                            value=["all"]
                        ),
                        id=FILTER_COMPONENT_IDS.artist_multiselect_filter + "_parent",
                        title="Artist",
                        style={"minWidth": "200px"}
                    ),

                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.popularity_min, value=popularity_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.popularity_max, value=popularity_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Popularity Range (0-100)"
                    ),

                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.energy_min, value=energy_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.energy_max, value=energy_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Energy Level (0.0-1.0)"
                    ),

                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.danceability_min, value=danceability_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.danceability_max, value=danceability_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Danceability (0.0-1.0)"
                    ),

                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.valence_min, value=valence_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.valence_max, value=valence_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Musical Positivity (Valence) (0.0-1.0)"
                    ),

                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.tempo_min, value=tempo_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.tempo_max, value=tempo_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Tempo (BPM) (0-220)"
                    ),

                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.duration_min, value=duration_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.duration_max, value=duration_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Song Duration (18-5400 seconds)"
                    )
                ],
            },
        ],
    ), html.Div(id='total_results', style={ 'paddingTop': 20, 'marginLeft': 50, 'fontStyle': 'italic', 'minHeight': 45 })])

    test_inputs: dict[str, TestInput] = {
        "genre_multiselect_filter": {
            "options": ["all"] + unique_genres[:3],
            "default": ["all"]
        },
        "artist_multiselect_filter": {
            "options": ["all"] + unique_artists[:3],
            "default": ["all"]
        },
        "popularity_min": {
            "options": [0, 25, 50],
            "default": 0
        },
        "popularity_max": {
            "options": [100, 75, 50],
            "default": 100
        },
        "energy_min": {
            "options": [0.0, 0.25, 0.5],
            "default": 0.0
        },
        "energy_max": {
            "options": [1.0, 0.75, 0.5],
            "default": 1.0
        },
        "danceability_min": {
            "options": [0.0, 0.25, 0.5],
            "default": 0.0
        },
        "danceability_max": {
            "options": [1.0, 0.75, 0.5],
            "default": 1.0
        },
        "valence_min": {
            "options": [0.0, 0.25, 0.5],
            "default": 0.0
        },
        "valence_max": {
            "options": [1.0, 0.75, 0.5],
            "default": 1.0
        },
        "tempo_min": {
            "options": [0, 55, 110],
            "default": 0
        },
        "tempo_max": {
            "options": [220, 165, 110],
            "default": 220
        },
        "duration_min": {
            "options": [18, 100, 200],
            "default": 18
        },
        "duration_max": {
            "options": [5400, 300, 200],
            "default": 5400
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

@cache.memoize()
def filter_data(df, **filters):
    logger.debug("Starting data filtering. Original shape: %s", df.shape)
    logger.debug("Applied filters: %s", filters)

    df = df.copy()

    # Filter by genre
    if len(filters["genre_multiselect_filter"]) > 0 and "all" not in filters["genre_multiselect_filter"]:
        selected_genres = filters["genre_multiselect_filter"]
        mask = df["genres"].apply(lambda x: any(genre in x for genre in selected_genres) if x and x != "[]" else False)
        df = df[mask]

    # Filter by artist
    if len(filters["artist_multiselect_filter"]) > 0 and "all" not in filters["artist_multiselect_filter"]:
        df = df[df["artists"].isin(filters["artist_multiselect_filter"])]

    # Filter by popularity range
    if "popularity" in df.columns:
        df = df[df["popularity"] >= float(filters["popularity_min"])]
        df = df[df["popularity"] <= float(filters["popularity_max"])]

    # Filter by energy range
    if "energy" in df.columns:
        df = df[df["energy"] >= float(filters["energy_min"])]
        df = df[df["energy"] <= float(filters["energy_max"])]

    # Filter by danceability range
    if "danceability" in df.columns:
        df = df[df["danceability"] >= float(filters["danceability_min"])]
        df = df[df["danceability"] <= float(filters["danceability_max"])]

    # Filter by valence range
    if "valence" in df.columns:
        df = df[df["valence"] >= float(filters["valence_min"])]
        df = df[df["valence"] <= float(filters["valence_max"])]

    # Filter by tempo range
    if "tempo" in df.columns:
        df = df[df["tempo"] >= float(filters["tempo_min"])]
        df = df[df["tempo"] <= float(filters["tempo_max"])]

    # Filter by duration range (convert from ms to seconds)
    if "duration_ms" in df.columns:
        duration_seconds = df["duration_ms"] / 1000
        df = df[(duration_seconds >= float(filters["duration_min"])) & (duration_seconds <= float(filters["duration_max"]))]

    logger.debug("Filtering complete. Final shape: %s", df.shape)
    logger.debug("Filtered data sample:\n%s", df.head())

    return df

@callback(Output("total_results", "children"), inputs=FILTER_CALLBACK_INPUTS)
def display_count(**kwargs):
    df = get_data()
    # Get total count
    count = len(df)

    filtered_df = filter_data(df, **kwargs)
    # Get filtered count
    filtered_count = len(filtered_df)

    return f"{filtered_count:,} / {count:,} rows"