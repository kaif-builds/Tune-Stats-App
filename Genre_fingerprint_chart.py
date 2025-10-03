import sys
import os
import traceback
from typing import TypedDict, Any, Tuple
from datetime import datetime, timedelta

from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from logger import logger

class TestInput(TypedDict):
    options: list[Any]
    default: Any

class ComponentResponse(TypedDict):
    layout: ddk.Card
    test_inputs: dict[str, TestInput]

component_id = "genre_sonic_fingerprint_heatmap"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    
    normalization_id = f"{component_id}_normalization"
    normalization_options = [
        {"label": "Raw Values", "value": "raw"},
        {"label": "Normalized (0-1)", "value": "normalized"}
    ]
    normalization_default = "normalized"

    title = "Genre Sonic Fingerprint Heatmap"
    description = "Heatmap showing average audio features for top genres with summary statistics"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Value Type:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=normalization_id,
                                options=normalization_options,
                                value=normalization_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(
                id=f"{component_id}_summary_cards",
                style={"marginBottom": "20px", "display": "flex", "flexWrap": "wrap", "gap": "15px"}
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    ddk.Graph(id=graph_id),
                ]
            ),
            html.Pre(id=error_id, style={"color": "red", "margin": "10px 0"}),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        normalization_id: {
            "options": [option["value"] for option in normalization_options],
            "default": normalization_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _update_logic(**kwargs) -> Tuple[go.Figure, list]:
    """Core chart update logic without error handling."""
    df = filter_data(get_data(), **kwargs)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No data is available to display",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, []

    normalization = kwargs.get(f'{component_id}_normalization', 'normalized')
    if normalization is None:
        normalization = 'normalized'

    logger.debug("Starting genre sonic fingerprint heatmap creation. df shape: %s", df.shape)

    # Parse genres and create genre-track mapping
    genre_tracks = []
    for idx, row in df.iterrows():
        try:
            genres_str = row['genres']
            if genres_str and genres_str != '[]':
                genres_str = genres_str.strip('[]')
                if genres_str:
                    genres = [g.strip().strip("'\"") for g in genres_str.split(',')]
                    genres = [g for g in genres if g and g != '']
                    
                    for genre in genres:
                        genre_tracks.append({
                            'genre': genre,
                            'acousticness': row['acousticness'],
                            'danceability': row['danceability'],
                            'energy': row['energy'],
                            'instrumentalness': row['instrumentalness'],
                            'liveness': row['liveness'],
                            'speechiness': row['speechiness'],
                            'valence': row['valence'],
                            'popularity': row['popularity']
                        })
        except Exception as e:
            continue

    if not genre_tracks:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No genre data available",
            annotations=[{
                "text": "No valid genre data found in the filtered dataset",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, []

    # Convert to DataFrame and aggregate by genre
    genre_df = pd.DataFrame(genre_tracks)
    
    # Get top 15-20 genres by track count
    genre_counts = genre_df['genre'].value_counts()
    top_genres = genre_counts.head(20).index.tolist()
    
    # Filter to top genres and calculate averages
    genre_df_filtered = genre_df[genre_df['genre'].isin(top_genres)]
    
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                     'liveness', 'speechiness', 'valence']
    
    genre_averages = genre_df_filtered.groupby('genre')[audio_features].mean().reset_index()
    
    # Sort by total track count (descending)
    genre_order = [genre for genre in top_genres if genre in genre_averages['genre'].values]
    genre_averages['genre'] = pd.Categorical(genre_averages['genre'], categories=genre_order, ordered=True)
    genre_averages = genre_averages.sort_values('genre')

    # Prepare data for heatmap
    heatmap_data = genre_averages[audio_features].values
    
    if normalization == 'normalized':
        # Manual normalization using numpy (0-1 scale for each feature)
        heatmap_data_normalized = np.zeros_like(heatmap_data)
        for i, feature in enumerate(audio_features):
            feature_values = heatmap_data[:, i]
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            if max_val > min_val:
                heatmap_data_normalized[:, i] = (feature_values - min_val) / (max_val - min_val)
            else:
                heatmap_data_normalized[:, i] = 0.5  # If all values are the same, set to middle
        heatmap_data = heatmap_data_normalized
        colorscale = 'Viridis'
        zmin, zmax = 0, 1
    else:
        colorscale = 'Viridis'
        zmin, zmax = None, None

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[feature.title() for feature in audio_features],
        y=genre_averages['genre'].tolist(),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Audio Features",
        yaxis_title="Genres",
        height=max(400, len(genre_order) * 25),
        margin={"l": 150, "r": 50, "t": 50, "b": 50}
    )

    # Calculate summary statistics
    total_tracks = len(df)
    avg_popularity = df['popularity'].mean()

    summary_cards = [
        html.Div([
            html.H4("Total Tracks", style={"margin": "0", "fontSize": "16px", "color": "#333"}),
            html.H2(f"{total_tracks:,}", style={"margin": "5px 0 0 0", "color": "#333"})
        ], style={
            "padding": "15px", 
            "backgroundColor": "white", 
            "borderRadius": "8px", 
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "border": "1px solid #ddd",
            "textAlign": "center",
            "minWidth": "150px"
        }),
        html.Div([
            html.H4("Average Popularity", style={"margin": "0", "fontSize": "16px", "color": "#333"}),
            html.H2(f"{avg_popularity:.1f}", style={"margin": "5px 0 0 0", "color": "#333"})
        ], style={
            "padding": "15px", 
            "backgroundColor": "white", 
            "borderRadius": "8px", 
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "border": "1px solid #ddd",
            "textAlign": "center",
            "minWidth": "150px"
        })
    ]

    return fig, summary_cards

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_summary_cards", "children")
    ],
    inputs={
        f'{component_id}_normalization': Input(f"{component_id}_normalization", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, list]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure, summary_cards = _update_logic(**kwargs)
        return figure, "", summary_cards

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg, []