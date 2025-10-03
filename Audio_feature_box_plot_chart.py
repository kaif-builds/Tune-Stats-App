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

component_id = "audio_feature_distribution_as"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    feature_control_id = f"{component_id}_feature"
    genre_control_id = f"{component_id}_genres"

    # Audio features available for selection
    audio_features = [
        {"label": "Acousticness", "value": "acousticness"},
        {"label": "Danceability", "value": "danceability"},
        {"label": "Energy", "value": "energy"},
        {"label": "Instrumentalness", "value": "instrumentalness"},
        {"label": "Liveness", "value": "liveness"},
        {"label": "Loudness", "value": "loudness"},
        {"label": "Speechiness", "value": "speechiness"},
        {"label": "Tempo", "value": "tempo"},
        {"label": "Valence", "value": "valence"}
    ]
    feature_default = "valence"

    # Get unique genres for multi-select (will be populated dynamically)
    df = get_data()
    # Parse genres from string representation of lists
    all_genres = []
    for genre_str in df['genres'].dropna():
        if genre_str != "[]":
            # Remove brackets and quotes, split by comma
            genres = genre_str.strip("[]").replace("'", "").replace('"', '').split(", ")
            all_genres.extend([g.strip() for g in genres if g.strip()])
    
    unique_genres = sorted(list(set(all_genres)))[:20]  # Limit to top 20 for performance
    genre_options = [{"label": genre, "value": genre} for genre in unique_genres]
    genre_default = unique_genres[:5] if len(unique_genres) >= 5 else unique_genres

    title = "Audio Feature Distribution"
    description = "Compare distributions of audio features across different genres using box plots"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            # Summary cards
            html.Div(
                id=f"{component_id}_summary",
                style={"marginBottom": "20px", "display": "flex", "gap": "15px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        children=[
                            html.H4("Total Tracks", style={"margin": "0", "fontSize": "14px", "fontWeight": "bold", "color": "black"}),
                            html.Div(id=f"{component_id}_total_tracks", style={"fontSize": "24px", "fontWeight": "bold", "color": "black"})
                        ],
                        style={
                            "padding": "15px",
                            "backgroundColor": "white",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "minWidth": "150px"
                        }
                    ),
                    html.Div(
                        children=[
                            html.H4("Average Popularity", style={"margin": "0", "fontSize": "14px", "fontWeight": "bold", "color": "black"}),
                            html.Div(id=f"{component_id}_avg_popularity", style={"fontSize": "24px", "fontWeight": "bold", "color": "black"})
                        ],
                        style={
                            "padding": "15px",
                            "backgroundColor": "white",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "minWidth": "150px"
                        }
                    )
                ]
            ),
            # Controls
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Audio Feature:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=feature_control_id,
                                options=audio_features,
                                value=feature_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Genres to Compare:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=genre_control_id,
                                options=genre_options,
                                value=genre_default,
                                multi=True,
                                style={"minWidth": "300px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    )
                ]
            ),
            # Graph with loading
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    ddk.Graph(id=graph_id)
                ]
            ),
            html.Pre(id=error_id, style={"color": "red", "margin": "10px 0"}),
            ddk.CardFooter(title=description)
        ],
        width=50
    )

    test_inputs: dict[str, TestInput] = {
        feature_control_id: {
            "options": [option["value"] for option in audio_features],
            "default": feature_default
        },
        genre_control_id: {
            "options": [option["value"] for option in genre_options],
            "default": genre_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _update_logic(**kwargs) -> Tuple[go.Figure, str, str]:
    """Core chart update logic without error handling."""
    df = filter_data(get_data(), **kwargs)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No data available",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, "0", "0"

    # Calculate summary statistics
    total_tracks = len(df)
    avg_popularity = df['popularity'].mean()
    
    # Extract control values
    selected_feature = kwargs.get(f'{component_id}_feature', 'valence')
    selected_genres = kwargs.get(f'{component_id}_genres', [])
    
    if selected_feature is None:
        selected_feature = 'valence'
    if selected_genres is None or len(selected_genres) == 0:
        # If no genres selected, show message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "Please select at least one genre to compare",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig, f"{total_tracks:,}", f"{avg_popularity:.1f}"

    # Process genres and create data for box plot
    plot_data = []
    
    for _, row in df.iterrows():
        genre_str = row['genres']
        if genre_str != "[]":
            # Parse genres from string
            genres = genre_str.strip("[]").replace("'", "").replace('"', '').split(", ")
            genres = [g.strip() for g in genres if g.strip()]
            
            # Check if any selected genre is in this track's genres
            for genre in genres:
                if genre in selected_genres:
                    plot_data.append({
                        'genre': genre,
                        'feature_value': row[selected_feature],
                        'artist': row['artists']
                    })

    if len(plot_data) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No data available for selected genres",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig, f"{total_tracks:,}", f"{avg_popularity:.1f}"

    plot_df = pd.DataFrame(plot_data)
    
    logger.debug("Starting chart creation. plot_df:\n%s", plot_df.head())
    
    # Create box plot
    fig = px.box(
        plot_df,
        x='genre',
        y='feature_value',
        title=f"Distribution of {selected_feature.title()} by Genre"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title=selected_feature.title(),
        showlegend=False
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig, f"{total_tracks:,}", f"{avg_popularity:.1f}"

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_total_tracks", "children"),
        Output(f"{component_id}_avg_popularity", "children")
    ],
    inputs={
        f'{component_id}_feature': Input(f"{component_id}_feature", "value"),
        f'{component_id}_genres': Input(f"{component_id}_genres", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, str, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure, total_tracks, avg_popularity = _update_logic(**kwargs)
        return figure, "", total_tracks, avg_popularity

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg, "0", "0"