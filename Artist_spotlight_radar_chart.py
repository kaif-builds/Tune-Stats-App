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

component_id = "artist_spotlight_radar_chart"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    artist_dropdown_id = f"{component_id}_artist"
    compare_toggle_id = f"{component_id}_compare"

    # Get data to populate artist dropdown
    df = get_data()
    unique_artists = df['artists'].dropna().replace('', np.nan).dropna().unique().tolist()
    unique_artists = [artist for artist in unique_artists if artist is not None and str(artist).strip()]
    unique_artists = sorted(unique_artists)
    
    artist_options = [{"label": str(artist), "value": artist} for artist in unique_artists]
    artist_default = unique_artists[0] if unique_artists else ""

    title = "Artist Spotlight Radar Chart"
    description = "Visualize an artist's audio DNA profile with optional genre comparison"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Select Artist:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=artist_dropdown_id,
                                options=artist_options,
                                value=artist_default,
                                style={"minWidth": "300px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Compare with Genre Average:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Checklist(
                                id=compare_toggle_id,
                                options=[{"label": "", "value": "enabled"}],
                                value=[],
                                style={"minWidth": "150px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
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
        width=50
    )

    test_inputs: dict[str, TestInput] = {
        artist_dropdown_id: {
            "options": [artist for artist in unique_artists[:10]],  # Limit for testing
            "default": artist_default
        },
        compare_toggle_id: {
            "options": [[], ["enabled"]],
            "default": []
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _update_logic(**kwargs) -> go.Figure:
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
        return empty_fig

    selected_artist = kwargs.get(f'{component_id}_artist')
    compare_enabled = 'enabled' in kwargs.get(f'{component_id}_compare', [])

    if not selected_artist:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Please select an artist",
            annotations=[{
                "text": "Please select an artist from the dropdown",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    # Filter data for selected artist
    artist_df = df[df['artists'] == selected_artist]
    
    if len(artist_df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data for selected artist",
            annotations=[{
                "text": f"No data available for {selected_artist}",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Starting chart creation. Artist: %s, df shape: %s", selected_artist, artist_df.shape)

    # Audio features for radar chart
    audio_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'speechiness', 'valence'
    ]

    # Calculate artist averages
    artist_values = []
    feature_labels = []
    
    for feature in audio_features:
        if feature in artist_df.columns:
            avg_value = artist_df[feature].mean()
            artist_values.append(avg_value)
            feature_labels.append(feature.title())

    # Ensure we have data to plot
    if not artist_values:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No audio features available",
            annotations=[{
                "text": "No audio features data available for this artist",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    # Create radar chart
    fig = go.Figure()

    # Add artist radar trace with bright cyan color
    fig.add_trace(go.Scatterpolar(
        r=artist_values,
        theta=feature_labels,
        fill='toself',
        name=selected_artist,
        line=dict(color='cyan', width=3),
        fillcolor='rgba(0, 255, 255, 0.3)',
        hovertemplate=f"<b>{selected_artist}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>"
    ))

    # Add genre average if comparison is enabled
    if compare_enabled:
        # Get genre data for comparison
        genre_values = []
        for feature in audio_features:
            if feature in df.columns:
                # Use overall dataset average as "genre average"
                avg_value = df[feature].mean()
                genre_values.append(avg_value)

        if genre_values:
            fig.add_trace(go.Scatterpolar(
                r=genre_values,
                theta=feature_labels,
                fill='toself',
                name='Dataset Average',
                line=dict(color='orange', width=3),
                fillcolor='rgba(255, 165, 0, 0.2)',
                hovertemplate="<b>Dataset Average</b><br>%{theta}: %{r:.3f}<extra></extra>"
            ))

    # Update layout for high contrast dark theme
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='white',
                gridwidth=1,
                tickcolor='white',
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                gridcolor='white',
                gridwidth=1,
                tickcolor='white',
                tickfont=dict(color='white', size=12)
            )
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        f'{component_id}_artist': Input(f"{component_id}_artist", "value"),
        f'{component_id}_compare': Input(f"{component_id}_compare", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_logic(**kwargs)
        return figure, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg