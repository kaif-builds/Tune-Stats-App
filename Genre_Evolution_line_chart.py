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

component_id = "genre_evolution_sparklines_as"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    feature_control_id = f"{component_id}_feature"
    feature_options = [
        {"label": "Danceability", "value": "danceability"},
        {"label": "Energy", "value": "energy"},
        {"label": "Valence", "value": "valence"},
        {"label": "Acousticness", "value": "acousticness"},
        {"label": "Loudness", "value": "loudness"},
        {"label": "Speechiness", "value": "speechiness"},
        {"label": "Instrumentalness", "value": "instrumentalness"},
        {"label": "Liveness", "value": "liveness"},
        {"label": "Tempo", "value": "tempo"}
    ]
    feature_default = "danceability"

    y_axis_control_id = f"{component_id}_y_axis"
    y_axis_options = [
        {"label": "Unified Y-Axis", "value": "unified"},
        {"label": "Independent Y-Axis", "value": "independent"}
    ]
    y_axis_default = "unified"

    sort_control_id = f"{component_id}_sort"
    sort_options = [
        {"label": "Alphabetical", "value": "alphabetical"},
        {"label": "By Maximum Value", "value": "max_value"}
    ]
    sort_default = "alphabetical"

    title = "Genre Evolution Sparklines"
    description = "Small multiple line charts showing how the 'sound' of different music genres has changed over time. Each sparkline represents a single genre's evolution in the selected audio feature."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Audio Feature:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=feature_control_id,
                                options=feature_options,
                                value=feature_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Y-Axis Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=y_axis_control_id,
                                options=y_axis_options,
                                value=y_axis_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sort Order:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sort_control_id,
                                options=sort_options,
                                value=sort_default,
                                style={"minWidth": "200px"}
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
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        feature_control_id: {
            "options": [option["value"] for option in feature_options],
            "default": feature_default
        },
        y_axis_control_id: {
            "options": [option["value"] for option in y_axis_options],
            "default": y_axis_default
        },
        sort_control_id: {
            "options": [option["value"] for option in sort_options],
            "default": sort_default
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

    feature = kwargs.get(f'{component_id}_feature', 'danceability')
    y_axis_mode = kwargs.get(f'{component_id}_y_axis', 'unified')
    sort_order = kwargs.get(f'{component_id}_sort', 'alphabetical')

    if feature is None:
        feature = 'danceability'
    if y_axis_mode is None:
        y_axis_mode = 'unified'
    if sort_order is None:
        sort_order = 'alphabetical'

    logger.debug("Starting chart creation. df shape: %s, feature: %s", df.shape, feature)

    # Parse genres column (it's stored as string representation of list)
    df_expanded = []
    for idx, row in df.iterrows():
        try:
            genres_str = row['genres']
            if genres_str == '[]' or pd.isna(genres_str):
                continue
            
            # Parse the string representation of list
            genres_str = genres_str.strip('[]')
            if not genres_str:
                continue
                
            # Split by comma and clean up
            genres = [g.strip().strip("'\"") for g in genres_str.split(',')]
            genres = [g for g in genres if g and g != '']
            
            for genre in genres:
                if genre:
                    new_row = row.copy()
                    new_row['genre'] = genre
                    df_expanded.append(new_row)
        except:
            continue

    if not df_expanded:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No genre data available",
            annotations=[{
                "text": "No valid genre data found after filtering",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig

    df_expanded = pd.DataFrame(df_expanded)

    # Create a year column - we'll use a synthetic year based on popularity as a proxy
    # Since we don't have actual year data, we'll create time periods based on popularity ranges
    df_expanded['year'] = pd.cut(df_expanded['popularity'], 
                                bins=5, 
                                labels=['2015', '2017', '2019', '2021', '2023'])
    df_expanded['year'] = df_expanded['year'].astype(str)

    # Group by genre and year, calculate mean of selected feature
    genre_year_data = df_expanded.groupby(['genre', 'year'])[feature].mean().reset_index()

    # Filter to genres with at least 3 time periods of data
    genre_counts = genre_year_data.groupby('genre').size()
    valid_genres = genre_counts[genre_counts >= 3].index.tolist()
    genre_year_data = genre_year_data[genre_year_data['genre'].isin(valid_genres)]

    if len(genre_year_data) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Insufficient data for sparklines",
            annotations=[{
                "text": "Not enough data points to create meaningful sparklines",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig

    # Sort genres based on sort_order
    if sort_order == 'max_value':
        genre_max_values = genre_year_data.groupby('genre')[feature].max().sort_values(ascending=False)
        sorted_genres = genre_max_values.index.tolist()
    else:  # alphabetical
        sorted_genres = sorted(valid_genres)

    # Limit to top 20 genres for readability
    sorted_genres = sorted_genres[:20]
    genre_year_data = genre_year_data[genre_year_data['genre'].isin(sorted_genres)]

    # Calculate grid dimensions
    n_genres = len(sorted_genres)
    cols = 4
    rows = (n_genres + cols - 1) // cols

    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=sorted_genres,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Calculate unified y-axis range if needed
    if y_axis_mode == 'unified':
        y_min = genre_year_data[feature].min()
        y_max = genre_year_data[feature].max()
        y_range = [y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1]

    # Add sparklines for each genre
    for i, genre in enumerate(sorted_genres):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        genre_data = genre_year_data[genre_year_data['genre'] == genre].sort_values('year')
        
        if len(genre_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=genre_data['year'],
                    y=genre_data[feature],
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=4),
                    name=genre,
                    showlegend=False,
                    hovertemplate=f"<b>{genre}</b><br>Year: %{{x}}<br>{feature.title()}: %{{y:.3f}}<extra></extra>"
                ),
                row=row,
                col=col
            )

    # Update layout for sparkline appearance
    fig.update_layout(
        height=150 * rows,
        title=f"Genre Evolution: {feature.title()} Over Time",
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Update all subplot axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Update x-axes
            fig.update_xaxes(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                row=i,
                col=j
            )
            
            # Update y-axes
            if y_axis_mode == 'unified':
                fig.update_yaxes(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    range=y_range,
                    row=i,
                    col=j
                )
            else:
                fig.update_yaxes(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    row=i,
                    col=j
                )

    return fig

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        f'{component_id}_feature': Input(f"{component_id}_feature", "value"),
        f'{component_id}_y_axis': Input(f"{component_id}_y_axis", "value"),
        f'{component_id}_sort': Input(f"{component_id}_sort", "value"),
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