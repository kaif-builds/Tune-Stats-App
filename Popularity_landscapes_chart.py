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

component_id = "popularity_landscapes_by"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    
    # Get data to determine available genres
    df = get_data()
    
    # Parse genres column to get unique genres
    all_genres = []
    for genres_str in df['genres'].dropna():
        if genres_str != '[]':
            # Remove brackets and quotes, split by comma
            genres_clean = genres_str.strip('[]').replace("'", "").replace('"', '')
            if genres_clean:
                genres_list = [g.strip() for g in genres_clean.split(',')]
                all_genres.extend(genres_list)
    
    unique_genres = sorted(list(set(all_genres)))
    
    # Limit to top genres by frequency for better visualization
    genre_counts = pd.Series(all_genres).value_counts()
    top_genres = genre_counts.head(15).index.tolist()
    
    # Control for genre selection
    genre_control_id = f"{component_id}_genre_selection"
    genre_options = [{"label": "Top 15 Genres", "value": "top_15"}] + [{"label": genre, "value": genre} for genre in top_genres]
    genre_default = "top_15"
    
    # Control for number of genres to show
    num_genres_id = f"{component_id}_num_genres"
    
    title = "Popularity Landscapes by Genre"
    description = "Ridgeline plot showing popularity score distributions across different music genres. Each genre displays as an overlapping density curve, creating a visual landscape for easy comparison of popularity patterns."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Genre Selection:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=genre_control_id,
                                options=genre_options,
                                value=genre_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Max Genres:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            html.Div(
                                children=dcc.Slider(
                                    id=num_genres_id,
                                    min=5,
                                    max=15,
                                    step=1,
                                    value=10,
                                    marks={
                                        int(5): "5",
                                        int(8): "8", 
                                        int(10): "10",
                                        int(12): "12",
                                        int(15): "15"
                                    },
                                    tooltip={"placement": "bottom"}
                                ),
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "width": "300px"}
                    ),
                ],
            ),
            # Summary cards
            html.Div(
                id=f"{component_id}_summary_cards",
                style={"marginBottom": "15px"}
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
        genre_control_id: {
            "options": [option["value"] for option in genre_options],
            "default": genre_default
        },
        num_genres_id: {
            "options": [5, 8, 10, 12, 15],
            "default": 10
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _create_summary_cards(df_expanded):
    """Create summary cards for the data."""
    total_tracks = len(df_expanded)
    avg_popularity = df_expanded['popularity'].mean()
    unique_artists = df_expanded['artists'].nunique()
    
    return html.Div(
        style={"display": "flex", "gap": "15px", "flexWrap": "wrap"},
        children=[
            html.Div(
                style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "minWidth": "150px"
                },
                children=[
                    html.H4(f"{total_tracks:,}", style={"margin": "0", "color": "#2c3e50"}),
                    html.P("Total Tracks", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#7f8c8d"})
                ]
            ),
            html.Div(
                style={
                    "backgroundColor": "#f8f9fa", 
                    "padding": "15px",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "minWidth": "150px"
                },
                children=[
                    html.H4(f"{avg_popularity:.1f}", style={"margin": "0", "color": "#2c3e50"}),
                    html.P("Avg Popularity", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#7f8c8d"})
                ]
            ),
            html.Div(
                style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px", 
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "minWidth": "150px"
                },
                children=[
                    html.H4(f"{unique_artists:,}", style={"margin": "0", "color": "#2c3e50"}),
                    html.P("Unique Artists", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#7f8c8d"})
                ]
            )
        ]
    )

def _update_logic(**kwargs) -> Tuple[go.Figure, Any]:
    """Core chart update logic without error handling."""
    df = filter_data(get_data(), **kwargs)
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No data available after filtering",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, html.Div()

    # Extract control values
    genre_selection = kwargs.get(f'{component_id}_genre_selection', 'top_15')
    max_genres = kwargs.get(f'{component_id}_num_genres', 10)
    
    if genre_selection is None:
        genre_selection = 'top_15'
    if max_genres is None:
        max_genres = 10

    # Parse genres and expand data
    expanded_data = []
    for idx, row in df.iterrows():
        genres_str = row['genres']
        if genres_str != '[]' and pd.notna(genres_str):
            genres_clean = genres_str.strip('[]').replace("'", "").replace('"', '')
            if genres_clean:
                genres_list = [g.strip() for g in genres_clean.split(',')]
                for genre in genres_list:
                    if genre:
                        expanded_data.append({
                            'genre': genre,
                            'popularity': row['popularity'],
                            'artists': row['artists']
                        })
    
    if not expanded_data:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No genre data available",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, html.Div()
    
    df_expanded = pd.DataFrame(expanded_data)
    
    # Convert popularity to numeric
    df_expanded['popularity'] = pd.to_numeric(df_expanded['popularity'], errors='coerce')
    df_expanded = df_expanded.dropna(subset=['popularity'])
    
    if len(df_expanded) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No valid popularity data available",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, html.Div()

    # Filter genres based on selection
    if genre_selection == 'top_15':
        genre_counts = df_expanded['genre'].value_counts()
        selected_genres = genre_counts.head(int(max_genres)).index.tolist()
    else:
        selected_genres = [genre_selection]
    
    df_filtered = df_expanded[df_expanded['genre'].isin(selected_genres)]
    
    if len(df_filtered) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No data for selected genres",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, html.Div()

    logger.debug("Starting ridgeline chart creation. df_filtered shape: %s", df_filtered.shape)
    
    # Create ridgeline plot using violin plots
    fig = go.Figure()
    
    # Sort genres by median popularity for better visualization
    genre_medians = df_filtered.groupby('genre')['popularity'].median().sort_values(ascending=False)
    sorted_genres = genre_medians.index.tolist()
    
    # Create violin plot for each genre
    for i, genre in enumerate(sorted_genres):
        genre_data = df_filtered[df_filtered['genre'] == genre]['popularity']
        
        if len(genre_data) > 1:  # Need at least 2 points for violin plot
            fig.add_trace(go.Violin(
                y=[genre] * len(genre_data),
                x=genre_data,
                name=genre,
                orientation='h',
                side='positive',
                width=0.8,
                points=False,
                meanline_visible=True,
                showlegend=False,
                hovertemplate=f"<b>{genre}</b><br>Popularity: %{{x}}<br>Count: {len(genre_data)}<extra></extra>"
            ))
    
    # Update layout for ridgeline appearance
    fig.update_layout(
        xaxis_title="Popularity Score",
        yaxis_title="Genre",
        height=max(400, len(sorted_genres) * 40),
        margin=dict(l=150, r=50, t=50, b=50),
        yaxis=dict(
            categoryorder='array',
            categoryarray=sorted_genres
        )
    )
    
    # Create summary cards
    summary_cards = _create_summary_cards(df_filtered)
    
    return fig, summary_cards

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_summary_cards", "children")
    ],
    inputs={
        f'{component_id}_genre_selection': Input(f"{component_id}_genre_selection", "value"),
        f'{component_id}_num_genres': Input(f"{component_id}_num_genres", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, Any]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure, summary_cards = _update_logic(**kwargs)
        return figure, "", summary_cards

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg, html.Div()