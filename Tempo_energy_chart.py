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

component_id = "tempo_energy_density_plot"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    
    # Controls
    tempo_min_id = f"{component_id}_tempo_min"
    tempo_max_id = f"{component_id}_tempo_max"
    genre_overlay_id = f"{component_id}_genre_overlay"
    
    # Get data for initial setup
    df = get_data()
    
    # Tempo range controls (60-180 BPM default)
    tempo_min_default = 60
    tempo_max_default = 180
    
    # Genre overlay options
    genre_overlay_options = [
        {"label": "None", "value": "none"},
        {"label": "Show Genre Clusters", "value": "show"}
    ]
    genre_overlay_default = "none"

    title = "Tempo vs Energy Landscape"
    description = "2D density visualization showing the relationship between tempo and energy in music tracks with optional genre cluster overlay"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Tempo Range (BPM):", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block", "color": "white"}),
                            html.Div([
                                dcc.Input(
                                    id=tempo_min_id,
                                    type="number",
                                    value=tempo_min_default,
                                    min=0,
                                    max=220,
                                    debounce=True,
                                    style={"width": "80px", "color": "white", "backgroundColor": "#2c3e50", "border": "1px solid #34495e"}
                                ),
                                html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center", "color": "white"}),
                                dcc.Input(
                                    id=tempo_max_id,
                                    type="number",
                                    value=tempo_max_default,
                                    min=0,
                                    max=220,
                                    debounce=True,
                                    style={"width": "80px", "color": "white", "backgroundColor": "#2c3e50", "border": "1px solid #34495e"}
                                )
                            ], style={
                                "display": "flex",
                                "alignItems": "center",
                                "flexWrap": "wrap"
                            })
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Genre Overlay:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block", "color": "white"}),
                            dcc.Dropdown(
                                id=genre_overlay_id,
                                options=genre_overlay_options,
                                value=genre_overlay_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
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
        tempo_min_id: {
            "options": [60, 80, 100, 120],
            "default": tempo_min_default
        },
        tempo_max_id: {
            "options": [140, 160, 180, 200],
            "default": tempo_max_default
        },
        genre_overlay_id: {
            "options": [option["value"] for option in genre_overlay_options],
            "default": genre_overlay_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _create_summary_cards(df: pd.DataFrame) -> html.Div:
    """Create summary cards for the filtered data."""
    if len(df) == 0:
        return html.Div()
    
    total_tracks = len(df)
    avg_popularity = df['popularity'].mean()
    
    return html.Div([
        html.Div([
            html.H4("Total Tracks", style={"margin": "0", "fontSize": "14px", "color": "black", "fontWeight": "bold"}),
            html.H2(f"{total_tracks:,}", style={"margin": "5px 0 0 0", "fontSize": "24px", "color": "black"})
        ], style={
            "backgroundColor": "white",
            "padding": "15px",
            "borderRadius": "8px",
            "textAlign": "center",
            "minWidth": "120px",
            "border": "2px solid #e0e0e0"
        }),
        html.Div([
            html.H4("Average Popularity", style={"margin": "0", "fontSize": "14px", "color": "black", "fontWeight": "bold"}),
            html.H2(f"{avg_popularity:.1f}", style={"margin": "5px 0 0 0", "fontSize": "24px", "color": "black"})
        ], style={
            "backgroundColor": "white",
            "padding": "15px",
            "borderRadius": "8px",
            "textAlign": "center",
            "minWidth": "120px",
            "border": "2px solid #e0e0e0"
        })
    ], style={
        "display": "flex",
        "gap": "15px",
        "flexWrap": "wrap"
    })

def _update_logic(**kwargs) -> Tuple[go.Figure, html.Div]:
    """Core chart update logic without error handling."""
    df = filter_data(get_data(), **kwargs)
    
    # Create summary cards
    summary_cards = _create_summary_cards(df)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No data is available to display",
                "showarrow": False,
                "font": {"size": 20, "color": "white"}
            }],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={"color": "white"}
        )
        return empty_fig, summary_cards

    # Extract control values
    tempo_min = kwargs.get(f'{component_id}_tempo_min', 60)
    tempo_max = kwargs.get(f'{component_id}_tempo_max', 180)
    genre_overlay = kwargs.get(f'{component_id}_genre_overlay', 'none')
    
    # Handle None values
    if tempo_min is None:
        tempo_min = 60
    if tempo_max is None:
        tempo_max = 180
    
    # Convert to numeric and filter tempo range
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    
    # Remove rows with NaN values
    df = df.dropna(subset=['tempo', 'energy'])
    
    # Apply tempo filter
    tempo_min = float(tempo_min)
    tempo_max = float(tempo_max)
    df_filtered = df[(df['tempo'] >= tempo_min) & (df['tempo'] <= tempo_max)]
    
    if len(df_filtered) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data in selected tempo range",
            annotations=[{
                "text": f"No tracks found in tempo range {tempo_min}-{tempo_max} BPM",
                "showarrow": False,
                "font": {"size": 16, "color": "white"}
            }],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={"color": "white"}
        )
        return empty_fig, summary_cards

    logger.debug("Starting chart creation. df_filtered shape: %s", df_filtered.shape)
    
    # Create 2D density plot
    fig = px.density_heatmap(
        df_filtered,
        x='tempo',
        y='energy',
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale='Viridis'
    )
    
    # Add genre overlay if requested
    if genre_overlay == 'show':
        # Parse genres and get top genres for overlay
        all_genres = []
        for genres_str in df_filtered['genres']:
            if genres_str and genres_str != '[]':
                try:
                    genres_list = eval(genres_str)
                    if isinstance(genres_list, list):
                        all_genres.extend(genres_list)
                except:
                    continue
        
        if all_genres:
            # Get top 5 genres
            genre_counts = pd.Series(all_genres).value_counts().head(5)
            top_genres = genre_counts.index.tolist()
            
            # Create genre column for top genres
            def get_primary_genre(genres_str):
                if not genres_str or genres_str == '[]':
                    return 'Other'
                try:
                    genres_list = eval(genres_str)
                    if isinstance(genres_list, list) and len(genres_list) > 0:
                        for genre in genres_list:
                            if genre in top_genres:
                                return genre
                    return 'Other'
                except:
                    return 'Other'
            
            df_filtered['primary_genre'] = df_filtered['genres'].apply(get_primary_genre)
            
            # Add scatter points for genre clusters
            colors = px.colors.qualitative.Set3
            for i, genre in enumerate(top_genres):
                genre_data = df_filtered[df_filtered['primary_genre'] == genre]
                if len(genre_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=genre_data['tempo'],
                        y=genre_data['energy'],
                        mode='markers',
                        name=genre,
                        marker=dict(
                            size=4,
                            color=colors[i % len(colors)],
                            opacity=0.7
                        ),
                        hovertemplate=f"<b>{genre}</b><br>Tempo: %{{x:.1f}} BPM<br>Energy: %{{y:.3f}}<extra></extra>"
                    ))

    # Update layout with white text
    fig.update_layout(
        xaxis_title="Tempo (BPM)",
        yaxis_title="Energy",
        font={"color": "white", "size": 12},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title=dict(text="Track Density", font=dict(color="white")),
            tickfont={"color": "white"}
        )
    )
    
    # Update axes with white text
    fig.update_xaxes(
        title_font={"color": "white", "size": 14},
        tickfont={"color": "white"},
        gridcolor='rgba(255,255,255,0.2)'
    )
    fig.update_yaxes(
        title_font={"color": "white", "size": 14},
        tickfont={"color": "white"},
        gridcolor='rgba(255,255,255,0.2)'
    )

    return fig, summary_cards

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_summary_cards", "children")
    ],
    inputs={
        f'{component_id}_tempo_min': Input(f"{component_id}_tempo_min", "value"),
        f'{component_id}_tempo_max': Input(f"{component_id}_tempo_max", "value"),
        f'{component_id}_genre_overlay': Input(f"{component_id}_genre_overlay", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, html.Div]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20, "color": "white"}}],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={"color": "white"}
    )
    empty_summary = html.Div()

    try:
        figure, summary_cards = _update_logic(**kwargs)
        return figure, "", summary_cards

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg, empty_summary