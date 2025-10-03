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

component_id = "hit_formula_scatter_plot"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    summary_id = f"{component_id}_summary"

    y_axis_id = f"{component_id}_y_axis"
    y_axis_options = [
        {"label": "Danceability", "value": "danceability"},
        {"label": "Energy", "value": "energy"},
        {"label": "Valence", "value": "valence"},
        {"label": "Acousticness", "value": "acousticness"},
        {"label": "Loudness", "value": "loudness"}
    ]
    y_axis_default = "danceability"

    title = "Hit Formula Explorer"
    description = "Explore the relationship between popularity and audio features, with color coding by genre to identify patterns in successful tracks."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Y-Axis Feature:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=y_axis_id,
                                options=y_axis_options,
                                value=y_axis_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(id=summary_id, style={"marginBottom": "15px"}),
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
        y_axis_id: {
            "options": [option["value"] for option in y_axis_options],
            "default": y_axis_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _update_logic(**kwargs) -> Tuple[go.Figure, Any]:
    df = filter_data(get_data(), **kwargs)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No data available",
                "showarrow": False,
                "font": {"size": 20, "color": "white"}
            }],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return empty_fig, html.Div()

    y_axis_feature = kwargs.get(f'{component_id}_y_axis', 'danceability')
    if y_axis_feature is None:
        y_axis_feature = 'danceability'

    logger.debug("Starting chart creation. df shape: %s, y_axis: %s", df.shape, y_axis_feature)

    df_clean = df.dropna(subset=['popularity', y_axis_feature, 'genres'])
    
    df_clean['genre_display'] = df_clean['genres'].apply(lambda x: 
        eval(x)[0] if isinstance(x, str) and x != '[]' and len(eval(x)) > 0 else 'No Genre'
    )
    
    genre_counts = df_clean['genre_display'].value_counts()
    top_genres = genre_counts.head(10).index.tolist()
    df_clean['genre_for_plot'] = df_clean['genre_display'].apply(
        lambda x: x if x in top_genres else 'Other'
    )

    total_tracks = len(df_clean)
    avg_popularity = df_clean['popularity'].mean()
    unique_artists = df_clean['artists'].nunique()

    summary_cards = html.Div(
        style={"display": "flex", "flexDirection": "row", "gap": "15px", "flexWrap": "wrap"},
        children=[
            html.Div(
                children=[
                    html.Div("Total Tracks", style={"fontSize": "14px", "color": "#333", "marginBottom": "5px"}),
                    html.Div(f"{total_tracks:,}", style={"fontSize": "24px", "color": "#333"})
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "minWidth": "120px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                }
            ),
            html.Div(
                children=[
                    html.Div("Average Popularity", style={"fontSize": "14px", "color": "#333", "marginBottom": "5px"}),
                    html.Div(f"{avg_popularity:.1f}", style={"fontSize": "24px", "color": "#333"})
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "minWidth": "120px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                }
            )
        ]
    )

    feature_labels = {
        'danceability': 'Danceability',
        'energy': 'Energy',
        'valence': 'Valence',
        'acousticness': 'Acousticness',
        'loudness': 'Loudness (dB)'
    }

    fig = px.scatter(
        df_clean,
        x='popularity',
        y=y_axis_feature,
        color='genre_for_plot',
        hover_data=['artists'],
        opacity=0.7
    )

    fig.update_layout(
        xaxis_title="Popularity",
        yaxis_title=feature_labels.get(y_axis_feature, y_axis_feature.title()),
        legend_title="Genre",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.2)", title_font_color="white")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.2)", title_font_color="white")

    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                     "Popularity: %{x}<br>" +
                     f"{feature_labels.get(y_axis_feature, y_axis_feature.title())}: %{{y}}<br>" +
                     "Genre: %{fullData.name}<extra></extra>"
    )

    return fig, summary_cards

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_summary", "children"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        f'{component_id}_y_axis': Input(f"{component_id}_y_axis", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, Any, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20, "color": "white"}}],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    try:
        figure, summary = _update_logic(**kwargs)
        return figure, summary, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, html.Div(), error_msg