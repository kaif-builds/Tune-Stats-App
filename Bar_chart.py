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

component_id = "create_a_bar_chart_with"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    summary_id = f"{component_id}_summary"

    groupby_control_id = f"{component_id}_groupby"
    limit_control_id = f"{component_id}_limit"

    df = get_data()
    
    groupby_options = [
        {"label": "Artists", "value": "artists"},
        {"label": "Key", "value": "key"},
        {"label": "Mode", "value": "mode"}
    ]
    groupby_default = "artists"

    limit_options = [
        {"label": "Top 10", "value": 10},
        {"label": "Top 15", "value": 15},
        {"label": "Top 20", "value": 20},
        {"label": "Top 25", "value": 25}
    ]
    limit_default = 15

    title = "Aurora Music Popularity Dashboard"
    description = "Interactive bar chart showing music popularity with aurora-inspired gradients and summary statistics"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Group By:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=groupby_control_id,
                                options=groupby_options,
                                value=groupby_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Show Top:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=limit_control_id,
                                options=limit_options,
                                value=limit_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    )
                ],
            ),
            html.Div(id=summary_id, style={"marginBottom": "20px"}),
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
        groupby_control_id: {
            "options": [option["value"] for option in groupby_options],
            "default": groupby_default
        },
        limit_control_id: {
            "options": [option["value"] for option in limit_options],
            "default": limit_default
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
                "font": {"size": 20, "color": "white"}
            }],
            paper_bgcolor="rgba(0,0,0,0.9)",
            plot_bgcolor="rgba(0,0,0,0.9)"
        )
        return empty_fig, []

    groupby_value = kwargs.get(f'{component_id}_groupby', 'artists')
    limit_value = kwargs.get(f'{component_id}_limit', 15)
    
    if groupby_value is None:
        groupby_value = 'artists'
    if limit_value is None:
        limit_value = 15

    logger.debug("Starting chart creation. df shape: %s, groupby: %s, limit: %s", df.shape, groupby_value, limit_value)

    # Calculate summary statistics
    total_tracks = len(df)
    avg_popularity = df['popularity'].mean() if len(df) > 0 else 0

    # Group and aggregate data
    if groupby_value == 'key':
        key_names = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
                    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
        df['key_name'] = df['key'].map(key_names)
        grouped = df.groupby('key_name')['popularity'].mean().reset_index()
        grouped.columns = ['group', 'popularity']
    elif groupby_value == 'mode':
        mode_names = {0: 'Minor', 1: 'Major'}
        df['mode_name'] = df['mode'].map(mode_names)
        grouped = df.groupby('mode_name')['popularity'].mean().reset_index()
        grouped.columns = ['group', 'popularity']
    else:  # artists
        grouped = df.groupby('artists')['popularity'].mean().reset_index()
        grouped.columns = ['group', 'popularity']

    # Sort and limit
    grouped = grouped.sort_values('popularity', ascending=False).head(limit_value)

    # Create aurora-inspired gradient colors
    n_bars = len(grouped)
    colors = []
    for i in range(n_bars):
        # Create gradient from cyan to purple to green
        ratio = i / max(1, n_bars - 1)
        if ratio < 0.5:
            # Cyan to purple
            r = int(0 + (128 * ratio * 2))
            g = int(255 - (127 * ratio * 2))
            b = int(255 - (127 * ratio * 2))
        else:
            # Purple to green
            ratio_adj = (ratio - 0.5) * 2
            r = int(128 - (128 * ratio_adj))
            g = int(128 + (127 * ratio_adj))
            b = int(128 - (128 * ratio_adj))
        
        colors.append(f'rgba({r},{g},{b},0.8)')

    # Create the bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped['group'],
        y=grouped['popularity'],
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f'{val:.1f}' for val in grouped['popularity']],
        textposition='outside',
        textfont=dict(color='white', size=12),
        hovertemplate='<b>%{x}</b><br>Popularity: %{y:.1f}<extra></extra>'
    ))

    # Update layout with aurora theme
    fig.update_layout(
        paper_bgcolor='rgba(10,10,30,0.95)',
        plot_bgcolor='rgba(10,10,30,0.95)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Update axes with proper title formatting
    fig.update_xaxes(
        title=groupby_value.title(),
        tickfont=dict(color='white'),
        gridcolor='rgba(255,255,255,0.1)'
    )

    fig.update_yaxes(
        title='Popularity',
        tickfont=dict(color='white'),
        gridcolor='rgba(255,255,255,0.1)'
    )

    # Create summary cards
    summary_cards = [
        html.Div(
            children=[
                html.H3(f"{total_tracks:,}", style={"margin": "0", "fontSize": "24px", "fontWeight": "bold", "color": "black"}),
                html.P("Total Tracks", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "black"})
            ],
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
                "textAlign": "center",
                "minWidth": "150px",
                "marginRight": "15px",
                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
            }
        ),
        html.Div(
            children=[
                html.H3(f"{avg_popularity:.1f}", style={"margin": "0", "fontSize": "24px", "fontWeight": "bold", "color": "black"}),
                html.P("Average Popularity", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "black"})
            ],
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
                "textAlign": "center",
                "minWidth": "150px",
                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
            }
        )
    ]

    return fig, summary_cards

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_summary", "children")
    ],
    inputs={
        f'{component_id}_groupby': Input(f"{component_id}_groupby", "value"),
        f'{component_id}_limit': Input(f"{component_id}_limit", "value"),
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
        
        # Wrap summary cards in horizontal container
        summary_container = html.Div(
            children=summary_cards,
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "wrap",
                "gap": "15px",
                "justifyContent": "flex-start"
            }
        )
        
        return figure, "", summary_container

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg, []