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

component_id = "design_a_futuristic_histogram"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    summary_id = f"{component_id}_summary"

    metric_control_id = f"{component_id}_metric"
    metric_options = [
        {"label": "Popularity", "value": "popularity"},
        {"label": "Danceability", "value": "danceability"},
        {"label": "Energy", "value": "energy"},
        {"label": "Valence", "value": "valence"},
        {"label": "Acousticness", "value": "acousticness"},
        {"label": "Tempo", "value": "tempo"}
    ]
    metric_default = "popularity"

    bins_control_id = f"{component_id}_bins"
    bins_min = 10
    bins_max = 50
    bins_default = 20
    bins_marks = {
        int(10): "10",
        int(20): "20", 
        int(30): "30",
        int(40): "40",
        int(50): "50"
    }

    title = "Futuristic Music Data Distribution"
    description = "A futuristic histogram with glowing neon bars representing music data distribution. Features gradient colors from cyan to magenta with a dark theme for an immersive data visualization experience."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                id=summary_id,
                style={"marginBottom": "20px"}
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Metric:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=metric_control_id,
                                options=metric_options,
                                value=metric_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Number of Bins:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            html.Div(
                                children=dcc.Slider(
                                    id=bins_control_id,
                                    min=bins_min,
                                    max=bins_max,
                                    step=5,
                                    value=bins_default,
                                    marks=bins_marks,
                                    tooltip={"placement": "bottom"}
                                ),
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "width": "300px"}
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
        metric_control_id: {
            "options": [option["value"] for option in metric_options],
            "default": metric_default
        },
        bins_control_id: {
            "options": list(bins_marks.keys()),
            "default": bins_default
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
                "font": {"size": 20, "color": "white"}
            }],
            paper_bgcolor="black",
            plot_bgcolor="black"
        )
        return empty_fig

    metric = kwargs.get(f'{component_id}_metric', 'popularity')
    if metric is None:
        metric = 'popularity'
    
    bins = kwargs.get(f'{component_id}_bins', 20)
    if bins is None:
        bins = 20

    logger.debug("Starting chart creation. df:\n%s", df.head())
    logger.debug("Selected metric: %s, bins: %s", metric, bins)

    # Convert to numeric and handle any conversion issues
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df.dropna(subset=[metric])

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No valid data for selected metric",
            annotations=[{
                "text": "No valid data available for the selected metric",
                "showarrow": False,
                "font": {"size": 20, "color": "white"}
            }],
            paper_bgcolor="black",
            plot_bgcolor="black"
        )
        return empty_fig

    # Create histogram data
    counts, bin_edges = np.histogram(df[metric], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Create gradient colors from cyan to purple to magenta
    n_bars = len(counts)
    colors = []
    for i in range(n_bars):
        ratio = i / max(1, n_bars - 1)
        if ratio <= 0.5:
            # Cyan to purple
            r = int(0 + (128 * ratio * 2))
            g = int(255 - (255 * ratio * 2))
            b = int(255)
        else:
            # Purple to magenta
            ratio_adj = (ratio - 0.5) * 2
            r = int(128 + (127 * ratio_adj))
            g = int(0)
            b = int(255 - (255 * ratio_adj))
        
        colors.append(f'rgba({r}, {g}, {b}, 0.8)')

    # Create the futuristic histogram
    fig = go.Figure()

    # Add glowing bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=bin_width * 0.8,
        marker=dict(
            color=colors,
            line=dict(
                color='rgba(255, 255, 255, 0.3)',
                width=1
            )
        ),
        name="Distribution",
        hovertemplate=f"<b>{metric.title()}</b><br>Range: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
    ))

    # Add glow effect with semi-transparent bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=bin_width * 1.2,
        marker=dict(
            color=[color.replace('0.8', '0.2') for color in colors],
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Update layout for futuristic dark theme
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(
            title=f"{metric.title()}",
            gridcolor='rgba(128, 128, 128, 0.3)',
            color='white',
            tickcolor='white'
        ),
        yaxis=dict(
            title="Frequency",
            gridcolor='rgba(128, 128, 128, 0.3)',
            color='white',
            tickcolor='white'
        ),
        bargap=0.1,
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60)
    )

    return fig

@callback(
    output=[
        Output(f"{component_id}_summary", "children"),
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        f'{component_id}_metric': Input(f"{component_id}_metric", "value"),
        f'{component_id}_bins': Input(f"{component_id}_bins", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[html.Div, go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20, "color": "white"}}],
        paper_bgcolor="black",
        plot_bgcolor="black"
    )

    try:
        df = filter_data(get_data(), **kwargs)
        
        # Create summary cards
        if len(df) > 0:
            total_tracks = len(df)
            avg_popularity = df['popularity'].mean()
            
            summary_cards = html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
                children=[
                    html.Div(
                        children=[
                            html.H4("Total Tracks", style={"margin": "0", "color": "#333", "fontSize": "14px"}),
                            html.H2(f"{total_tracks:,}", style={"margin": "5px 0 0 0", "color": "#333", "fontSize": "24px"})
                        ],
                        style={
                            "backgroundColor": "white",
                            "padding": "15px",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "minWidth": "150px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                        }
                    ),
                    html.Div(
                        children=[
                            html.H4("Average Popularity", style={"margin": "0", "color": "#333", "fontSize": "14px"}),
                            html.H2(f"{avg_popularity:.1f}", style={"margin": "5px 0 0 0", "color": "#333", "fontSize": "24px"})
                        ],
                        style={
                            "backgroundColor": "white",
                            "padding": "15px",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "minWidth": "150px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                        }
                    )
                ]
            )
        else:
            summary_cards = html.Div()

        figure = _update_logic(**kwargs)
        return summary_cards, figure, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return html.Div(), empty_fig, error_msg