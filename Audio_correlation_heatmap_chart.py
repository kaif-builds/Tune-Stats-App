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
from scipy.stats import pearsonr

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from logger import logger

class TestInput(TypedDict):
    options: list[Any]
    default: Any

class ComponentResponse(TypedDict):
    layout: ddk.Card
    test_inputs: dict[str, TestInput]

component_id = "audio_feature_correlation_matrix"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"
    
    features_checklist_id = f"{component_id}_features"
    significance_toggle_id = f"{component_id}_significance"
    
    # Audio features available for correlation analysis
    audio_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]
    
    features_options = [{"label": feature.title(), "value": feature} for feature in audio_features]
    features_default = audio_features  # All features selected by default
    
    title = "Audio Feature Correlation Matrix"
    description = "Correlation heatmap showing relationships between audio features with optional significance indicators"
    
    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            
            # Summary cards
            html.Div(
                id=f"{component_id}_summary_cards",
                style={"marginBottom": "20px"}
            ),
            
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Features to Include:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Checklist(
                                id=features_checklist_id,
                                options=features_options,
                                value=features_default,
                                inline=True,
                                style={"minWidth": "400px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Show Significance:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Checklist(
                                id=significance_toggle_id,
                                options=[{"label": "", "value": "enabled"}],
                                value=[],
                                style={"minWidth": "100px"}
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
        features_checklist_id: {
            "options": [audio_features[:3], audio_features[:5], audio_features],
            "default": features_default
        },
        significance_toggle_id: {
            "options": [[], ["enabled"]],
            "default": []
        }
    }
    
    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _create_summary_cards(df: pd.DataFrame) -> html.Div:
    """Create summary cards for the component."""
    if len(df) == 0:
        return html.Div()
    
    total_tracks = len(df)
    avg_popularity = df['popularity'].mean()
    
    card_style = {
        "backgroundColor": "#f8f9fa",
        "border": "1px solid #dee2e6",
        "borderRadius": "8px",
        "padding": "15px",
        "margin": "5px",
        "textAlign": "center",
        "minWidth": "150px"
    }
    
    return html.Div(
        style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "gap": "10px"},
        children=[
            html.Div(
                style=card_style,
                children=[
                    html.H4(f"{total_tracks:,}", style={"margin": "0", "color": "#495057"}),
                    html.P("Total Tracks", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#6c757d"})
                ]
            ),
            html.Div(
                style=card_style,
                children=[
                    html.H4(f"{avg_popularity:.1f}", style={"margin": "0", "color": "#495057"}),
                    html.P("Average Popularity", style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#6c757d"})
                ]
            ),
        ]
    )

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
                "font": {"size": 20}
            }]
        )
        return empty_fig, summary_cards
    
    # Extract control values
    selected_features = kwargs.get(f'{component_id}_features', [])
    show_significance = 'enabled' in kwargs.get(f'{component_id}_significance', [])
    
    if not selected_features or len(selected_features) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Select at least 2 features",
            annotations=[{
                "text": "Please select at least 2 audio features to display correlation matrix",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig, summary_cards
    
    logger.debug("Starting correlation matrix creation. Selected features: %s", selected_features)
    
    # Calculate correlation matrix
    feature_data = df[selected_features].copy()
    
    # Convert to numeric and handle any non-numeric values
    for col in feature_data.columns:
        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
    
    # Drop rows with any NaN values
    feature_data = feature_data.dropna()
    
    if len(feature_data) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No valid data",
            annotations=[{
                "text": "No valid numeric data available for selected features",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig, summary_cards
    
    # Calculate correlation matrix
    corr_matrix = feature_data.corr()
    
    # Calculate p-values for significance if requested
    p_values = None
    if show_significance:
        n_features = len(selected_features)
        p_values = np.ones((n_features, n_features))
        
        for i, feat1 in enumerate(selected_features):
            for j, feat2 in enumerate(selected_features):
                if i != j:
                    try:
                        _, p_val = pearsonr(feature_data[feat1], feature_data[feat2])
                        p_values[i, j] = p_val
                    except:
                        p_values[i, j] = 1.0
    
    # Create text annotations for the heatmap
    text_annotations = []
    for i, feat1 in enumerate(selected_features):
        row_text = []
        for j, feat2 in enumerate(selected_features):
            corr_val = corr_matrix.loc[feat1, feat2]
            text = f"{corr_val:.3f}"
            
            # Add significance indicators
            if show_significance and p_values is not None and i != j:
                p_val = p_values[i, j]
                if p_val < 0.001:
                    text += "***"
                elif p_val < 0.01:
                    text += "**"
                elif p_val < 0.05:
                    text += "*"
            
            row_text.append(text)
        text_annotations.append(row_text)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[feat.title() for feat in selected_features],
        y=[feat.title() for feat in selected_features],
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Correlation Coefficient",
            title_side="right"
        ),
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        xaxis_title="Audio Features",
        yaxis_title="Audio Features",
        height=max(400, len(selected_features) * 50),
        margin=dict(l=100, r=100, t=50, b=100)
    )
    
    # Add significance legend if showing significance
    if show_significance:
        fig.add_annotation(
            x=1.15,
            y=1,
            xref="paper",
            yref="paper",
            text="Significance:<br>*** p < 0.001<br>** p < 0.01<br>* p < 0.05",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    return fig, summary_cards

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_summary_cards", "children")
    ],
    inputs={
        f'{component_id}_features': Input(f"{component_id}_features", "value"),
        f'{component_id}_significance': Input(f"{component_id}_significance", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, html.Div]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )
    
    empty_summary = html.Div()
    
    try:
        figure, summary_cards = _update_logic(**kwargs)
        return figure, "", summary_cards
    
    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return empty_fig, error_msg, empty_summary