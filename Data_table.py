from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import sys
import traceback
import os
from logger import logger

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS


def component():
    '''Return a component with an Ag-Grid table displaying filtered data'''
    layout = ddk.Card(
        id="data_table",
        children=[
            ddk.CardHeader(title="Data Table View", style={"color": "white"}),
            html.Div(
                id="data_table_summary",
                children=[],
                style={
                    "marginBottom": "20px",
                    "padding": "20px",
                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    "borderRadius": "10px"}
            ),
            dag.AgGrid(
                id="data_table_grid",
                columnDefs=[],
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 50,
                    "domLayout": "normal",
                    "rowSelection": "multiple",
                    "defaultColDef": {
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                        "floatingFilter": True
                    }
                },
                rowData=[]
            ),
            ddk.CardFooter(
                title="Full data table view with filtering, sorting, and pagination capabilities. Limited to a maximum of 10000 rows.",
                style={
                    "color": "white"})
        ],
        width=100,
        style={
            "background": "linear-gradient(135deg, #2c3e50 0%, #34495e 100%)",
            "color": "white"}
    )

    return {
        "layout": layout,
        "test_inputs": {}
    }


def _update_logic(**kwargs):
    '''Core data table update logic without error handling.'''
    logger.debug("Starting data table update. kwargs: %s", kwargs)

    df = filter_data(get_data(), **kwargs)

    logger.debug("Filtered data shape: %s", df.shape)

    if len(df) == 0:
        return [], []

    df = df.head(10_000)

    # Create column definitions
    column_defs = []
    important_columns = [
        "artists",
        "genres",
        "popularity",
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "tempo"]

    for col in df.columns:
        if col.startswith('_') or col == 'index':
            continue

        # Prioritize important columns
        if col in important_columns or len(column_defs) < 15:
            col_def = {
                "headerName": col.replace('_', ' ').title(),
                "field": col,
                "filter": True,
                "sortable": True
            }

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                col_def["valueFormatter"] = {
                    "function": "d3.timeFormat('%Y-%m-%d')(new Date(params.value))"}
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_def["type"] = "numericColumn"
                col_def["filter"] = "agNumberColumnFilter"
                if col in ["popularity", "danceability",
                           "energy", "valence", "acousticness"]:
                    col_def["valueFormatter"] = {
                        "function": "params.value?.toFixed(2)"}
            elif pd.api.types.is_string_dtype(df[col]):
                col_def["filter"] = "agTextColumnFilter"
                if col in ["artists", "genres"]:
                    col_def["width"] = 200

            column_defs.append(col_def)

    logger.debug("Created %d column definitions", len(column_defs))

    # Clean the data
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            df_cleaned[col] = df_cleaned[col].replace('', np.nan)

    row_data = df_cleaned.to_dict('records')

    for row in row_data:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None

    return column_defs, row_data


@callback(
    output=[
        Output("data_table_grid", "columnDefs"),
        Output("data_table_grid", "rowData")
    ],
    inputs=FILTER_CALLBACK_INPUTS
)
def update_table(**kwargs):
    '''Update the data table based on filters and controls'''
    try:
        return _update_logic(**kwargs)

    except Exception as e:
        logger.debug("Error updating data table: %s", str(e))
        print(f"Error updating data table: {e}, {traceback.format_exc()}")
        return [], []


@callback(
    output=Output("data_table_summary", "children"),
    inputs=FILTER_CALLBACK_INPUTS
)
def update_summary(**kwargs):
    '''Update the summary cards based on filters'''
    try:
        df = filter_data(get_data(), **kwargs)

        if len(df) == 0:
            return html.Div("No data available with current filters", style={
                            "textAlign": "center", "color": "#666"})

        # Create summary cards
        total_tracks = len(df)
        avg_popularity = df["popularity"].mean(
        ) if "popularity" in df.columns else 0

        summary_cards = html.Div([
            html.Div([
                html.H4(f"{total_tracks:,}", style={"margin": "0",
                        "color": "#333", "fontWeight": "bold"}),
                html.P(
                    "Total Tracks",
                    style={
                        "margin": "0",
                        "fontSize": "14px",
                        "color": "#333"})
            ], style={
                "textAlign": "center",
                "padding": "20px",
                "backgroundColor": "white",
                "border": "2px solid #e0e0e0",
                "borderRadius": "10px",
                "marginRight": "15px",
                "minWidth": "140px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
            }),
            html.Div([
                html.H4(f"{avg_popularity:.1f}", style={
                        "margin": "0", "color": "#333", "fontWeight": "bold"}),
                html.P(
                    "Average Popularity",
                    style={
                        "margin": "0",
                        "fontSize": "14px",
                        "color": "#333"})
            ], style={
                "textAlign": "center",
                "padding": "20px",
                "backgroundColor": "white",
                "border": "2px solid #e0e0e0",
                "borderRadius": "10px",
                "minWidth": "140px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
            })
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "15px", "justifyContent": "center"})

        return summary_cards

    except Exception as e:
        logger.debug("Error updating summary: %s", str(e))
        return html.Div("Error loading summary", style={
                        "textAlign": "center", "color": "#666"})
