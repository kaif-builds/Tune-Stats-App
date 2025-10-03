from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import numpy as np
import pandas as pd
import sys
import traceback
import os

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from logger import logger

def component():
    '''Return a component with data cards displaying key metrics'''
    layout = ddk.Row(
        id="data_cards",
        children=[
            ddk.DataCard(
                id='total_tracks_card',
                value='...',
                label='Total Tracks',
                width=33
            ),
            ddk.DataCard(
                id='avg_popularity_card',
                value='...',
                label='Average Popularity',
                width=33
            ),
            ddk.DataCard(
                id='avg_duration_card',
                value='...',
                label='Average Duration (minutes)',
                width=34
            )
        ])

    return {
        "layout": layout,
        "test_inputs": {}
    }

@callback(
    [
        Output('total_tracks_card', 'value'),
        Output('avg_popularity_card', 'value'),
        Output('avg_duration_card', 'value')
    ],
    FILTER_CALLBACK_INPUTS
)
def update(**kwargs):
    '''Update all data cards with the filtered data metrics'''
    try:
        logger.debug("Starting data cards update with kwargs: %s", kwargs)
        
        # Get data and apply filters
        df = filter_data(get_data(), **kwargs)
        
        logger.debug("Filtered data shape: %s", df.shape)
        
        # Check if dataframe is empty
        if len(df) == 0:
            logger.debug("No data after filtering")
            return ["No Data", "No Data", "No Data"]

        # Calculate metrics based on the filtered data
        total_tracks = len(df)
        avg_popularity = df["popularity"].mean()
        avg_duration_minutes = df["duration_ms"].mean() / 60000  # Convert from ms to minutes
        
        logger.debug("Calculated metrics - Total: %s, Avg Popularity: %.2f, Avg Duration: %.2f min", 
                    total_tracks, avg_popularity, avg_duration_minutes)

        # Format the results
        total_tracks_formatted = f"{total_tracks:,}"
        avg_popularity_formatted = f"{avg_popularity:.1f}"
        avg_duration_formatted = f"{avg_duration_minutes:.1f}"

        return [total_tracks_formatted, avg_popularity_formatted, avg_duration_formatted]
        
    except Exception as e:
        logger.debug("Error updating data cards: %s", str(e))
        print(f"Error updating data cards: {e}\n{traceback.format_exc()}")
        return ["Error", "Error", "Error"]