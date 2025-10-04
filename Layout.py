import dash_design_kit as ddk
from dash import html
import importlib
import traceback

# --- 1. Component Import Section ---
# This section attempts to import every component listed in your new specification.
# It uses the exact same try/except pattern as your previous working app.

component_registry = {}  # Map of component names to their instances
failed_components = {}  # Map of component names to their error messages

# Define the list of all components to load for the new "Tune Stats" app
COMPONENTS_TO_LOAD = [
    "filter_component",
    "data_cards",
    "data_table",
    "hit_formula_scatter_plot",
    "genre_sonic_fingerprint_heatmap",
    "artist_spotlight_radar_chart",
    "futuristic_histogram",
    "audio_feature_correlation_matrix",
    "tempo_energy_density_plot",
    "popularity_landscapes",
    "loudness_landscape",
    "audio_feature_distribution",
    "genre_popularity_comparison",
    "side_by_side_popularity",
    "musical_key_popularity",
    "artist_dna_comparison",
    "bar_chart",
    "dark_themed_visualization",
    "genre_evolution_sparklines"
]

# Dynamically import each component and store it or log an error
for name in COMPONENTS_TO_LOAD:
    try:
        module = importlib.import_module(f"components.{name}")
        component_registry[name] = module.component
        print(f"[SUCCESS] Imported component: {name}")
    except Exception as e:
        failed_components[name] = traceback.format_exc()
        print(f"[FAILURE] Failed to import {name}: {e}")


# --- 2. Main Layout Definition ---
# This function builds the visual layout of your entire application.
def layout(preview=False):
    
    layout_items = []

    # --- Hero/Header Section ---
    # Based on your new "Tune Stats" specification
    app_title = "Tune Stats: Visualizing Music DNA and Hit Features"
    app_description = "Interactive app visualizing Spotify track data, exploring audio features, genres, artists, and music trends"
    app_tags = [
        ddk.Tag(text="Data Updated: 2025-09-25", icon="calendar"),
        ddk.Tag(text="Created by: Plotly Studio", icon="user"),
        ddk.Tag(text="Data Source: Tune Stats dataset", icon="database"),
    ]

    layout_items.append(
        ddk.Hero(title=app_title, description=app_description, children=app_tags)
    )

    # --- Add Core Components (Filters, Cards, Table) ---
    if 'filter_component' in component_registry:
        layout_items.append(component_registry['filter_component']()['layout'])
    if 'data_cards' in component_registry:
        layout_items.append(component_registry['data_cards']()['layout'])

    # --- Dynamically Add All Chart Components ---
    # This loop will add every successfully loaded chart to the layout
    chart_layouts = []
    for name, component_func in component_registry.items():
        # Exclude the non-chart components we've already added
        if name not in ['filter_component', 'data_cards', 'data_table']:
            try:
                chart_card = component_func()['layout']
                chart_card.width = 50 # Set to 50% for a two-column layout
                if preview:
                    # Add an edit button in preview mode, just like the old code
                    chart_card.children[0].children = [
                        html.Button(
                            children=[ddk.Icon(icon_name="pencil"), "Edit"],
                            id={"type": "edit-component-button", "index": name},
                            style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": 1}
                        )
                    ]
                chart_layouts.append(chart_card)
            except Exception as e:
                # If a component's layout function fails, show an error card
                error_card = ddk.Card(width=50, children=[
                    ddk.CardHeader(title=f'Error loading layout for: {name}', style={"color": "red"}),
                    html.Pre(str(e))
                ])
                chart_layouts.append(error_card)

    layout_items.append(ddk.Row(children=chart_layouts))

    # Add the data table at the very end
    if 'data_table' in component_registry:
        layout_items.append(component_registry['data_table']()['layout'])
    
    # --- Display Errors for Any Failed Imports ---
    if failed_components:
        error_messages = [html.H4("The following components failed to import:")]
        for name, error in failed_components.items():
            error_messages.append(html.Details([
                html.Summary(f"Error in: {name}"),
                html.Pre(error, style={'color': 'red', 'whiteSpace': 'pre-wrap'})
            ]))
        layout_items.append(ddk.Card(children=error_messages, width=100))
    
    # --- Return the final app structure ---
    return layout_items