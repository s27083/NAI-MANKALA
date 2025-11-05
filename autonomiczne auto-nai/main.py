"""main.py
Aplikacja Dash wizualizująca stałą mapę, pojazd i działanie logiki rozmytej.

UI:
- Graf środowiska (mapa + pozycja pojazdu + promienie czujników)
- Suwak prędkości bazowej pojazdu
- Bieżące dane (pozycja, prędkość)
- Wykres krzywych przynależności odległości oraz aktualnego pomiaru z przodu

Uruchomienie:
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

from map import Map
from vehicle import Vehicle
from fuzzy_logic import FuzzyController


# ====== Inicjalizacja świata ======
world = Map()
car = Vehicle()
car.reset_to_start(world)
ctrl = FuzzyController(max_range=car.sensor_range)


app = Dash(__name__)
app.title = "Autonomiczne auto – Fuzzy Logic"


def make_env_figure(debug: dict) -> go.Figure:
    """Tworzy wykres środowiska z mapą i pojazdem."""
    fig = go.Figure()

    # Promienie czujników
    rays: List[tuple] = debug.get("rays", [])
    for (p0, p1) in rays:
        fig.add_trace(go.Scatter(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]],
            mode="lines", line=dict(color="orange", width=2)
        ))

    # Punkt środka pojazdu (dla czytelności)
    px, py = debug.get("position", (car.x, car.y))
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers",
                             marker=dict(size=8, color="orange")))

    # Figury: mapa + pojazd
    shapes = []
    shapes.extend(world.plot_shapes())
    shapes.extend(car.plot_shapes())
    fig.update_layout(
        width=900, height=700,
        xaxis=dict(range=[0, world.width], visible=False),
        yaxis=dict(range=[0, world.height], visible=False, scaleanchor="x", scaleratio=1),
        shapes=shapes,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
    )
    return fig


def make_membership_figure(front_d: float) -> go.Figure:
    """Wykres krzywych przynależności dla odległości + pionowa linia frontu."""
    curves = ctrl.distance_membership_curves(200)
    x = curves["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=curves["near"], name="Near", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=x, y=curves["mid"], name="Mid", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=x, y=curves["far"], name="Far", line=dict(color="#2ca02c")))
    # Aktualny pomiar z przodu
    fig.add_trace(go.Scatter(x=[front_d, front_d], y=[0, 1], mode="lines",
                             name="Front distance", line=dict(color="black", dash="dash")))
    fig.update_layout(
        width=900, height=300,
        xaxis=dict(range=[0, car.sensor_range], title="Odległość [px]"),
        yaxis=dict(range=[0, 1.05], title="Przynależność"),
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h"),
    )
    return fig


app.layout = html.Div([
    html.H3("Autonomiczne auto – logika rozmyta"),
    html.Div([
        html.Div([
            dcc.Graph(id="env", animate=False),
            dcc.Interval(id="tick", interval=50, n_intervals=0),
        ], style={"flex": "1"}),
    ], style={"display": "flex"}),

    html.Div([
        html.Div([
            html.Label("Prędkość bazowa [px/s]"),
            dcc.Slider(id="speed", min=40, max=160, step=2, value=80,
                       marks=None, tooltip={"placement": "bottom"}),
            html.Div(id="speed_text", style={"marginTop": "6px"}),
            html.Div(id="pos_text", style={"marginTop": "6px"}),
        ], style={"width": "280px", "marginRight": "30px"}),

        html.Div([
            dcc.Graph(id="membership", animate=False),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "alignItems": "flex-start", "marginTop": "16px"}),
])


# ====== Callbacks ======
@app.callback(
    Output("env", "figure"),
    Output("membership", "figure"),
    Output("speed_text", "children"),
    Output("pos_text", "children"),
    Input("tick", "n_intervals"),
    Input("speed", "value"),
)
def on_tick(n: int, speed_value: float):
    # Aktualizacja prędkości bazowej z suwaka
    car.base_speed = float(speed_value)
    dt = 0.05
    debug = car.step(dt, world, ctrl)

    env_fig = make_env_figure(debug)
    mem_fig = make_membership_figure(debug.get("d_front", 0.0))

    speed_info = f"Prędkość efektywna: {debug['speed']:.1f} px/s (bazowa: {car.base_speed:.1f})"
    pos_info = f"Pozycja: ({debug['position'][0]:.1f}, {debug['position'][1]:.1f}), kąt: {math.degrees(debug['theta']):.1f}°"

    # Reset jeśli osiągnęliśmy cel (dla demonstracji)
    if world.goal_reached(debug["position"], car.radius):
        car.reset_to_start(world)

    return env_fig, mem_fig, speed_info, pos_info


if __name__ == "__main__":
    app.run_server(debug=False, port=8050)