"""vehicle.py
Implementacja prostego pojazdu (koło) poruszającego się w 2D
z czujnikami odległości (promienie) i kontrolą fuzzy.

Pojazd porusza się kinematycznie: x += v*dt*cos(theta), y += v*dt*sin(theta)
Skręt realizowany jako zmiana kąta: theta += turn_rate*dt,
gdzie turn_rate powstaje ze znormalizowanego wyjścia fuzzy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

from map import Map
from fuzzy_logic import FuzzyController


Point = Tuple[float, float]


@dataclass
class Vehicle:
    radius: float = 12.0
    base_speed: float = 80.0  # piksele/s – regulowana suwakiem w UI
    sensor_range: float = 220.0
    max_turn_rate: float = 1.3  # rad/s dla turn_norm=1

    def __post_init__(self):
        # Czujniki: kąty względem osi pojazdu (radiany)
        self.sensor_angles = [-math.radians(45), 0.0, math.radians(45)]  # left, front, right
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def reset_to_start(self, world: Map):
        self.x, self.y, self.theta = world.start_pose

    # ====== Percepcja ======
    def sense(self, world: Map) -> Tuple[float, float, float, List[Tuple[Point, Point]]]:
        """Zwraca (d_left, d_front, d_right, segmenty_promieni_do_rysowania)."""
        rays = []
        dists = []
        for ang in self.sensor_angles:
            a = self.theta + ang
            dx, dy = math.cos(a), math.sin(a)
            d = world.sensor_distance((self.x, self.y), (dx, dy), self.sensor_range)
            dists.append(d)
            rays.append(((self.x, self.y), (self.x + dx * d, self.y + dy * d)))
        return dists[0], dists[1], dists[2], rays

    # ====== Aktualizacja stanu ======
    def step(self, dt: float, world: Map, ctrl: FuzzyController) -> dict:
        """Wykonuje jeden krok symulacji. Zwraca dane debugowe.

        - dt: krok czasu [s]
        - world: mapa do kolizji i czujników
        - ctrl: kontroler fuzzy
        """
        # Pomiar czujników
        d_left, d_front, d_right, rays = self.sense(world)

        # Decyzja fuzzy: skręt i modyfikator prędkości
        turn_norm, speed_factor, debug = ctrl.compute(d_left, d_front, d_right)
        turn_rate = self.max_turn_rate * turn_norm
        speed = self.base_speed * speed_factor

        # Próba ruchu
        new_theta = self.theta + turn_rate * dt
        nx = self.x + speed * dt * math.cos(new_theta)
        ny = self.y + speed * dt * math.sin(new_theta)

        # Detekcja kolizji – jeśli kolizja, nie wykonuj ruchu i lekko skręć w stronę od większej odległości
        if not world.collides((nx, ny), self.radius):
            self.x, self.y, self.theta = nx, ny, new_theta
        else:
            # Minimalna zmiana kierunku aby wyjść ze styku
            if d_left < d_right:
                self.theta += 0.5 * dt
            else:
                self.theta -= 0.5 * dt

        debug.update({
            "d_left": d_left,
            "d_front": d_front,
            "d_right": d_right,
            "rays": rays,
            "speed": speed,
            "turn_rate": turn_rate,
            "position": (self.x, self.y),
            "theta": self.theta,
        })
        return debug

    # ====== Pomocnicze do rysowania ======
    def plot_shapes(self) -> List[dict]:
        """Zwraca figurę koła pojazdu (do layout.shapes)."""
        r = self.radius
        return [
            dict(type="circle", x0=self.x - r, y0=self.y - r, x1=self.x + r, y1=self.y + r,
                 line=dict(color="rgba(120,120,255,1)", width=2), fillcolor="rgba(120,120,255,0.4)")
        ]