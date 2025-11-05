"""map.py
Stała, bardzo prosta mapa – labirynt w stylu "zygzak" z pionowymi słupkami
tworzącymi naprzemienne przejścia góra/dół. Zaprojektowane tak, aby wymuszać
kilka zakrętów i pokazać działanie fuzzy logic.

Układ współrzędnych: oś X w prawo, oś Y w górę (jak na wykresie Plotly),
jednostka to piksel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    """Iloczyn wektorowy 2D (a x b)."""
    return ax * by - ay * bx


def ray_segment_intersection(origin: Point, direction: Point, seg: Segment) -> float | None:
    """Zwraca odległość t>=0 wzdłuż promienia origin + t*direction do przecięcia z odcinkiem.

    direction może być dowolny wektor (nie musi być znormalizowany). Jeśli brak przecięcia,
    zwracane jest None.
    """
    (px, py) = origin
    (dx, dy) = direction
    (ax, ay), (bx, by) = seg
    sx, sy = (bx - ax), (by - ay)

    denom = _cross(dx, dy, sx, sy)
    if abs(denom) < 1e-9:
        return None  # równoległe lub niemal równoległe

    apx, apy = (ax - px), (ay - py)
    t = _cross(apx, apy, sx, sy) / denom
    u = _cross(apx, apy, dx, dy) / denom

    if t >= 0 and 0.0 <= u <= 1.0:
        return t
    return None


def distance_point_to_segment(p: Point, a: Point, b: Point) -> float:
    """Odległość punktu p od odcinka AB."""
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = (bx - ax), (by - ay)
    wx, wy = (px - ax), (py - ay)
    vv = vx * vx + vy * vy
    if vv == 0:
        # A i B takie same
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    cx, cy = (ax + t * vx), (ay + t * vy)
    return math.hypot(px - cx, py - cy)


def circle_segment_collision(center: Point, radius: float, seg: Segment) -> bool:
    """Sprawdza czy okrąg (pojazd) styka się lub przecina się z odcinkiem ściany."""
    (a, b) = seg
    return distance_point_to_segment(center, a, b) <= radius


@dataclass
class Map:
    width: int = 800
    height: int = 600

    def __post_init__(self):
        # Lista odcinków ścian
        self.walls: List[Segment] = []

        # Parametry prostego labiryntu
        m = 20               # margines od ramki
        gap = 140            # wysokość szczeliny (przejścia)
        bar_w = 30           # szerokość pionowego słupka
        xs = [160, 300, 440, 580]  # pozycje słupków

        # Pionowe przeszkody z naprzemiennym przejściem góra/dół
        bars: List[Tuple[float, float, float, float]] = []
        for i, x in enumerate(xs):
            if i % 2 == 0:
                # przejście na górze – słupek od y=m+gap do dołu
                bars.append((x, m + gap, x + bar_w, self.height - m))
            else:
                # przejście na dole – słupek od góry do y=height-m-gap
                bars.append((x, m, x + bar_w, self.height - m - gap))

        self.bars = bars

        # Cel w prawym dolnym rogu (po ostatnim słupku z przejściem na dole)
        self.goal_rect = (self.width - 140, self.height - 140, self.width - 60, self.height - 60)

        self._build_walls()

    @property
    def start_pose(self) -> Tuple[float, float, float]:
        """Zwraca start: (x, y, heading_rad)."""
        # Start w lewym górnym obszarze, skierowany w prawo
        return (60.0, 60.0, 0.0)

    def _add_rect_edges(self, x0: float, y0: float, x1: float, y1: float):
        """Dodaje krawędzie prostokąta jako odcinki."""
        a = (x0, y0)
        b = (x1, y0)
        c = (x1, y1)
        d = (x0, y1)
        self.walls += [(a, b), (b, c), (c, d), (d, a)]

    def _build_walls(self):
        # Ramka zewnętrzna
        self._add_rect_edges(20, 20, self.width - 20, self.height - 20)
        # Wewnętrzne bary
        for (x0, y0, x1, y1) in self.bars:
            self._add_rect_edges(x0, y0, x1, y1)

    def sensor_distance(self, origin: Point, direction: Point, max_range: float) -> float:
        """Oblicza najbliższe przecięcie promienia z dowolną ścianą w zasięgu.
        Zwraca odległość obciętą do max_range.
        """
        # Normalizujemy kierunek, aby t było w pikselach
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm == 0:
            return max_range
        dx, dy = dx / norm, dy / norm
        best = max_range
        for seg in self.walls:
            t = ray_segment_intersection(origin, (dx, dy), seg)
            if t is not None and t < best:
                best = t
        return min(best, max_range)

    def collides(self, center: Point, radius: float) -> bool:
        """Czy koło o danym promieniu koliduje z którąś ścianą."""
        for seg in self.walls:
            if circle_segment_collision(center, radius, seg):
                return True
        return False

    def goal_reached(self, center: Point, radius: float) -> bool:
        """Prosty test czy środek koła znajduje się w prostokącie celu."""
        x, y = center
        x0, y0, x1, y1 = self.goal_rect
        return (x0 + radius) <= x <= (x1 - radius) and (y0 + radius) <= y <= (y1 - radius)

    def plot_shapes(self):
        """Zwraca listę figur dla Plotly (rect) wizualizujących mapę."""
        shapes = []

        # Ramka
        shapes.append(
            dict(type="rect", x0=20, y0=20, x1=self.width - 20, y1=self.height - 20,
                 line=dict(color="black", width=6), fillcolor="rgba(0,0,0,0)")
        )

        # Bary – wypełnione na czarno
        for (x0, y0, x1, y1) in self.bars:
            shapes.append(
                dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                     line=dict(color="black", width=6), fillcolor="rgba(0,0,0,1)")
            )

        # Cel – czerwony prostokąt
        gx0, gy0, gx1, gy1 = self.goal_rect
        shapes.append(
            dict(type="rect", x0=gx0, y0=gy0, x1=gx1, y1=gy1,
                 line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.6)")
        )
        return shapes