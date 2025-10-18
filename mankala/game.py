"""
Zasady (Kalah): https://pl.wikipedia.org/wiki/Kalahaa
Autorzy: Kamil Bogdański Adrian Kemski
Instrukcja środowiska: Python 3.10+, uruchom `python3 main.py`
"""

from typing import List, Tuple

NUM_PITS = 6
SEEDS_PER_PIT = 4


def store_index(player: int) -> int:
    return 6 if player == 0 else 13


def opponent(player: int) -> int:
    return 1 - player


def player_pit_range(player: int):
    return range(0, NUM_PITS) if player == 0 else range(7, 7 + NUM_PITS)


def is_pit(index: int) -> bool:
    return index in range(0, NUM_PITS) or index in range(7, 7 + NUM_PITS)


def opposite_pit(index: int) -> int:
    # Działa dla dołków 0..5 i 7..12: 0<->12, 1<->11, ..., 5<->7
    return 12 - index


def initial_state() -> List[int]:
    # Indeksy: 0..5 (pity gracza 0), 6 (skarbiec gracza 0), 7..12 (pity gracza 1), 13 (skarbiec gracza 1)
    return [SEEDS_PER_PIT] * NUM_PITS + [0] + [SEEDS_PER_PIT] * NUM_PITS + [0]


def side_empty(board: List[int], player: int) -> bool:
    return all(board[i] == 0 for i in player_pit_range(player))


def collect_remaining(board: List[int]) -> None:
    # Przenosi pozostałe kamyki do skarbców po zakończeniu gry
    for p in [0, 1]:
        s_idx = store_index(p)
        for i in player_pit_range(p):
            board[s_idx] += board[i]
            board[i] = 0


def valid_moves(board: List[int], player: int) -> List[int]:
    return [i for i in player_pit_range(player) if board[i] > 0]


def apply_move(board: List[int], player: int, pit_index: int) -> Tuple[List[int], int, bool, bool]:
    """
    Zwraca: (nowa_plansza, następny_gracz, extra_ruch, gra_skończona)
    """
    new_board = board.copy()
    seeds = new_board[pit_index]
    new_board[pit_index] = 0

    idx = pit_index
    while seeds > 0:
        idx = (idx + 1) % 14
        # Pomijamy skarbiec przeciwnika
        if idx == store_index(opponent(player)):
            continue
        new_board[idx] += 1
        seeds -= 1

    extra_turn = idx == store_index(player)

    # Przechwycenie, jeśli ostatni kamyk wylądował w pustym dołku gracza
    if is_pit(idx):
        if idx in player_pit_range(player) and new_board[idx] == 1:
            opp = opposite_pit(idx)
            if new_board[opp] > 0:
                new_board[store_index(player)] += new_board[opp] + new_board[idx]
                new_board[opp] = 0
                new_board[idx] = 0

    # Sprawdzenie końca gry
    game_over = side_empty(new_board, 0) or side_empty(new_board, 1)
    next_player = player if extra_turn else opponent(player)

    if game_over:
        collect_remaining(new_board)
        # Po zebraniu kamyków gra niewątpliwie kończy się
        next_player = opponent(player)  # bez znaczenia

    return new_board, next_player, extra_turn, game_over


def score(board: List[int], perspective_player: int) -> int:
    # Ocena: różnica skarbców z perspektywy gracza "perspective_player"
    return board[store_index(perspective_player)] - board[store_index(opponent(perspective_player))]