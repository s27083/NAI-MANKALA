"""
Zasady (Kalah): https://pl.wikipedia.org/wiki/Kalaha
Autorzy: Kamil Bogdański Adrian Kemski
Instrukcja środowiska: Python 3.10+, uruchom `python3 main.py`

AI: wykorzystuje bibliotekę easyAI (Negamax + alpha–beta).
Instalacja: `pip install easyAI`
"""

from typing import List

from game import (
    valid_moves,
    apply_move,
    score,
    store_index,
    side_empty,
)

try:
    from easyAI import TwoPlayerGame, AI_Player, Negamax
except ImportError as e:
    raise ImportError(
        "Brak biblioteki easyAI. Zainstaluj ją poleceniem: pip install easyAI"
    ) from e


class KalahEasyAI(TwoPlayerGame):
    def __init__(self, board: List[int], current_player: int, depth: int):
        self.board = board.copy()
        self.current_player = current_player
        self.players = [AI_Player(Negamax(depth)), AI_Player(Negamax(depth))]

    def possible_moves(self):
        return [str(i) for i in valid_moves(self.board, self.current_player)]

    def make_move(self, move):
        idx = int(move)
        self.board, self.current_player, _, _ = apply_move(
            self.board, self.current_player, idx
        )

    def is_over(self):
        return side_empty(self.board, 0) or side_empty(self.board, 1)

    def win(self):
        if not self.is_over():
            return False
        p0 = self.board[store_index(0)]
        p1 = self.board[store_index(1)]
        return (p0 > p1 and self.current_player == 0) or (
            p1 > p0 and self.current_player == 1
        )

    def scoring(self):
        return score(self.board, self.current_player)

    def show(self):
        pass


def choose_move(board: List[int], player: int, depth: int) -> int:
    game = KalahEasyAI(board, player, depth)
    ai_player = AI_Player(Negamax(depth))
    move = ai_player.ask_move(game)
    return int(move)