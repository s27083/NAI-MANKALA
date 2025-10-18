"""
Zasady (Kalah): https://pl.wikipedia.org/wiki/Kalaha
Autorzy: Kamil Bogdański Adrian Kemski
Instrukcja przygotowania środowiska: Python 3.10+, uruchom `python3 main.py`
"""

import argparse
from typing import List

import game
import ai


def format_board(b: List[int]) -> str:
    # Reprezentacja:
    #   [13]  12 11 10  9  8  7
    #         0   1  2  3  4  5   [6]
    top = " ".join(f"{b[i]:2d}" for i in range(12, 6, -1))
    bottom = " ".join(f"{b[i]:2d}" for i in range(0, 6))
    idx_top = " ".join(f"{i:2d}" for i in range(12, 6, -1))
    idx_bottom = " ".join(f"{i:2d}" for i in range(0, 6))
    lines = [
        f"   [13:{b[13]:2d}]   {top}",
        f"            {idx_top}",
        f"            {idx_bottom}",
        f"   {bottom}   [6:{b[6]:2d}]",
    ]
    return "\n".join(lines)


def human_move(b: List[int]) -> int:
    while True:
        raw = input("Wybierz dołek (1-6, odpowiada indeksom 0-5): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not raw.isdigit():
            print("Podaj liczbę 1-6.")
            continue
        pit = int(raw)
        if pit < 1 or pit > 6:
            print("Zakres 1-6.")
            continue
        idx = pit - 1
        if game.initial_state():  # tylko by uniknąć lintera — funkcja używana gdzie indziej
            pass
        return idx


def play_cli(depth: int, ai_vs_ai: bool, human_is_player0: bool):
    b = game.initial_state()
    player_to_move = 0
    human_player = 0 if human_is_player0 else 1

    print("Start gry Kalah (Mankala).")
    print("Twoja strona to dołki 0-5 (wybierasz 1-6).")
    print(format_board(b))

    while True:
        if game.side_empty(b, 0) or game.side_empty(b, 1):
            game.collect_remaining(b)
            print("Koniec gry.")
            print(format_board(b))
            p0 = b[game.store_index(0)]
            p1 = b[game.store_index(1)]
            if p0 > p1:
                print(f"Wygrywa gracz 0 ({p0} vs {p1}).")
            elif p1 > p0:
                print(f"Wygrywa gracz 1 ({p1} vs {p0}).")
            else:
                print(f"Remis ({p0} vs {p1}).")
            break

        if ai_vs_ai or player_to_move != human_player:
            move = ai.choose_move(b, player_to_move, depth)
            side = "AI" if ai_vs_ai or player_to_move != human_player else "Człowiek"
            print(f"{side} (gracz {player_to_move}) wybiera dołek {move}.")
        else:
            # Ruch człowieka — tylko pity po jego stronie
            while True:
                idx = human_move(b)
                if idx not in game.player_pit_range(player_to_move):
                    print("To nie jest Twój dołek. Spróbuj ponownie.")
                    continue
                if b[idx] == 0:
                    print("Ten dołek jest pusty.")
                    continue
                move = idx
                break

        b, next_player, extra, game_over = game.apply_move(b, player_to_move, move)
        print(format_board(b))
        if extra:
            print(f"Gracz {player_to_move} ma dodatkowy ruch!")
        player_to_move = next_player


def main():
    parser = argparse.ArgumentParser(description="Mankala (Kalah) — gra z AI")
    parser.add_argument("--depth", type=int, default=6, help="Głębokość minimax (1-10)")
    parser.add_argument("--ai-vs-ai", action="store_true", help="Uruchom pojedynek AI vs AI")
    parser.add_argument(
        "--human-player0",
        action="store_true",
        help="Człowiek gra jako gracz 0 (domyślnie tak)",
    )
    args = parser.parse_args()

    depth = max(1, min(10, args.depth))
    play_cli(depth, args.ai_vs_ai, args.human_player0 or True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.")