# Mankala (Kalah) — prosta gra turowa z AI

Link do zasad: https://pl.wikipedia.org/wiki/Kalaha

Autorzy: Kamil Bogdański Adrian Kemski

Instrukcja przygotowania środowiska:
- Wymagany Python 3.10 lub nowszy.
- Zainstaluj bibliotekę AI: `pip install easyAI`
- Uruchom grę: `python3 main.py`
- Opcjonalnie: `python3 main.py --ai-vs-ai --depth 6` uruchamia pojedynek AI vs AI.

Opis:
- Implementacja wariantu Kalah (Mankala) — gra dwuosobowa, deterministyczna, o sumie zerowej.
- W zestawie znajduje się sztuczna inteligencja oparta o minimax z przycinaniem alfa–beta.

Sterowanie (tryb człowiek vs AI):
- Wybierasz dołek z Twojej strony planszy (oznaczony 1–6).
- Zasady: ostatni kamyk w Twoim skarbcu daje dodatkowy ruch; jeśli ostatni kamyk wyląduje w pustym dołku po Twojej stronie, przejmujesz kamyki z dołka naprzeciwko.

Pliki:
- `main.py` — interfejs CLI, uruchomienie gry i opcje.
- `game.py` — logika gry i operacje na planszy.
- `ai.py` — implementacja minimax + alfa–beta.