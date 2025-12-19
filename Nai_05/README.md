# Projekt Sieci Neuronowych

To repozytorium zawiera implementacje sieci neuronowych do zadań klasyfikacji (obrazy oraz dane tabelaryczne) przy użyciu TensorFlow/Keras.

## Struktura Projektu

```
.
├── requirements.txt         # Zależności projektu
├── task1_abalone.py         # Klasyfikacja Abalone (Zadanie 1)
├── task2_cifar10_cnn.py     # Rozpoznawanie Zwierząt - CIFAR-10 (Zadanie 2)
├── task3_fashion_mnist.py   # Rozpoznawanie Ubrań - Fashion-MNIST (Zadanie 3)
├── task4_wine.py            # Klasyfikacja Win - Wine Dataset (Zadanie 4)
└── results/                 # Katalog zawierający logi i wykresy
    ├── task1/               # Wyniki dla Abalone
    ├── task2/               # Wyniki dla CIFAR-10
    ├── task3/               # Wyniki dla Fashion-MNIST
    └── task4/               # Wyniki dla Wine
```

## Konfiguracja (Setup)

1. **Przygotowanie Środowiska**
   Upewnij się, że masz zainstalowanego Pythona 3.8+. Zalecane jest użycie wirtualnego środowiska.

   ```bash
   # Utwórz wirtualne środowisko
   python3 -m venv venv
   
   # Aktywuj środowisko
   source venv/bin/activate  # Na Windows: venv\Scripts\activate
   ```

2. **Instalacja Zależności**
   ```bash
   pip install -r requirements.txt
   ```

## Zadanie 1: Zbiór Abalone
Zadanie klasyfikacji wieku uchowców (Abalone) na podstawie cech fizycznych.
- **Problem**: Klasyfikacja na 3 grupy wiekowe (Młody, Średni, Stary).
- **Model**: Prosta sieć neuronowa (MLP).
- **Uruchomienie**: `python task1_abalone.py`

## Zadanie 2: Rozpoznawanie Zwierząt (CIFAR-10)
Zadanie implementuje Konwolucyjną Sieć Neuronową (CNN) do klasyfikacji obrazów ze zbioru CIFAR-10 na 10 kategorii (samolot, samochód, ptak, kot, jeleń, pies, żaba, koń, statek, ciężarówka).

### Cechy
- **Architektura**: 3-blokowa sieć CNN z Normalizacją Wsadową (Batch Normalization), MaxPooling i Dropout.
- **Walidacja**: 2-krotna Walidacja Krzyżowa (Stratified K-Fold).
- **Ewaluacja**: Dokładność (Accuracy), Macierz Pomyłek (Confusion Matrix).

### Uruchomienie
```bash
python task2_cifar10_cnn.py
```
Wyniki zostaną zapisane w `results/task2/`.

## Zadanie 3: Rozpoznawanie Ubrań (Fashion-MNIST)
Zadanie porównuje dwie architektury sieci neuronowych do klasyfikacji ubrań ze zbioru Fashion-MNIST.

### Cechy
- **Porównanie**: Porównuje "Mały Model" (MLP) z "Dużym Modelem" (Głęboka sieć CNN).
- **Walidacja**: 2-krotna Walidacja Krzyżowa.
- **Ewaluacja**: Porównanie dokładności, Macierze Pomyłek.

### Uruchomienie
```bash
python task3_fashion_mnist.py
```
Wyniki zostaną zapisane w `results/task3/`.

## Zadanie 4: Klasyfikacja Win
Zadanie klasyfikacji win na 3 gatunki na podstawie analizy chemicznej.
- **Zbiór danych**: Wine Dataset (Scikit-Learn).
- **Metoda**: Prosta sieć neuronowa (MLP) z 3-krotną Walidacją Krzyżową.
- **Uruchomienie**: `python task4_wine.py`

## Wyniki
Sprawdź katalog `results/` aby zobaczyć:
- `training_logs.txt`: Szczegółowe metryki treningowe.
- `history_*.png`: Wykresy krzywych straty (Loss) i dokładności (Accuracy).
- `confusion_matrix_*.png`: Macierze pomyłek obrazujące skuteczność modelu.
