# Zadanie 4: Klasyfikacja (Drzewo vs SVM)

Problem: Klasyfikacja danych dwoma algorytmami (Drzewo decyzyjne, SVM), z metrykami, wizualizacjami i porównaniem kerneli.

Autorzy: Adrian Kemski s27444, Kamil Bogdański s27083

Referencje:
- Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes — https://jns.edu.al/wp-content/uploads/2024/01/M.UlqinakuA.Ktona-FINAL.pdf
- Zbiór 1: Heart Failure Clinical Data (Kaggle) — https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
- Zbiór 2: Breast Cancer Wisconsin (Diagnostic) (UCI) — https://archive.ics.uci.edu/dataset/33/breast+cancer+wisconsin+diagnostic

## Wymagania
- Python 3.9 (użyto venv z modułu rekomendacji)
- Zależności: `classification_task/requirements.txt`

## Dane w repozytorium
- Plik CSV jest już dołączony: `classification_task/data/heart_failure_clinical_records_dataset.csv`

## Uruchomienie (przykłady)
- SVM na zbiorze serca (z porównaniem kerneli i predykcją próbki):
  `cd /Users/adriankemski/NAI-MANKALA && recommendation_engine/.venv/bin/python -m classification_task.main --dataset heart --csv /Users/adriankemski/NAI-MANKALA/classification_task/data/heart_failure_clinical_records_dataset.csv --model svm --kernel rbf --report text --run-kernel-exp --predict-sample`
- Drzewo decyzyjne na zbiorze serca (entropy, gryf max_depth=5):
  `cd /Users/adriankemski/NAI-MANKALA && recommendation_engine/.venv/bin/python -m classification_task.main --dataset heart --csv /Users/adriankemski/NAI-MANKALA/classification_task/data/heart_failure_clinical_records_dataset.csv --model dt --criterion entropy --max-depth 5 --report text --predict-sample`
- SVM na zbiorze piersi (bez CSV):
  `cd /Users/adriankemski/NAI-MANKALA && recommendation_engine/.venv/bin/python -m classification_task.main --dataset breast --model svm --kernel rbf --report text --run-kernel-exp --predict-sample`

## Co jest generowane
- Metryki jakości: accuracy, precision, recall, F1, ROC-AUC, TP/TN/FP/FN (w konsoli lub JSON)
- Wizualizacje:
  - `classification_task/output/<dataset>_corr.png` — macierz korelacji
  - `classification_task/output/<dataset>_<model>_cm.png` — macierz pomyłek
- Podsumowanie kerneli SVM:
  - `classification_task/output/kernel_summary.txt`
- Predykcja przykładowej próbki testowej:
  - `classification_task/output/sample_prediction_<dataset>_<model>.txt`

## Ustawienia modeli
- Drzewo decyzyjne: `--criterion {gini, entropy}`, `--max-depth <int>`
- SVM: `--kernel {linear, rbf, poly, sigmoid}`, `--C <float>`, `--gamma {scale, auto, liczba}`, `--degree <int>`

## Podsumowanie kerneli SVM
- Wyniki porównawcze zapisane w `classification_task/output/kernel_summary.txt`.
- Obserwacje (Breast Cancer):
  - `linear` z większym `C` (np. 10) poprawia `accuracy` i `recall` względem `C=1` (0.982 vs 0.974; `recall` 0.986 vs 0.972).
  - `rbf` z `gamma=scale/auto` daje najwyższe, zbalansowane wyniki (`accuracy` ~0.982, `f1` ~0.986, `recall` ~0.986).
  - `rbf` z większym `gamma` (np. 0.1) obniża `recall` i stabilność (`recall` ~0.944), typowy efekt zbyt „wąskiego” kernela.
  - `poly` (deg=3/5) osiąga `recall`≈1.0, ale kosztem spadku `precision` (0.878/0.809) i `accuracy` (0.912/0.851) — wzrost FP, ryzyko nadmiernego dopasowania.
  - `sigmoid` daje wyniki pośrednie (`accuracy` ~0.92, `precision` ~0.96, `recall` ~0.92), zwykle gorsze od `rbf/linear`.

## Notatka o wynikach
- Na zbiorze serca SVM (RBF) osiąga zwykle wysokie ROC-AUC, ale niższy recall; drzewo (entropy) często zwiększa recall kosztem precision.
- Szczegółowe wyniki dla kerneli znajdziesz w `classification_task/output/kernel_summary.txt`.
