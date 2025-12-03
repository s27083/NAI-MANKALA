"""
Zadanie 4: Klasyfikacja danych dwoma algorytmami (Drzewo, SVM).
Autorzy: Adrian Kemski s27444, Kamil Bogdański s27083
Użycie:
  # Zbiór 1: Heart Failure (Kaggle) — pobierz CSV i wskaż ścieżkę
  python -m classification_task.main --dataset heart --csv path/to/heart_failure_clinical_records_dataset.csv \
      --model svm --kernel rbf --report text

  # Zbiór 2: Breast Cancer (sklearn)
  python -m classification_task.main --dataset breast --model dt --criterion entropy --report text

Referencje:
 - Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes
   https://jns.edu.al/wp-content/uploads/2024/01/M.UlqinakuA.Ktona-FINAL.pdf
 - Zbiór 1: Heart Failure Clinical Data (Kaggle)
   https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
 - Zbiór 2: Breast Cancer Wisconsin (Diagnostic) (UCI)
   https://archive.ics.uci.edu/dataset/33/breast+cancer+wisconsin+diagnostic
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loader import HeartFailureLoader, BreastCancerLoader
from .models import DecisionTreeModel, SVMModel
from .evaluation import Evaluator
from .visualization import Visualizer

DATASET_LINKS = {
    "heart_failure": "https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data",
    "breast_cancer": "https://archive.ics.uci.edu/dataset/33/breast+cancer+wisconsin+diagnostic",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zadanie 4 — klasyfikacja danych: Drzewo vs SVM")
    parser.add_argument("--dataset", choices=["heart", "breast"], required=True, help="Wybór zbioru danych")
    parser.add_argument("--csv", type=str, default=None, help="Ścieżka do CSV dla zbioru Heart Failure (Kaggle)")
    parser.add_argument("--model", choices=["dt", "svm"], default="svm", help="Wybór klasyfikatora")
    parser.add_argument("--criterion", choices=["gini", "entropy"], default="gini", help="Kryterium drzewa")
    parser.add_argument("--max-depth", type=int, default=None, help="Maksymalna głębokość drzewa")
    parser.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="rbf", help="Kernel SVM")
    parser.add_argument("--C", type=float, default=1.0, help="Parametr C dla SVM")
    parser.add_argument("--gamma", type=str, default="scale", help="Gamma dla SVM (scale, auto lub liczba)")
    parser.add_argument("--degree", type=int, default=3, help="Stopień dla kernel=poly")
    parser.add_argument("--report", choices=["text", "json"], default="text", help="Format raportu")
    parser.add_argument("--output", type=str, default=None, help="Ścieżka do zapisu wyników")
    parser.add_argument("--run-kernel-exp", action="store_true", help="Uruchom eksperyment porównania kernel SVM")
    parser.add_argument("--predict-sample", action="store_true", help="Wyświetl predykcję dla przykładowej próbki z testu")
    parser.add_argument("--sample-index", type=int, default=0, help="Indeks próbki testowej dla predykcji")
    return parser.parse_args(argv)


def load_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, str]:
    if args.dataset == "heart":
        if not args.csv:
            raise SystemExit("Dla zbioru heart musisz podać --csv ze ścieżką do pliku Kaggle.")
        X, y = HeartFailureLoader(csv_path=Path(args.csv)).load()
        name = "heart_failure"
    else:
        X, y = BreastCancerLoader().load()
        name = "breast_cancer"
    return X, y, name


def build_model(args: argparse.Namespace):
    if args.model == "dt":
        model = DecisionTreeModel(criterion=args.criterion, max_depth=args.max_depth).build()
    else:
        gamma_val = args.gamma
        try:
            gamma_val = float(args.gamma) if args.gamma not in ("scale", "auto") else args.gamma
        except ValueError:
            gamma_val = "scale"
        model = SVMModel(kernel=args.kernel, C=args.C, gamma=gamma_val, degree=args.degree).build()
    return model


def kernel_experiments(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> Path:
    evaluator = Evaluator()
    configs = [
        ("linear_C1", dict(kernel="linear", C=1.0)),
        ("linear_C10", dict(kernel="linear", C=10.0)),
        ("rbf_scale", dict(kernel="rbf", gamma="scale", C=1.0)),
        ("rbf_auto", dict(kernel="rbf", gamma="auto", C=1.0)),
        ("rbf_g0.1", dict(kernel="rbf", gamma=0.1, C=1.0)),
        ("poly_deg3", dict(kernel="poly", degree=3, C=1.0, gamma="scale")),
        ("poly_deg5", dict(kernel="poly", degree=5, C=1.0, gamma="scale")),
        ("sigmoid", dict(kernel="sigmoid", gamma="scale", C=1.0)),
    ]
    lines: list[str] = ["Podsumowanie wpływu kerneli SVM na wyniki:"]
    for name, cfg in configs:
        clf = SVMModel(**cfg).build()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)  # type: ignore
            except Exception:
                y_proba = None
        m = evaluator.evaluate(y_test.values, y_pred, y_proba)
        lines.append(f"{name}: accuracy={m['accuracy']:.3f}, f1={m['f1']:.3f}, precision={m['precision']:.3f}, recall={m['recall']:.3f}")
    out = out_dir / "kernel_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    X, y, name = load_dataset(args)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model(args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_test.values, y_pred, y_proba)

    viz = Visualizer(output_dir=Path("classification_task/output"))
    corr_path = viz.correlation_heatmap(X_train, name)
    cm_path = viz.confusion_heatmap(y_test.values, y_pred, f"{name}_{args.model}")

    sample_path = None
    if args.predict_sample:
        idx = max(0, min(int(args.sample_index), len(X_test) - 1))
        x_sample = X_test.iloc[[idx]]
        y_true_sample = y_test.iloc[idx]
        y_pred_sample = model.predict(x_sample)[0]
        proba_sample = None
        if hasattr(model, "predict_proba"):
            try:
                proba_sample = model.predict_proba(x_sample)
            except Exception:
                proba_sample = None
        lines_sample = [
            f"Próbka testowa indeks: {idx}",
            f"Prawdziwa klasa: {y_true_sample}",
            f"Predykcja: {y_pred_sample}",
        ]
        if proba_sample is not None:
            probs = proba_sample[0].tolist()
            lines_sample.append(f"Prawdopodobieństwa klas: {probs}")
        sample_path = Path("classification_task/output") / f"sample_prediction_{name}_{args.model}.txt"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text("\n".join(lines_sample), encoding="utf-8")

    if args.run_kernel_exp and args.model == "svm":
        summary_path = kernel_experiments(X_train, y_train, X_test, y_test, Path("classification_task/output"))
    else:
        summary_path = None

    if args.report == "json":
        import json
        payload = {
            "dataset": name,
            "dataset_link": DATASET_LINKS.get(name),
            "model": args.model,
            "metrics": metrics,
            "visualizations": {
                "correlation": str(corr_path),
                "confusion": str(cm_path),
                "kernel_summary": str(summary_path) if summary_path else None,
            },
            "sample_prediction": str(sample_path) if sample_path else None,
        }
        out_text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        lines = [
            f"Zbiór: {name}",
            f"Link do zbioru: {DATASET_LINKS.get(name)}",
            f"Model: {args.model}",
            "Metryki:",
            *(f" - {k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)),
            "",
            f"Wizualizacje zapisano: {corr_path}",
            f"Macierz pomyłek: {cm_path}",
        ]
        if summary_path:
            lines.append(f"Podsumowanie kernel: {summary_path}")
        if sample_path:
            lines.append(f"Predykcja próbki: {sample_path}")
        out_text = "\n".join(lines)

    print(out_text)
    if args.output:
        out_file = Path(args.output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(out_text, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
