import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

# Ustawienie ziarna losowości (seed) dla powtarzalności wyników
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_data():
    """
    Pobiera dane Fashion MNIST i przygotowuje je do treningu.
    """
    (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
    
    # Normalizacja wartości pikseli (0-255 -> 0-1)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Obrazy Fashion MNIST są w skali szarości (28, 28), dodajemy wymiar kanału (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return X_train, y_train, X_test, y_test, class_names

def create_small_model(input_shape, num_classes):
    """Mały model (Prosta sieć MLP)"""
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, "Small_Model"

def create_large_model(input_shape, num_classes):
    """Duży model (Głęboka sieć CNN)"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, "Large_Model"

def run_cv_and_evaluate(create_model_fn, X, y, X_test, y_test, class_names, save_dir):
    """
    Przeprowadza walidację krzyżową i ewaluację dla podanej funkcji tworzącej model.
    """
    input_shape = X.shape[1:]
    num_classes = len(class_names)
    
    # Utworzenie tymczasowego modelu, aby pobrać nazwę
    _, model_name = create_model_fn(input_shape, num_classes)
    
    print(f"\nPrzetwarzanie modelu: {model_name}...")
    
    k_folds = 2
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    
    fold_no = 1
    cv_accuracies = []
    
    # Walidacja krzyżowa (Cross Validation)
    for train_index, val_index in skf.split(X, y):
        print(f"  Iteracja (Fold) {fold_no}/{k_folds}...")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        model, _ = create_model_fn(input_shape, num_classes)
        
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=3, # Niska liczba epok dla demonstracji
            batch_size=64,
            validation_data=(X_val_fold, y_val_fold),
            verbose=0
        )
        
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_accuracies.append(scores[1])
        fold_no += 1

    avg_cv_acc = np.mean(cv_accuracies)
    print(f"  Średnia dokładność CV: {avg_cv_acc:.4f}")
    
    # Ponowne trenowanie na pełnym zbiorze treningowym dla ostatecznego testu
    print(f"  Ponowne trenowanie {model_name} na pełnym zbiorze danych...")
    final_model, _ = create_model_fn(input_shape, num_classes)
    history = final_model.fit(X, y, epochs=3, batch_size=64, verbose=1)
    
    # Ewaluacja na zbiorze testowym
    y_pred_prob = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"  Dokładność na zbiorze testowym: {test_acc:.4f}")
    
    # Zapis macierzy pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Macierz Pomyłek - {model_name}')
    plt.ylabel('Prawdziwa Etykieta')
    plt.xlabel('Przewidziana Etykieta')
    plt.savefig(f'{save_dir}/confusion_matrix_{model_name}.png')
    plt.close()
    
    # Zapis historii trenowania ostatecznego modelu
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Dokładność')
    plt.plot(history.history['loss'], label='Strata')
    plt.title(f'Historia trenowania - {model_name}')
    plt.legend()
    plt.savefig(f'{save_dir}/history_{model_name}.png')
    plt.close()
    
    return {
        'name': model_name,
        'cv_acc': avg_cv_acc,
        'test_acc': test_acc,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names)
    }

def main():
    save_dir = 'results/task3'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Konfiguracja logowania
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(f"{save_dir}/training_logs.txt", "w")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger()

    print("Ładowanie danych Fashion-MNIST...")
    X, y, X_test, y_test, class_names = load_data()
    
    # Porównanie Modeli
    res_small = run_cv_and_evaluate(create_small_model, X, y, X_test, y_test, class_names, save_dir)
    res_large = run_cv_and_evaluate(create_large_model, X, y, X_test, y_test, class_names, save_dir)
    
    # Wyniki końcowego porównania
    print("\n" + "="*60)
    print("WYNIKI PORÓWNANIA MODELI")
    print("="*60)
    print(f"{'Metryka':<20} | {res_small['name']:<15} | {res_large['name']:<15}")
    print("-" * 60)
    print(f"{'Dokładność CV':<20} | {res_small['cv_acc']:.4f}{' '*9} | {res_large['cv_acc']:.4f}")
    print(f"{'Dokładność Test':<20} | {res_small['test_acc']:.4f}{' '*9} | {res_large['test_acc']:.4f}")
    print("-" * 60)
    
    print(f"\nSzczegółowy raport dla {res_large['name']} (Najlepszy Kandydat):")
    print(res_large['classification_report'])

if __name__ == "__main__":
    main()
