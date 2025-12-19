import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

# Ustawienie ziarna losowości (seed) dla powtarzalności wyników
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_data():
    """
    Pobiera zbiór danych Wine (wina) z biblioteki scikit-learn.
    """
    print("Ładowanie zbioru danych Wine (wina)...")
    data = load_wine()
    X = data.data
    y = data.target
    class_names = data.target_names
    
    # Skalowanie cech (standaryzacja)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, class_names

def create_model(input_shape, num_classes):
    """
    Tworzy prostą sieć neuronową (MLP).
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    save_dir = 'results/task4'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Konfiguracja logowania (zapis wyjścia do pliku)
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

    X, y, class_names = load_data()
    
    # Podział na zbiór treningowy i testowy (odłożony - Hold-out)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    k_folds = 3
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    
    print(f"\nRozpoczynanie {k_folds}-krotnej walidacji krzyżowej...")
    
    fold_no = 1
    cv_accuracies = []
    
    for train_index, val_index in skf.split(X_train_full, y_train_full):
        print(f"\nIteracja (Fold) {fold_no}/{k_folds}...")
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]
        
        model = create_model((X_train.shape[1],), len(class_names))
        
        history = model.fit(
            X_train, y_train,
            epochs=50, # Mały zbiór danych, szybkie trenowanie
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f"  Dokładność: {scores[1]:.4f}")
        cv_accuracies.append(scores[1])
        
        # Rysowanie wykresów historii trenowania dla danej iteracji
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Trening')
        plt.plot(history.history['val_accuracy'], label='Walidacja')
        plt.title(f'Fold {fold_no} Dokładność')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Trening')
        plt.plot(history.history['val_loss'], label='Walidacja')
        plt.title(f'Fold {fold_no} Strata')
        plt.legend()
        plt.savefig(f'{save_dir}/history_fold_{fold_no}.png')
        plt.close()
        
        fold_no += 1
        
    print(f"\nŚrednia dokładność CV: {np.mean(cv_accuracies):.4f}")
    
    # Ostateczne trenowanie na pełnym zbiorze treningowym
    print("\nPonowne trenowanie na pełnym zbiorze treningowym...")
    final_model = create_model((X_train_full.shape[1],), len(class_names))
    final_model.fit(X_train_full, y_train_full, epochs=50, batch_size=16, verbose=1)
    
    # Ewaluacja na zbiorze testowym
    print("\nEwaluacja na zbiorze testowym...")
    y_pred_prob = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność na zbiorze testowym: {test_acc:.4f}")
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz Pomyłek - Klasyfikacja Wina')
    plt.ylabel('Prawdziwa Etykieta')
    plt.xlabel('Przewidziana Etykieta')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    print(f"Wyniki zapisano w katalogu {save_dir}")

if __name__ == "__main__":
    main()
