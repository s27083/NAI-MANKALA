import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

# Ustawienie ziarna losowości (seed) dla powtarzalności wyników
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_and_preprocess_data():
    """
    Pobiera i przetwarza zbiór danych Abalone.
    Zmienia problem regresji (przewidywanie liczby pierścieni) na klasyfikację (młody, średni, stary).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
    
    print("Pobieranie zbioru danych Abalone...")
    df = pd.read_csv(url, names=column_names)
    
    # Problem klasyfikacji: Grupowanie pierścieni (wiek) w klasy
    # Młody: 1-8, Średni: 9-10, Stary: 11+
    def bin_rings(rings):
        if rings <= 8:
            return 0 # Młody
        elif rings <= 10:
            return 1 # Średni
        else:
            return 2 # Stary
            
    df['Target'] = df['Rings'].apply(bin_rings)
    class_names = ['Young (1-8)', 'Medium (9-10)', 'Old (11+)']
    
    # Usunięcie oryginalnej kolumny celu (regresji)
    df = df.drop('Rings', axis=1)
    
    # Kodowanie kolumny płci (Sex) - zamiana tekstu na liczby
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    
    # Rozdzielenie cech (X) i etykiet (y)
    X = df.drop('Target', axis=1).values
    y = df['Target'].values
    
    # Skalowanie cech (standaryzacja) - ważne dla sieci neuronowych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, class_names

def create_model(input_shape, num_classes):
    """
    Tworzy prostą sieć neuronową (MLP) do klasyfikacji.
    """
    model = tf.keras.models.Sequential([
        # Pierwsza warstwa gęsta z 64 neuronami i funkcją aktywacji ReLU
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        # Normalizacja wsadowa dla stabilizacji treningu
        tf.keras.layers.BatchNormalization(),
        # Dropout dla redukcji przeuczenia
        tf.keras.layers.Dropout(0.2),
        
        # Druga warstwa gęsta
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Warstwa wyjściowa z funkcją Softmax (prawdopodobieństwa klas)
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    save_dir = 'results/task1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Konfiguracja logowania (zapis wyjścia do pliku oraz na konsolę)
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

    X, y, class_names = load_and_preprocess_data()
    
    # Podział na zbiór treningowy i testowy (odłożony - Hold-out)
    # 80% trening, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    k_folds = 2
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    
    print(f"\nRozpoczynanie {k_folds}-krotnej walidacji krzyżowej (Cross-Validation)...")
    
    fold_no = 1
    cv_accuracies = []
    
    for train_index, val_index in skf.split(X_train_full, y_train_full):
        print(f"\nIteracja (Fold) {fold_no}/{k_folds}...")
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]
        
        model = create_model((X_train.shape[1],), len(class_names))
        
        history = model.fit(
            X_train, y_train,
            epochs=20, # Liczba epok
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f"  Dokładność (Accuracy): {scores[1]:.4f}")
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
        plt.title(f'Fold {fold_no} Strata (Loss)')
        plt.legend()
        plt.savefig(f'{save_dir}/history_fold_{fold_no}.png')
        plt.close()
        
        fold_no += 1
        
    print(f"\nŚrednia dokładność CV: {np.mean(cv_accuracies):.4f}")
    
    # Ostateczne trenowanie na pełnym zbiorze treningowym
    print("\nPonowne trenowanie na pełnym zbiorze treningowym...")
    final_model = create_model((X_train_full.shape[1],), len(class_names))
    final_model.fit(X_train_full, y_train_full, epochs=20, batch_size=32, verbose=1)
    
    # Ewaluacja na zbiorze testowym
    print("\nEwaluacja na zbiorze testowym...")
    y_pred_prob = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność na zbiorze testowym: {test_acc:.4f}")
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Macierz pomyłek (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz Pomyłek - Abalone')
    plt.ylabel('Prawdziwa Etykieta')
    plt.xlabel('Przewidziana Etykieta')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    print(f"Wyniki zapisano w katalogu {save_dir}")

if __name__ == "__main__":
    main()
