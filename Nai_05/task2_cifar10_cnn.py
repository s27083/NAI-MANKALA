import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
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
    Pobiera zbiór danych CIFAR-10 i normalizuje go.
    """
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    
    # Normalizacja wartości pikseli do zakresu 0-1
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return X_train, y_train, X_test, y_test, class_names

def create_cnn_model(input_shape, num_classes):
    """
    Tworzy model konwolucyjnej sieci neuronowej (CNN).
    Składa się z 3 bloków konwolucyjnych oraz warstw gęstych na końcu.
    """
    model = models.Sequential([
        # Blok Konwolucyjny 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Blok Konwolucyjny 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Blok Konwolucyjny 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Spłaszczenie (Flatten) i warstwy gęste (Dense)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history, fold_no, save_dir):
    """
    Rysuje wykresy dokładności i straty dla danej iteracji.
    """
    plt.figure(figsize=(12, 5))
    
    # Dokładność (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Trening Dokładność')
    plt.plot(history.history['val_accuracy'], label='Walidacja Dokładność')
    plt.title(f'Fold {fold_no} - Dokładność')
    plt.ylabel('Dokładność')
    plt.xlabel('Epoka')
    plt.legend()
    
    # Strata (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Trening Strata')
    plt.plot(history.history['val_loss'], label='Walidacja Strata')
    plt.title(f'Fold {fold_no} - Strata')
    plt.ylabel('Strata')
    plt.xlabel('Epoka')
    plt.legend()
    
    plt.savefig(f'{save_dir}/history_fold_{fold_no}.png')
    plt.close()

def main():
    # Tworzenie katalogu na wyniki
    save_dir = 'results/task2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Przekierowanie standardowego wyjścia do konsoli i pliku
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

    print("Ładowanie danych CIFAR-10...")
    X, y, X_test, y_test, class_names = load_data()
    
    # Wykonujemy K-krotną walidację krzyżową na zbiorze treningowym
    k_folds = 2 # Zmniejszona liczba iteracji dla szybkości
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    
    fold_no = 1
    accuracies = []
    
    print(f"\nRozpoczynanie {k_folds}-krotnej walidacji krzyżowej na zbiorze treningowym...")
    
    for train_index, val_index in skf.split(X, y):
        print(f"\nTrenowanie Iteracja (Fold) {fold_no}...")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        model = create_cnn_model(X.shape[1:], len(class_names))
        
        # Wczesne zatrzymanie (Early Stopping) aby zapobiec przeuczeniu i zaoszczędzić czas
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=3, # Zmniejszona liczba epok dla demonstracji
            batch_size=64,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Ewaluacja na zbiorze walidacyjnym
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f'Wynik dla iteracji {fold_no}: {model.metrics_names[0]}: {scores[0]}; {model.metrics_names[1]}: {scores[1]*100}%')
        accuracies.append(scores[1])
        
        # Rysowanie historii
        plot_history(history, fold_no, save_dir)
        
        fold_no += 1
    
    print("\n" + "="*50)
    print(f"Średnia dokładność CV: {np.mean(accuracies)*100:.2f}% (+/- {np.std(accuracies)*100:.2f}%)")
    print("="*50)
    
    # Ostateczna ewaluacja na osobnym zbiorze testowym
    # Trenujemy nowy model na CAŁYM zbiorze treningowym
    
    print("\nPonowne trenowanie na pełnym zbiorze treningowym dla ostatecznego testu...")
    final_model = create_cnn_model(X.shape[1:], len(class_names))
    final_model.fit(X, y, epochs=3, batch_size=64, verbose=1) 
    
    print("\nEwaluacja na zbiorze testowym...")
    y_pred_prob = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # y_test ma wymiary (N, 1), spłaszczamy go
    y_true = y_test.flatten()
    
    test_acc = accuracy_score(y_true, y_pred)
    print(f"\nDokładność na zbiorze testowym: {test_acc:.4f}")
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz Pomyłek - Zbiór Testowy')
    plt.ylabel('Prawdziwa Etykieta')
    plt.xlabel('Przewidziana Etykieta')
    plt.savefig(f'{save_dir}/confusion_matrix_test.png')
    plt.close()
    
    print(f"Wyniki zapisano w katalogu {save_dir}")

if __name__ == "__main__":
    main()
