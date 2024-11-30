import torch
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def check_data_quality(X, y):
    # 1. Перевірка на пропущені значення
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Пропущені значення знайдено у наборі даних!")

    # 2. Перевірка на аномалії
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    outliers = ((X < X_mean - 3 * X_std) |
                (X > X_mean + 3 * X_std)).any(axis=1)
    if np.sum(outliers) > 0:
        print(f"Попередження: У даних знайдено {np.sum(outliers)} аномалій.")
        # Аномалії можна або видалити, або залишити:
        X = X[~outliers]
        y = y[~outliers]

    # 3. Перевірка на типи даних
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("Некоректний тип даних у X!")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("Некоректний тип даних у y!")

    print("Набір даних пройшов перевірку якості.")
    return X, y


def load_data(batch_size=32):
    # Завантаження даних
    data = load_wine()
    X = data.data
    y = data.target

    # Перевірка даних
    X, y = check_data_quality(X, y)

    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to tensors
    X_train, y_train = torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(
        X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test, y_test = torch.tensor(
        X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(X_train[0]), len(data.target_names)
