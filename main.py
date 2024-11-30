import os
import torch
import torch.nn as nn

# Імпортуємо модулі, які знаходяться в тому ж каталозі
from dataset import load_data
from model import WineClassifier
from train import train_model
from evaluate import evaluate_model
from visualize import plot_losses

# Основна функція


def main():
    # Завантаження даних
    train_loader, val_loader, test_loader, input_size, num_classes = load_data()

    # Ініціалізація моделі
    model = WineClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4)

    # Навчання моделі
    print("Навчання моделі...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer)

    # Оцінка моделі
    print("Оцінка моделі на тестових даних...")
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Точність моделі на тестових даних: {test_accuracy:.4f}")

    # Збереження графіку втрат
    print("Збереження графіку втрат...")
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    plot_losses(train_losses, val_losses, loss_plot_path)
    print(f"Графік втрат збережено за адресою: {loss_plot_path}")


if __name__ == "__main__":
    main()
