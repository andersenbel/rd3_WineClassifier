import torch.nn as nn


class WineClassifier(nn.Module):
    """
    Клас для нейронної мережі, яка виконує класифікацію вина.
    """

    def __init__(self, input_size, num_classes):
        """
        Ініціалізація нейронної мережі.

        :param input_size: Кількість вхідних ознак (features)
        :param num_classes: Кількість класів для класифікації
        """
        super(WineClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),  # Додаємо Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout для регуляризації
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # Додаємо Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.3),  # Ще один Dropout
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)  # Для багатокласової класифікації
        )

    def forward(self, x):
        """
        Виконує прямий прохід через нейронну мережу.

        :param x: Вхідні дані
        :return: Результати моделі
        """
        return self.model(x)
