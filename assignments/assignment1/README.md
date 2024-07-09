# Задание 1

В этом задании мы реализуем некоторые методы машинного обучения, которые помогут нам с реализацией нейросетей в следующих заданиях.

Перед выполнением задания:
- Запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки.
- Установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с pip, вам сюда - https://pip.pypa.io/en/stable/quickstart/).

## Часть 1
Метод К-ближайших соседей (K-neariest neighbor classifier)

`KNN.ipynb` - следуйте инструкциям в ноутбуке.

## Часть 2
Линейный классификатор (Linear classifier)

`Linear classifier.ipynb` - следуйте инструкциям в ноутбуке.


num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # nearest training samples
            nearest_indices = np.argsort(dists[i])[:self.k]
            nearest_labels = self.train_y[nearest_indices]
            pred[i] = np.argmax(np.bincount(nearest_labels))
        return pred
