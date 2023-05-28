import glob
import json as json
import os
from functools import reduce

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
import keras.backend as K
import tensorflow as tf
from typing import List

_frequency = 200


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def make_dataset(X, y, is_train):
    print(f"x=${X.shape}")
    print(f"y=${y.shape}")
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if is_train:
        dataset = dataset.shuffle(32)

    dataset = dataset.batch(4)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


class Dataset:
    exerciseType = to_categorical([0, 1, 2, 3, 4], 5)

    # 0 - predict
    # 1 - train
    # 2 - test
    # 3 - raw_text
    def __init__(self, is_load_type: int = True, window_time: int = 1, raw_json: dict = {}):
        self.x_fields = ["acc_gx", "acc_gy", "acc_gz", "gyr_gx", "gyr_gy", "gyr_gz"]  # "timestamp", "fixingPoint",
        self.y_fields = ["exerciseType"]

        self.df: DataFrame = DataFrame()
        self.window_size = int(window_time * _frequency)  # размер окна
        self.x = np.ndarray(shape=(0, self.window_size, len(self.x_fields)))  # сырые фичи для обучения
        self.y = np.ndarray(shape=(0,
                                   5))  # выход. массив длинною в 5 элементов, где каждый элемент - вероятность принадлежности к классу по индексу
        self.x_feature = np.ndarray(shape=(0, 1))  # фичи для обучения извлеченные из сырых
        self.y_feature = np.ndarray(shape=(0, 1))  # выход для обучения.
        if is_load_type == 0:
            self._load_predict_data()

        if is_load_type == 1:
            self._load_data(is_train=True)

        if is_load_type == 2:
            self._load_data(is_train=False)

        if is_load_type == 3:
            self._load_from_text(raw_json)

        self._feature_extraction()

    # загрузка данных из папки
    def _load_from_text(self, json_text: dict):
        dfs = []

        def convert(data, count):
            local_buffer = []

            def lamb(x: dict):
                x['id'] = count
                x["acc" + '_gx'] = x.pop('gx')
                x["acc" + '_gy'] = x.pop('gy')
                x["acc" + '_gz'] = x.pop('gz')
                return x

            size = min(len(data['accelerometerValue']), len(data['gyroscopeValue']))
            size = size - size % self.window_size  # кейс, когда данных не хватает, мы тогда обрезаем датасет под целочисленный размер окна
            data['accelerometerValue'] = data['accelerometerValue'][:size]
            data['gyroscopeValue'] = data['gyroscopeValue'][:size]

            c = list(map(lamb, data['accelerometerValue']))

            for i in range(size):
                c[i]["gyr" + '_gx'] = data['gyroscopeValue'][i]['gx']
                c[i]["gyr" + '_gy'] = data['gyroscopeValue'][i]['gy']
                c[i]["gyr" + '_gz'] = data['gyroscopeValue'][i]['gz']

            return list(c)

        count = 0


        buffer = []

        reading = json_text

        for line in reading:
            dat = convert(line['data'], count)[:_frequency * line['duration']]
            pd_data = pd.json_normalize(dat)
            if pd_data.shape[1] == 0:
                continue
            pd_data_x = pd_data[self.x_fields].to_numpy().reshape(-1, self.window_size, len(self.x_fields))
            self.x = np.concatenate([self.x, pd_data_x])
            if line["trainId"] == 9999:
                line["trainId"] = 3
            temp_y = np.concatenate(
                [Dataset.exerciseType[line["trainId"] - 1].reshape(1, -1) for _ in
                 range(pd_data_x.shape[0])])
            self.y = np.concatenate([self.y, temp_y])
            buffer.append(dat)
            count += 1

        buffer = reduce(lambda x, y: x + y, buffer)
        json_data = pd.json_normalize(buffer)

        dfs.append(json_data)
        buffer.clear()
        df = pd.concat(dfs)
        return df

    # загрузка данных из папки
    def _load_from_path(self, path: str):
        json_dir = path
        json_pattern = os.path.join(json_dir, '*.json')
        file_list = glob.glob(json_pattern)
        dfs = []

        def convert(data, count):
            local_buffer = []

            def lamb(x: dict):
                x['id'] = count
                x["acc" + '_gx'] = x.pop('gx')
                x["acc" + '_gy'] = x.pop('gy')
                x["acc" + '_gz'] = x.pop('gz')
                return x

            size = min(len(data['accelerometerValue']), len(data['gyroscopeValue']))
            size = size - size % self.window_size  # кейс, когда данных не хватает, мы тогда обрезаем датасет под целочисленный размер окна
            data['accelerometerValue'] = data['accelerometerValue'][:size]
            data['gyroscopeValue'] = data['gyroscopeValue'][:size]

            c = list(map(lamb, data['accelerometerValue']))

            for i in range(size):
                c[i]["gyr" + '_gx'] = data['gyroscopeValue'][i]['gx']
                c[i]["gyr" + '_gy'] = data['gyroscopeValue'][i]['gy']
                c[i]["gyr" + '_gz'] = data['gyroscopeValue'][i]['gz']

            return list(c)

        count = 0
        for file in file_list:

            buffer = []
            with open(file) as f:
                reading = json.loads(f.read())

            for line in reading:
                dat = convert(line['data'], count)[:_frequency * line['duration']]
                pd_data = pd.json_normalize(dat)
                if pd_data.shape[1] == 0:
                    continue
                pd_data_x = pd_data[self.x_fields].to_numpy().reshape(-1, self.window_size, len(self.x_fields))
                self.x = np.concatenate([self.x, pd_data_x])
                if line["trainId"] == 9999:
                    line["trainId"] = 3
                temp_y = np.concatenate(
                    [Dataset.exerciseType[line["trainId"] - 1].reshape(1, -1) for _ in
                     range(pd_data_x.shape[0])])
                self.y = np.concatenate([self.y, temp_y])
                buffer.append(dat)
                count += 1

            if len(buffer) == 0:  # фикс для Сани
                continue
            buffer = reduce(lambda x, y: x + y, buffer)
            json_data = pd.json_normalize(buffer)

            dfs.append(json_data)
            buffer.clear()
        df = pd.concat(dfs)
        return df

    # загрузка тренеровочных или тестовых данных из папки
    def _load_data(self, is_train: bool = True):
        self.df = self._load_from_path("../data/train") if is_train else self._load_from_path(
            "../data/test")

    def _load_predict_data(self):
        self.df = self._load_from_path('../data/predict')

    def _feature_extraction(self):
        new_x = []
        for i in self.x:  # По каждой строке
            i: ndarray  # (Размер окна _1000, количество параметром _6)

            # ================ чтобы набить новые признаки, редактировать в этом окне ========================
            res = []
            n = i.shape[-1]
            for k in range(n):
                a: ndarray = i[:, k]
                res.append(a.std())
                res.append(a.mean())
                res.append(a.max())
                res.append(a.min())
                res.append(a.var())

            cov1 = np.cov(i[:, 0], i[:, 1])
            res.append(cov1[0][1])

            cov1 = np.cov(i[:, 0], i[:, 2])
            res.append(cov1[0][1])

            cov1 = np.cov(i[:, 1], i[:, 2])
            res.append(cov1[0][1])

            cov1 = np.cov(i[:, 3], i[:, 4])
            res.append(cov1[0][1])

            cov1 = np.cov(i[:, 3], i[:, 5])
            res.append(cov1[0][1])

            cov1 = np.cov(i[:, 4], i[:, 5])
            res.append(cov1[0][1])

            corr1 = np.corrcoef(i[:, 0], i[:, 1])
            res.append(corr1[0][1])

            corr1 = np.corrcoef(i[:, 0], i[:, 2])
            res.append(corr1[0][1])

            corr1 = np.corrcoef(i[:, 1], i[:, 2])
            res.append(corr1[0][1])

            corr1 = np.corrcoef(i[:, 3], i[:, 4])
            res.append(corr1[0][1])

            corr1 = np.corrcoef(i[:, 3], i[:, 5])
            res.append(corr1[0][1])

            corr1 = np.corrcoef(i[:, 4], i[:, 5])
            res.append(corr1[0][1])

            # ================================================================================================
            new_x.append(np.array(res).reshape(1, -1))
            res.clear()

        self.x_feature = np.concatenate(new_x)
        new_y = []
        for i in self.y:
            for j in range(i.shape[-1]):
                if i[j] == 1:
                    new_y.append(j)
        self.y_feature = np.array(new_y)


def _load_data(path: str) -> DataFrame:
    json_dir = path
    json_pattern = os.path.join(json_dir, '*.json')
    file_list = glob.glob(json_pattern)
    dfs = []

    def convert(data, fixingPoint, typeBreathing):
        local_buffer = []

        def lamb(x: dict):
            x['typeBreathing'] = typeBreathing
            x["fixingPoint"] = fixingPoint
            x["acc" + '_gx'] = x.pop('gx')
            x["acc" + '_gy'] = x.pop('gy')
            x["acc" + '_gz'] = x.pop('gz')
            return x

        size = min(len(data['accelerometerValue']), len(data['gyroscopeValue']))
        data['accelerometerValue'] = data['accelerometerValue'][:size]
        data['gyroscopeValue'] = data['gyroscopeValue'][:size]

        c = list(map(lamb, data['accelerometerValue']))

        for i in range(size):
            c[i]["gyr" + '_gx'] = data['gyroscopeValue'][i]['gx']
            c[i]["gyr" + '_gy'] = data['gyroscopeValue'][i]['gy']
            c[i]["gyr" + '_gz'] = data['gyroscopeValue'][i]['gz']

        return list(c)

    for file in file_list:

        buffer = []
        with open(file) as f:
            reading = json.loads(f.read())['list']

        for line in reading:
            buffer.append(convert(line['data'], line['fixingPoint'], line['typeBreathing']))

        buffer = reduce(lambda x, y: x + y, buffer)
        json_data = pd.json_normalize(buffer)
        dfs.append(json_data)
        buffer.clear()
    df = pd.concat(dfs)
    return df


def load_train() -> DataFrame:
    return _load_data("data\\train")


def load_test() -> DataFrame:
    return _load_data("data\\test")


# /home/aioki/Desktop/diplom_vitek/custom/Handler.py:248: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
def save_plot(data, field: List[str], title: str, filename: str):
    fig = plt.figure()
    for i_field in field:
        plt.plot(data[f'{i_field}'])

    plt.title(f'model {title}')
    plt.ylabel(f'{title}')
    plt.xlabel('epoch')
    plt.legend(field, loc='upper left')
    return fig.savefig(filename)


def save_plot_one_field(data, field: str, title: str, filename: str):
    return save_plot(data, [field, f"val_{field}"], title, filename)


if __name__ == '__main__':
    # asd = {"field1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "field2": [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]}
    # for i in range(2):
    #     save_plot(asd, ["field1", "field2"], "title", f"file#{i}.png")

    df = Dataset()
    None
