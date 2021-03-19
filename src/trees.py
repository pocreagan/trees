from typing import Tuple
import numpy as np
import pandas as pd
from functools import cached_property
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# noinspection PyUnresolvedReferences
from tensorflow.keras import models
from dataclasses import InitVar, dataclass


@dataclass
class KnownSet:
    raw: InitVar[pd.DataFrame]
    features: pd.DataFrame = None  # type: ignore
    labels: pd.Series = None  # type: ignore

    def __post_init__(self, raw: pd.DataFrame) -> None:
        self.features = raw.copy()
        # noinspection SpellCheckingInspection
        self.labels = self.features.pop('BHAGE')


class TreeModel:
    input_fp = r"dat/Spp34_Data_ML_IN.csv"
    output_fp = r"dat/Spp34_Data_ML_OUT.csv"
    drop_features = tuple()
    categorical_features = ('Species_Index',)
    train_ratio = .8

    @cached_property
    def raw_dataset(self):
        return pd.read_csv(self.input_fp, na_values='?', comment='\t',
                           sep=',', skipinitialspace=True)

    @cached_property
    def dataset(self) -> pd.DataFrame:
        dataset = self.raw_dataset.copy()

        for k in self.drop_features:
            dataset = dataset.drop(k, 1)

        for k in self.categorical_features:
            dataset[k] = dataset[k].map({v: f'{k}=[{v}]' for v in dataset[k].unique()})  # type: ignore
            dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

        return dataset

    @cached_property
    def unknown_dataset(self) -> pd.DataFrame:
        return self.dataset[self.dataset.BHAGE == 0]  # type: ignore

    @cached_property
    def known_dataset(self) -> pd.DataFrame:
        return self.dataset.drop(self.unknown_dataset.index)  # type: ignore

    @cached_property
    def training_dataset(self) -> pd.DataFrame:
        return self.known_dataset.sample(frac=self.train_ratio, random_state=0)

    @cached_property
    def training_data(self) -> KnownSet:
        return KnownSet(self.training_dataset)

    @cached_property
    def testing_data(self) -> KnownSet:
        return KnownSet(self.known_dataset.drop(self.training_dataset.index))  # type: ignore

    @cached_property
    def predictive_data(self) -> KnownSet:
        return KnownSet(self.unknown_dataset)

    @cached_property
    def test_predictions(self) -> pd.Series:
        return self.load_model().predict(self.testing_data.features).flatten()  # type: ignore

    def persist_dataframe(self, df: pd.DataFrame):
        df.to_csv(self.output_fp, sep=',')

    def train_model(self, node_counts: Tuple[int], epochs: int, val_ratio: float):
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(self.training_data.features))
        # noinspection PyTypeChecker
        self.model = keras.Sequential(
            [normalizer] + [
                layers.Dense(nodes, activation='relu') for nodes in node_counts
            ] + [layers.Dense(1)]
        )
        self.model.compile(loss='mean_absolute_error',
                           optimizer=tf.keras.optimizers.Adam(0.001))
        self.history = self.model.fit(self.training_data.features, self.training_data.labels,
                                      validation_split=val_ratio,
                                      verbose=1, epochs=epochs)
        self.__plot_training_loss()
        self.model.save('model')

    def __plot_training_loss(self) -> None:
        if self.history is not None:
            plt.plot(self.history.history['loss'], label='loss')
            plt.plot(self.history.history['val_loss'], label='val_loss')
            plt.ylim([0, 60])
            plt.xlabel('Epoch')
            plt.ylabel('Error [years]')
            plt.legend()
            plt.grid(True)
            self.finish_plot('plots/training_error.png')

    def plot_test_predictions(self) -> None:
        plt.axes(aspect='equal')
        plt.scatter(self.testing_data.labels, self.test_predictions)
        plt.xlabel('Known Ages')
        plt.ylabel('Predictions')
        limits = [0, int(max(self.testing_data.labels)*1.1)]
        plt.xlim(limits)
        plt.ylim(limits)
        _ = plt.plot(limits, limits)
        self.finish_plot('plots/test_predictions.png')

    def plot_prediction_error(self) -> None:
        error = self.test_predictions - self.testing_data.labels
        plt.hist(error, bins=100)
        plt.xlabel('Prediction Error [years]')
        _ = plt.ylabel('Count')
        self.finish_plot('plots/prediction_err_hist.png')

    @staticmethod
    def load_model():
        return tf.keras.models.load_model('model')

    @staticmethod
    def finish_plot(filename: str) -> None:
        plt.savefig(filename)
        plt.waitforbuttonpress()
        plt.clf()

    def __init__(self) -> None:
        self.history = None
        self.model = None


if __name__ == '__main__':
    tree = TreeModel()
    tree.train_model(tuple(128 for _ in range(12)), 150, .2)
    tree.plot_test_predictions()
    tree.plot_prediction_error()
