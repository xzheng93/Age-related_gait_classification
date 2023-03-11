from keras_tuner import HyperModel
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, BatchNormalization, ReLU, Dropout, \
    Bidirectional, Input, Conv2D, MaxPooling2D, TimeDistributed, ConvLSTM2D, Flatten, \
    MaxPooling1D, GRU, LSTM
import tensorflow as tf
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

# defines the hyper-parameters space for deep learning models
class BiLSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_layer = Input(shape=self.input_shape)
        x = input_layer
        for i in range(hp.Int("layers_number", 1, 3)):

            x = Bidirectional(LSTM(units=hp.Int(f"units_{i}", min_value=2, max_value=768, step=10),
                     bias_initializer='zeros', activation='tanh', return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D()(x)
            # drop out
            if hp.Boolean(f"dropout_{i}"):
                x = Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.9, step=0.1))(x)

        x = Dense(hp.Int("dense_unit_last", min_value=2, max_value=768, step=2))(x)

        # adding a maxpooling layer
        # flattening the output in order to apply the fully connected layer
        x = Flatten()(x)

        if hp.Boolean("dropout_last"):
            x = Dropout(rate=hp.Float("dropout_rate_last", min_value=0.1, max_value=0.9, step=0.1))(x)

        # adding softmax layer for the classification
        output_layer = Dense(2, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

class BiConvLSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_layer = Input(shape=(None, self.input_shape[0], self.input_shape[1], 1))
        x = input_layer
        for i in range(hp.Int("num_layers", 1, 3)):
            filter_kernel = hp.Int(f"kernel_{i}_1", min_value=1, max_value=6, step=2)
            x = Bidirectional(ConvLSTM2D(hp.Int(f"units_{i}", min_value=2, max_value=300, step=10),
                                         (filter_kernel, filter_kernel), padding='same',
                                         return_sequences=True))(x)
            x = BatchNormalization()(x)
            pool_kernel = hp.Int(f"pool_kernel_{i}", min_value=1, max_value=6, step=2)
            x = TimeDistributed(MaxPooling2D(pool_size=(pool_kernel, pool_kernel), padding='same'))(x)
            if hp.Boolean(f"dropout_{i}"):
                x = Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.9, step=0.1))(x)

        x = Dense(hp.Int("dense_unit_last", min_value=2, max_value=300, step=2))(x)
        x = TimeDistributed(Flatten())(x)

        if hp.Boolean("dropout_last"):
            x = Dropout(rate=hp.Float("dropout_rate_last", min_value=0.1, max_value=0.9, step=0.1))(x)

        output_layer = Dense(2, activation="softmax")(x)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CNNLSTM2DHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_layer = Input(shape=(None, self.input_shape[0], self.input_shape[1], 1))
        x = input_layer
        for i in range(hp.Int("num_layers", 1, 3)):
            filter_kernel = hp.Int(f"kernel_{i}_1", min_value=1, max_value=8, step=2)
            x = ConvLSTM2D(filters=hp.Int(f"units_{i}", min_value=2, max_value=624, step=10),
                           kernel_size=(filter_kernel, filter_kernel),
                           bias_initializer='zeros', activation='tanh', padding='same',
                           return_sequences=True)(x)
            x = BatchNormalization()(x)
            pool_kernel = hp.Int(f"pool_kernel_{i}", min_value=1, max_value=8, step=2)
            x = TimeDistributed(MaxPooling2D(pool_size=(pool_kernel, pool_kernel),
                                             padding='same'))(x)
            if hp.Boolean(f"dropout_{i}"):
                x = Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.9, step=0.1))(x)

        x = Dense(hp.Int("dense_unit_last", min_value=2, max_value=624, step=2))(x)

        # flattening the output in order to apply the fully connected layer
        x = TimeDistributed(Flatten())(x)

        if hp.Boolean("dropout_last"):
            x = Dropout(rate=hp.Float("dropout_rate_last", min_value=0.1, max_value=0.9, step=0.1))(x)

        # adding softmax layer for the classification
        output_layer = Dense(2, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class GRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_layer = Input(self.input_shape)
        x = input_layer
        # Tune the number of Con layers.
        for i in range(hp.Int("num_layers", 1, 3)):
            x = GRU(hp.Int(f"units_{i}", min_value=2, max_value=768, step=10), return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D()(x)
            # drop out
            if hp.Boolean(f"dropout_{i}"):
                x = Dropout(rate=hp.Float(f"dropout_rate{i}", min_value=0.1, max_value=0.9, step=0.1))(x)

        x = Dense(hp.Int("dense_unit_last", min_value=2, max_value=768, step=2))(x)

        # pooling
        x = Flatten()(x)

        if hp.Boolean("dropout_last"):
            x = Dropout(rate=hp.Float("dropout_rate_last", min_value=0.1, max_value=0.9, step=0.1))(x)

        # softmax output
        output_layer = Dense(2, activation="softmax")(x)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CNN1DHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_layer = Input(self.input_shape)
        # Tune the number of Con layers.
        x = input_layer
        for i in range(hp.Int("num_layers", 1, 3)):
            x = Conv1D(filters=hp.Int(f"units_{i}", min_value=2, max_value=768, step=10),
                       kernel_size=hp.Int(f"kernel_{i}", min_value=1, max_value=15, step=2),
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D()(x)
            # drop out
            if hp.Boolean(f"dropout_{i}"):
                x = Dropout(rate=hp.Float(f"dropout_rate{i}", min_value=0.1, max_value=0.9, step=0.1))(x)

        x = Dense(hp.Int("dense_unit_last", min_value=2, max_value=768, step=2))(x)

        # pooling
        x = Flatten()(x)

        if hp.Boolean("dropout_last"):
            x = Dropout(rate=hp.Float("dropout_rate_last", min_value=0.1, max_value=0.9, step=0.1))(x)

        # softmax output
        output_layer = Dense(2, activation="softmax")(x)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class LSTM1DHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_layer = Input(shape=self.input_shape)
        x = input_layer
        for i in range(hp.Int("layers_number", 1, 3)):

            x = LSTM(units=hp.Int(f"units_{i}", min_value=2, max_value=768, step=10),
                     bias_initializer='zeros', activation='tanh', return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D()(x)
            # drop out
            if hp.Boolean(f"dropout_{i}"):
                x = Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.9, step=0.1))(x)

        x = Dense(hp.Int("dense_unit_last", min_value=2, max_value=768, step=2))(x)

        # adding a maxpooling layer
        # flattening the output in order to apply the fully connected layer
        x = Flatten()(x)

        if hp.Boolean("dropout_last"):
            x = Dropout(rate=hp.Float("dropout_rate_last", min_value=0.1, max_value=0.9, step=0.1))(x)

        # adding softmax layer for the classification
        output_layer = Dense(2, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

