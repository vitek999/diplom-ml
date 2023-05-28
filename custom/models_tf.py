# ---------------------------------------------------------------------------------------------------#
# File name: models_tf.py                                                                           #
# Autor: Chrissi2802                                                                                #
# Created on: 05.10.2022                                                                            #
# ---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides the models for tensorflow.
import keras
from keras.models import Model
import keras.layers as layer

output_size = 5


def mlp_net_tf(data):
    """This function creates a MLP model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape = (data.shape[-1:]))

    x = layer.Flatten()(x_input)

    x = layer.Dense(256, activation="relu")(x) #256
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)

    x = layer.Dense(128, activation="relu")(x) #128
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="MLP_NET_TF")

    return model



def mlp_net_tf_v2(data):
    """This function creates a MLP model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape = (data.shape[-2:]))

    # x_input = layer.Input(8) # -2

    x = layer.Flatten()(x_input)

    x = layer.Dense(256, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="MLP_NET_TF_V2")

    return model


def cnn_net_tf(data):
    """This function creates a CNN model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape=(data.shape[-1:]))
    # x_input = layer.Input(8) # -2

    x = layer.Conv1D(64, 10, activation="relu")(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Conv1D(128, 10, activation="relu")(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="CNN_NET_TF")

    return model

def cnn_net_tf_v2(data):
    """This function creates a CNN model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape=(data.shape[-2:]))
    # x_input = layer.Input(8) # -2

    x = layer.Conv1D(64, 10, activation="relu")(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Conv1D(128, 10, activation="relu")(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="CNN_NET_TF_v2")

    return model


def gru_net_tf(data):
    """This function creates a GRU model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape = (data.shape[-2:]))

    # x_input = layer.Input(8)  # -2

    x = layer.Bidirectional(layer.GRU(256, return_sequences=True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.GRU(128, return_sequences=True))(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="GRU_NET_TF")

    return model


def lstm_net_tf(data):
    """This function creates a LSTM model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape=(data.shape[-2:]))
    # x_input = layer.Input(shape = 8)

    # w/o bidirectional

    x = layer.Bidirectional(layer.LSTM(256, return_sequences=True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.LSTM(128, return_sequences=True))(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="LSTM_NET_TF")

    return model


def lstm_net_tf_v2(data):
    """This function creates a LSTM model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    # x_input = layer.Input(shape = (data.shape[-2:]))
    x_input = layer.Input(shape=8)

    # w/o bidirectional

    x = layer.Bidirectional(layer.LSTM(256, return_sequences=True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.LSTM(128, return_sequences=True))(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="LSTM_NET_TF")

    return model


def gru_net_big_tf(data):
    """This function creates a big GRU model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape=(data.shape[-2:]))

    x = layer.Bidirectional(layer.GRU(512, return_sequences=True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.GRU(256, return_sequences=True))(x)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.GRU(128, return_sequences=True))(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(output_size, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="GRU_NET_BIG_TF")

    return model


def conv_lstm_net_tf(data):
    """This function creates a convolutional LSTM model in TensorFlow."""
    # Input:
    # data; NumPy array, data fed into the model, here only relevant to find out the input shape
    # Output:
    # model; TensorFlow / Keras model, model for training and testing

    x_input = layer.Input(shape=(data.shape[-4:]))

    x = layer.Bidirectional(layer.ConvLSTM2D(64, (3, 3), return_sequences=True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.ConvLSTM2D(128, (3, 3), return_sequences=True))(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)

    x = layer.Dense(128, activation="relu")(x)
    x = layer.BatchNormalization()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(6, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="CONV_LSTM_NET_TF")

    return model
