#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 14:22
# @Author  : ganliang
# @File    : keras_lstm.py
# @Desc    : lstm训练
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

main_input = Input(shape=(100,), dtype="int32", name="maininput")
auxiliary_input = Input(shape=(5,), name="auxinput")

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

lstrn_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation="sigmoid", name="auxoutput")(lstrn_out)

x = keras.layers.concatenate([lstrn_out, auxiliary_output])
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)

main_output = Dense(1, activation="sigmoid", name="main_output")(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer=keras.optimizers.RMSprop(), loss="binary_crossentropy", loss_weights=[1., 0.2])

model.summary()

model.fit([[],[]], [[],[]], batch_size=64, epochs=10)
