#%%
import gen_feature_files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa as lb
from keras import layers

import os
import csv
import random
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from tensorflow import keras
from tensorflow.keras import layers
import json

ff_folder = './features/ff'
war_folder = './features/war'
extra_folder = './features/extra_spice'

init_ff_ds = tf.data.Dataset.from_generator(
  gen_feature_files.make_gen(ff_folder),
  output_types=(tf.float32, tf.int32),
  output_shapes=(tf.TensorShape((1172, 40, 1)), tf.TensorShape([]))
)

init_war_ds = tf.data.Dataset.from_generator(
  gen_feature_files.make_gen(war_folder),
  output_types=(tf.float32, tf.int32),
  output_shapes=(tf.TensorShape((1172, 40, 1)), tf.TensorShape([]))
)

extra_ds = tf.data.Dataset.from_generator(
  gen_feature_files.make_gen(extra_folder),
  output_types=(tf.float32, tf.int32),
  output_shapes=(tf.TensorShape((1172, 40, 1)), tf.TensorShape([]))
)

ultimate_ds = tf.data.experimental.sample_from_datasets(
  (init_ff_ds, init_war_ds, extra_ds),
  (0.44, 0.46, 0.1)
).shuffle(128).batch(32, drop_remainder=True)

features =  92
nr_of_rlayers = 2
poolings = [5, 2, 2, 2]
c_dropout = 0.2
r_dropout = 0.5
f_dropout = 0.3

model = keras.Sequential()
for i, pool in enumerate(poolings):
  if i == 0:
    model.add(layers.Conv2D(
      features,
      5,
      batch_input_shape=(32, 1172, 40, 1),
      padding='same',
      activation='relu',
      kernel_initializer='he_uniform'
      ))
  else:
    model.add(layers.Conv2D(
      features,
      5,
      padding='same',
      activation='relu',
      kernel_initializer='he_uniform'
      ))
  model.add(layers.MaxPool2D((1, pool)))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(c_dropout))

model.add(layers.Permute((1,3,2)))
model.add(layers.Reshape((1172, features)))

for n in range(nr_of_rlayers):
  model.add(layers.GRU(features, return_sequences=True, dropout=r_dropout))

model.add(layers.MaxPool1D(1172))

model.add(layers.Flatten())
model.add(layers.Dropout(f_dropout))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
#%%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
model.fit(
  ultimate_ds,
  epochs=25,
  workers=4
)
results = model.predict(np.load('test_features.npy'))
print(f'there are {len(results)} results')
print(f'with {np.mean(results>0.5) * 100:.0f}% positive ')
with open('submission.csv', 'w') as f:
  f.write('ID,Predicted\n')
  i = 0
  for r in results:
    f.write(f'{i},{r[0]}\n')
    i += 1