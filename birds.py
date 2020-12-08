#%%
import gen_feature_files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa as lb
from keras import layers
import sounddevice as sd
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
#%%

ff_folder = './features/ff'
war_folder = './features/war'

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

ff_ds = init_ff_ds.shuffle(128).batch(32, drop_remainder=True)
war_ds = init_war_ds.shuffle(128).batch(32, drop_remainder=True)

features_list = [32, 64, 92, 128]
nr_of_rlayers_list = [1, 2, 3]
poolings_list = [
  [5, 2, 2, 2],
  [2, 2, 2, 5],
  [5, 4, 2],
  [10, 2, 2],
  [4, 5, 2]
]
dropout_list = [0.2, 0.3, 0.5]
#%%
results = []

for i in range(50):
  features = random.choice(features_list)
  nr_of_rlayers = random.choice(nr_of_rlayers_list)
  poolings = random.choice(poolings_list)
  c_dropout = random.choice(dropout_list)
  r_dropout = random.choice(dropout_list)
  f_dropout = random.choice([0, *dropout_list])

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

  auc = tf.keras.metrics.AUC()
  auc_name = auc.name
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
  model.save_weights('model.h5')
  hist = model.fit(
    war_ds,
    epochs=20,
    validation_data=ff_ds.take(32),
    workers=4
  )
  war_val_aucs = [val.item() for val in hist.history[auc_name]]
  model.load_weights('model.h5')
  hist = model.fit(
    ff_ds,
    epochs=20,
    validation_data=war_ds.take(32),
    workers=4
  )
  ff_val_aucs = [val.item() for val in hist.history[auc_name]]

  results.append({
    'features': features,
    'rlayers': nr_of_rlayers,
    'pooling': poolings,
    'c_dropout': c_dropout,
    'r_dropout': r_dropout,
    'f_dropout': f_dropout,
    'ff_val_aucs': ff_val_aucs,
    'war_val_aucs': war_val_aucs
  })

  with open('results.json', 'w') as outfile:
    json.dump(results, outfile)
#%%
'''
model = keras.Sequential()

model.add(layers.Conv2D(64, 5, batch_input_shape=(32, 1172, 40, 1), padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPool2D((1, 5)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, 5, padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPool2D((1, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, 5, padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPool2D((1, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, 5, padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPool2D((1, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Permute((1,3,2)))
model.add(layers.Reshape((1172, 64)))
model.add(layers.GRU(64, return_sequences=True, dropout=0.25))
model.add(layers.GRU(64, return_sequences=True, dropout=0.25))

model.add(layers.MaxPool1D(1172))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
epochs = 10
result = model.fit(
  ff_ds,
  epochs=epochs,
  validation_data=war_ds.take(32),
  workers=4,
)
print(result)

#%%
results = model.predict(np.load('test_features.npy'))
print(f'there are {len(results)} results')
print(f'with {np.mean(results>0.5) * 100:.0f}% positive ')
with open('submission.csv', 'w') as f:
  f.write('ID,Predicted\n')
  i = 0
  for r in results:
    f.write(f'{i},{r[0]}\n')
    i += 1
'''
#%%
'''
#%%
for i in range(1172):
  if i == 0:
    continue
  if 1172 % i == 0:
    print(i)
#%%
with np.load('features.npz') as data:
  ff_x = data['ff_x']
  ff_y = data['ff_y']
  war_x = data['war_x']
  war_y = data['war_y']
  length = len(ff_y) + len(war_y)

print(f'ff_x shape: {ff_x.shape}')
print(f'ff_y shape: {ff_y.shape}')
print(f'war_x shape: {war_x.shape}')
print(f'war_y shape: {war_y.shape}')
ff_ds = tf.data.Dataset.from_tensor_slices((ff_x, ff_y)) # .shuffle(100).batch(32)
war_ds = tf.data.Dataset.from_tensor_slices((war_x, war_y)) # .shuffle(100).batch(32)
combined = ff_ds.concatenate(war_ds) 
test = combined.shuffle(length).batch(32)
#%%
'''