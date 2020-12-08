#%%
import librosa as lb
from IPython.display import clear_output
import numpy as np
import csv
import os
from time import time

TARGET_LENGTH = 600000
TARGET_FS = 44100
SPEC_WIDTH = 1172

def generate_data(folderpath, class_dict, dirname):
  for dir, _, files in os.walk(folderpath):
    for file in files:
      if file[-4:] == '.wav':
        audio, fs = lb.load(os.path.join(dir, file), sr=None)
        if len(audio) >= TARGET_LENGTH:
          continue
        if fs != TARGET_FS:
          audio = lb.resample(audio, fs, TARGET_FS)
        np.savez(
          f'features/{dirname}/{file[:-4]}',
          x=process_file(audio),
          y=class_dict[file[:-4]]
        )

def write_confident_testfiles():
  vals = []
  with open('submission71.csv', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      confidence = float(row['Predicted'])
      id = row['ID']
      if confidence > 0.95 or confidence < 0.1133:
        if confidence > 0.9:
          val = 1
        else:
          val = 0
        audio = np.load(f'./audio/{id}.npy')
        audio = lb.resample(audio, 48000, TARGET_FS)
        np.savez(
          f'features/extra_spice/{id}',
          x=process_file(audio),
          y=val
        )
        vals.append(val)
  vals = np.array(vals)
  print(np.mean(vals==1))

def process_file(audio):
  audio = audio * 1/np.max(np.abs(audio))
  audio = lb.util.pad_center(audio, TARGET_LENGTH)
  kwargs_for_mel = {'n_mels': 40}
  audio = lb.feature.melspectrogram(
    y=audio, 
    sr=TARGET_FS, 
    n_fft=1024, 
    hop_length=512, 
    power=1.0,
    **kwargs_for_mel)
  audio = lb.core.amplitude_to_db(audio.T)
  audio = lb.util.normalize(audio)
  return audio[:,:, None]


def make_train_files():
  war_folder = './warblrb10k_public_wav'
  ff_folder = './ff1010bird_wav'

  ff_classes = dict()
  war_classes = dict()
  with open('ff1010bird_metadata_2018.csv', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      ff_classes[row['itemid']] = int(row['hasbird'])
  with open('warblrb10k_public_metadata_2018.csv', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      war_classes[row['itemid']] = int(row['hasbird'])
  print('classes read')
  generate_data(ff_folder, ff_classes, 'ff')
  print('ff files done')
  generate_data(war_folder, war_classes, 'war')
  print('war files done')

def make_gen(folder):
  def gen():
    for dir, _, files in os.walk(folder):
      for file in files:
        file = np.load(os.path.join(dir, file))
        yield file['x'], file['y']
  return gen

def make_test_files():
  folder = './audio'
  audios = np.zeros((4512, 1172, 40, 1))
  origin_fs = 48000
  f = 0
  indexes = []
  for dir, _, files in os.walk(folder):
    for file in files:
      audio = np.load(os.path.join(dir, file))
      if len(audio) != 480000:
        print('sos')

      audio = lb.resample(audio, origin_fs, TARGET_FS)

      index = int(file[:-4])
      indexes.append(index)
      audios[index] = process_file(audio)

      clear_output(True)
      print(f' file {f}/{len(files)} done')
      f += 1
  for i in range(4512):
    if i not in indexes:
      print(f'BIG ERROR ::::: {i}')
  np.save(
    'test_features',
    np.array(audios)
  )
#%%
if __name__ == '__main__':
  make_train_files()
  make_test_files()