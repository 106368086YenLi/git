import os
import re
from glob import glob
import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.io import wavfile

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout, Flatten
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import to_categorical
from keras_tqdm import TQDMNotebookCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

POSSIBLE_LABELS = 'yes no up down left right on off go stop silence unknown'.split()    # All the labels used
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
print('id2name', len(id2name), id2name)
print('name2id', len(name2id), name2id)

def load_train_data(data_dir):
    pattern = re.compile('(.+\/)?(\w+)\/([^_]+)_.+wav')
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    all_files = [re.sub(r'\\', r'/', file) for file in all_files]

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    check = np.zeros(len(POSSIBLE_LABELS))
    train, valid = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]
            check[label_id] = 1
            sample = (label, label_id, uid, entry)
            if uid in valset:
                valid.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} valid samples'.format(len(train), len(valid)))
    columns_list = ['label', 'label_id', 'user_id', 'wav_file']
    train_df = pd.DataFrame(train, columns=columns_list)
    valid_df = pd.DataFrame(valid, columns=columns_list)
    print(check)
    return train_df, valid_df

train_df, valid_df = load_train_data('./data/')
print(train_df.head())
print(train_df.label.value_counts())
silence_files = train_df[train_df.label == 'silence']
train_df      = train_df[train_df.label != 'silence']

def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])

def process_wav_file(fname):
    wav = read_wav_file(fname)

    L = 16000

    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i:(i+L)]
        j = np.random.randint(0, rem_len)
        silence_part_left = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])

    specgram = stft(wav, 16000, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None)   # NFFT process
    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))

    return np.stack([phase, amp], axis=2)

def train_generator(train_batch_size):
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n=int(2048/(len(POSSIBLE_LABELS)))))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i]))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes=len(POSSIBLE_LABELS))
            yield x_batch, y_batch

def valid_generator(valid_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), valid_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + valid_batch_size, len(ids))
            i_valid_batch = ids[start:end]
            for i in i_valid_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i]))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes=len(POSSIBLE_LABELS))
            yield x_batch, y_batch

x = Input(shape = (257,98,2))
x = BatchNormalization()(x)
num_filters = [64, 64, 128, 128, 256]
num_convs = [1, 2, 1, 1, 1]

#(257, 98, 2) => (128, 49, 64) => (64, 24, 128) => (32, 12, 128) => (16, 6, 256) => 12 labels

for i in range(5):
    for j in range(num_convs[i]):
        x = Conv2D(filters=num_filters[i], kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(len(POSSIBLE_LABELS), activation='softmax')(x)
model = Model(inputs=inp, outputs = out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           min_delta=0.01,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=0.01,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath='output/starter.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')]

history = model.fit_generator(generator=train_generator(50),
                              steps_per_epoch=120,
                              epochs=20,
                              verbose=2,
                              # callbacks=callbacks,
                              validation_data=valid_generator(64),
                              validation_steps=int(np.ceil(valid_df.shape[0]/64)))


test_paths = glob('data/test/*wav')
def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x))
            x_batch = np.array(x_batch)
            yield x_batch

predictions = model.predict_generator(test_generator(50), int(np.ceil(len(test_paths)/50)))
classes = np.argmax(predictions, axis=1)

submission = dict()

for i in range(0, 10500):
    fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
    submission[fname] = label

df = pd.DataFrame(list(submission.items()), columns=['fname', 'label'])
df = df.sort_values('fname')
df.to_csv(os.path.join('weights', '106368086_hw4.csv'), index=False)


