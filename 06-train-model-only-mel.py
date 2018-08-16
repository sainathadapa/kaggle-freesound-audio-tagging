"""Train only the part of the model, that depends on the Log Mel-Spec features"""

import pickle
import sys
import os
import numpy as np
import pandas as pd
import keras as kr
from keras import backend as ktf
from tqdm import tqdm
from utils import mel_0_1, get_random_eraser, CyclicLR, pushbullet_callback
from mel_model_funcs import TrainGenerator, ValGenerator, create_mel_model

# Load data
train_metadata = pd.read_csv('./data/train.csv')
fnames_train_all = train_metadata.fname.values

mel_train_all_data = {
    fname: mel_0_1(np.load('./data/mel_spec_train/' + fname + '.npy'))
    for fname in tqdm(train_metadata.fname.values)
}

# this fold
with open('./data/folds.pkl', 'rb') as f:
    folds = pickle.load(f)
this_fold = int(sys.argv[1])
print('this fold:', this_fold)

# label text -> label id
y_train_all = train_metadata.label.tolist()
labels = list(sorted(list(set(y_train_all))))
num_classes = len(labels)
label2int = {l: i for i, l in enumerate(labels)}
int2label = {i: l for i, l in enumerate(labels)}
y_train_all_idx = [label2int[l] for l in y_train_all]
train_metadata['label_idx'] = pd.Series(y_train_all_idx,
                                        index=train_metadata.index)
# train and valid sets
train_metadata.set_index('fname', inplace=True)
fnames_train, fnames_valid = folds[this_fold]
y_train = kr.utils.to_categorical(
    train_metadata.label_idx.loc[fnames_train].values,
    num_classes)
y_valid = kr.utils.to_categorical(
    train_metadata.label_idx.loc[fnames_valid].values,
    num_classes)

print(fnames_train.shape)
print(fnames_valid.shape)
print(y_train.shape)
print(y_valid.shape)

# Instantiate train and val generators
batch_size = 64

datagen = kr.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.6,
    height_shift_range=0,
    horizontal_flip=True,
    preprocessing_function=get_random_eraser(v_l=0, v_h=1)
)

train_generator = TrainGenerator(
    fnames_train,
    y_one_hot=y_train,
    batch_size=batch_size,
    alpha=1,
    datagen=datagen,
    mel_data=mel_train_all_data)

val_generator = ValGenerator(
    fnames_valid,
    y_one_hot=y_valid,
    batch_size=batch_size,
    mel_data=mel_train_all_data)


# Define model
model = create_mel_model()
model.summary()

this_fold_dir = 'model_outs/mel_model/fold' + str(this_fold)
os.makedirs(this_fold_dir, exist_ok=True)

print('Train with CyclicLR...')
callbacks = [
    kr.callbacks.ModelCheckpoint(this_fold_dir + '/best_model_1.h5',
                                 verbose=1,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True),

    CyclicLR(base_lr=0.0001,
             max_lr=0.001,
             step_size=len(train_generator),
             mode='triangular'),

    kr.callbacks.CSVLogger(this_fold_dir + '/train.log', append=True)
]

if 'PB_API_KEY' in os.environ:
    callbacks.append(pushbullet_callback(this_fold))

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=100,
                    verbose=2,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    max_queue_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    callbacks=callbacks)


print('Fine tuning 1 with ReduceLROnPlateau...')
model.load_weights(this_fold_dir + '/best_model_1.h5')

ktf.set_value(model.optimizer.lr, 0.00001)

callbacks = [
    kr.callbacks.ModelCheckpoint(this_fold_dir + '/best_model_2.h5',
                                 verbose=1,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True),

    kr.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10),

    kr.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                   verbose=1, min_delta=0.0001, mode='min'),

    kr.callbacks.CSVLogger(this_fold_dir + '/train.log', append=True),
]

if 'PB_API_KEY' in os.environ:
    callbacks.append(pushbullet_callback(this_fold))

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=100,
                    verbose=2,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    max_queue_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    callbacks=callbacks)


print('Fine tuning 2 with ReduceLROnPlateau...')
model.load_weights(this_fold_dir + '/best_model_2.h5')

ktf.set_value(model.optimizer.lr, 0.0001)

callbacks = [
    kr.callbacks.ModelCheckpoint(this_fold_dir + '/best_model_3.h5',
                                 verbose=1,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True),

    kr.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10),

    kr.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                   verbose=1, min_delta=0.0001, mode='min'),

    kr.callbacks.CSVLogger(this_fold_dir + '/train.log', append=True),

]

if 'PB_API_KEY' in os.environ:
    callbacks.append(pushbullet_callback(this_fold))

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=100,
                    verbose=2,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    max_queue_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    callbacks=callbacks)
