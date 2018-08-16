"""Train the full model, after loading weights from the mel-only model from previous script"""

import os
import pickle
import sys
import pandas as pd
import numpy as np
import keras as kr
from keras import backend as ktf
from tqdm import tqdm
from utils import mel_0_1, get_random_eraser, CyclicLR, pushbullet_callback
from mel_and_pca_model_funcs import TrainGenerator, ValGenerator, create_mel_and_pca_model
from mel_model_funcs import create_mel_model

# Load train data
fnames_train_all = pd.read_pickle('./data/train_tab_feats.pkl').fname.values
pca_train_all = np.load('./data/train_pca_feats.npy')[:, :350]
pca_train_all = pd.DataFrame(pca_train_all, index=fnames_train_all)
train_metadata = pd.read_csv('./data/train.csv').set_index('fname')
train_metadata = train_metadata.loc[fnames_train_all, :]

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
fnames_train, fnames_valid = folds[this_fold]

pca_train = pca_train_all.loc[fnames_train, :].values
pca_valid = pca_train_all.loc[fnames_valid, :].values

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

# class weights because the classes are imbalanced
class_weights = train_metadata\
        .groupby('label_idx')['label']\
        .count()\
        .apply(lambda x: 1/x)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to_dict()

# Pre-load all the mel-spec data for faster computation
mel_train_all_data = {
    fname: mel_0_1(np.load('./data/mel_spec_train/' + fname + '.npy'))
    for fname in tqdm(train_metadata.index.values)
}

# Load the weights from the previous model
model = create_mel_model()
model.summary()

model.load_weights('model_outs/mel_model/fold{}/best_model_3.h5'.format(this_fold))
layers_to_copy = ['batch_normalization_1', 'conv2d_1',
                  'conv2d_2', 'global_average_pooling2d_1']
prev_model_weights = [model.get_layer(x).get_weights().copy()
                      for x in layers_to_copy]
monet_weights = [layer.get_weights().copy()
                 for layer in model.get_layer('mobilenetv2_1.00_224').layers]

del model
ktf.clear_session()

# Create the model currently to be trained
model = create_mel_and_pca_model()
model.summary()

# Set weights from previous model
for x, y in zip(layers_to_copy, prev_model_weights):
    model.get_layer(x).set_weights(y)
for i, x in enumerate(monet_weights):
    model\
        .get_layer('mobilenetv2_1.00_224')\
        .layers[i]\
        .set_weights(x)


# Instantiate train and val generators
batch_size = 64
num_classes = len(labels)

datagen = kr.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.6,
    height_shift_range=0,
    horizontal_flip=True,
    preprocessing_function=get_random_eraser(v_l=0, v_h=1))

train_generator = TrainGenerator(
    fnames_train,
    pca_train,
    y_one_hot=y_train,
    batch_size=batch_size,
    alpha=0.5,
    datagen=datagen,
    mel_data=mel_train_all_data)

val_generator = ValGenerator(
    fnames_valid,
    pca_valid,
    y_one_hot=y_valid,
    batch_size=batch_size,
    mel_data=mel_train_all_data)

this_fold_dir = './model_outs/mel_and_pca_model/fold{}'.format(this_fold)
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
