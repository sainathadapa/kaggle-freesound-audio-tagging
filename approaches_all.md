| NB no. | Preprocessing & Augmentation                                                                                                                                                                                                                                                                                                                                | Model                                                                    | Folds  | Val Loss | Val Acc | Kaggle MAP@3 Public / Private LB Scores |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|--------|----------|---------|-----------------------------------------|
| 2      | Randomly chosen 2s audio. Padded with zero if length less than 2s. MFCC features.                                                                                                                                                                                                                                                                           | Architecture #1                                                          | 2(/10) | 1.3163   | 0.6471  | 0.799557 / 0.745958                     |
| 3      | Same as Notebook 2. Used MelSpec instead of MFCC features.                                                                                                                                                                                                                                                                                                  | Architecture #2                                                          | 5      | 1.7410   | 0.5599  | 0.739202 / 0.715678                     |
| 5      | Randomly chosen 2s audio. Padded with zero if length less than 2s. Audio sampled at 16kHz.                                                                                                                                                                                                                                                                  | Architecture #3                                                          | 10     | 1.4575   | 0.5876  | 0.786267 / 0.776623                     |
| 6      | Same as NB 5                                                                                                                                                                                                                                                                                                                                                | Architecture from NB 5 (#3). Used balanced class weights while training. | 10     | 1.3862   | 0.6111  | 0.767995 / 0.772773                     |
| 9      | Created a tabular feature set data, by computing summary metrics for MFCC, Chromagra, Spectral centroid, Tonal Centroid features, etc.                                                                                                                                                                                                                      | LightGBM model                                                           | 10     | 0.86904  |         | 0.869324 / 0.857197                     |
| 11     | Applied PCA on the tabular feature set from the above script, and took the first 350 features.                                                                                                                                                                                                                                                              | (Dense NN) Architecture #4                                               | 10     | 1.0203   | 0.7179  | 0.827242 / 0.803438                     |
| 13     | MelSpec features on Silence trimmed audio. (sr=22050, n_fft=1764, hop_length=220, n_mels=64). Take the first 401 frames. np.pad('symmetric') if length is shorter. Augmentation: MixUp(1,1), RandomEraser, ImageDataGenerator(width_shift_range=0.6, horizontal_flip=True)                                                                                  | Architecture #5                                                          | 1      | 0.9226   | 0.7806  |                                         |
| 15-01  | NB 13 except here using 501 frames instead of 401.                                                                                                                                                                                                                                                                                                          | Architecture #5                                                          | 1      | 0.9155   | 0.7658  | 0.881506 / 0.849729                     |
| 15-02  | NB 15-01                                                                                                                                                                                                                                                                                                                                                    | Architecture #5, but with Dropout layers removed (#6)                    | 1      | 0.8123   | 0.8133  |                                         |
| 15-03  | NB 15-01, but with ImageDataGenerator(width_shift_range = 0.3, horizontal_flip=True)                                                                                                                                                                                                                                                                        | Architecture #6                                                          | 1      | 0.8216   | 0.8101  |                                         |
| 17     | Take 502 MelSpec frames during which the average db is the highest. np.pad('symmetric') if length is shorter.  Augmentation: MixUp(1,1), RandomEraser, ImageDataGenerator(width_shift_range=0.3, horizontal_flip=True)                                                                                                                                      | Architecture #6                                                          | 1      | 0.7863   | 0.8165  |                                         |
| 18     | NB 15-03                                                                                                                                                                                                                                                                                                                                                    | Architecture #7                                                          | 1      | 1.2478   | 0.6741  |                                         |
| 19     | NB 15-03                                                                                                                                                                                                                                                                                                                                                    | Architecture #8 (Replaced GlobalAverage2D in #7 with Flatten)            | 1      | 0.9914   | 0.7458  |                                         |
| 20     | MelSpec features on Silence trimmed audio. (sr=22050, n_fft=1764, hop_length=220, n_mels=64). Take the first 501 frames. np.pad('symmetric') if length is shorter. Augmentation: MixUp(1,1), RandomEraser, ImageDataGenerator(width_shift_range=0.6, horizontal_flip=True) Using the top 350 PCA features like in NB 11. MixUp is not applied to this data. | Architecture #9. Loaded weights from the model in NB 15-3                | 1      | 0.7510   | 0.8207  | 0.9086387 / 0.877469                    |
| 22     | Take 502 MelSpec frames during which the average db is the highest. np.pad('constant', 0) if length is shorter. Augmentation: MixUp(1,1), RandomEraser, ImageDataGenerator(width_shift_range=0.3, horizontal_flip=True)                                                                                                                                     | Architecture #6                                                          | 1      | 0.8342   | 0.7922  |                                         |
| 23     | NB 20                                                                                                                                                                                                                                                                                                                                                       | Architecture #9, but trained with balanced class weights                 | 1      | 0.7618   | 0.8175  | 0.920819 / 0.879266                     |
| 29     | Output from Pre-trained Soundnet as features                                                                                                                                                                                                                                                                                                                | Random Forest                                                            | 1      |          | 0.55    |                                         |
| 32     | Take random 500 MelSpec frames. np.pad('symmetric') with random offsets. Augmentation: MixUp(1,1), RandomEraser, ImageDataGenerator(width_shift_range=0.6, horizontal_flip=True)                                                                                                                                                                            | Architecture #6                                                          | 1      | 0.8395   | 0.7975  |                                         |
| 33     | NB 32 but with MixUp Beta(10,1)                                                                                                                                                                                                                                                                                                                             | Architecture #6                                                          | 1      | 0.8374   | 0.8133  |                                         |
| 34     | NB 32 but with MixUp Beta(0.5,0.5)                                                                                                                                                                                                                                                                                                                          | Architecture #6                                                          | 1      | 0.792    | 0.8165  |                                         |
| 37     | NB 15-2                                                                                                                                                                                                                                                                                                                                                     | Architecture #10 (MobileNet)                                             | 1      |          |         | 0.918604 / 0.920323                     |
| 38     | NB 15-2, but with 512 frames                                                                                                                                                                                                                                                                                                                                | Architecture #10                                                         | 1      |          |         |                                         |
| 39-01  | NB 38                                                                                                                                                                                                                                                                                                                                                       | Architecture #11 (MobileNetV2)                                           | 1      |          |         | 0.937984 / 0.915447                     |
| 39-02  | NB 38                                                                                                                                                                                                                                                                                                                                                       | Architecture #11. Trained using ReduceLROnPlateau instead of CyclicLR    | 1      | 0.6920   | 0.8449  |                                         |
| 40     | NB 20, except for 512 frames instead of 501                                                                                                                                                                                                                                                                                                                 | Architecture #12                                                         | 1      | 0.5931   | 0.8766  |                                         |
| 41     | Randomized length of mel frames per epoch                                                                                                                                                                                                                                                                                                                   | Architecture #11                                                         | 1      | -        | -       | 0.939091 / 0.926353                     |
|        | Final submissions:                                                                                                                                                                                                                                                                                                                                          |                                                                          |        |          |         |                                         |
| F1     | MelSpec features on Silence trimmed audio. (sr=22050, n_fft=1764, hop_length=220, n_mels=64). Take the first 512 frames. np.pad('symmetric') if length is shorter. Augmentation: MixUp(1,1), RandomEraser, ImageDataGenerator(width_shift_range=0.6, horizontal_flip=True)                                                                                  | Architecture #11                                                         | 10     |          |         | 0.954595 / 0.937644                     |
| F2     | F1 but: Randomize the length of mel frames at each batch, and also select random spans of such length                                                                                                                                                                                                                                                       | Architecture #11                                                         | 10     |          |         | 0.963455 / 0.944187                     |
| F3     | F1 + PCA features                                                                                                                                                                                                                                                                                                                                           | Architecture #12                                                         | 10     |          |         | 0.962347 / 0.939697                     |
| F4     | F2 + PCA features                                                                                                                                                                                                                                                                                                                                           | Architecture #12                                                         | 10     |          |         | 0.966223 / 0.939183                     |
| F5     | Same as F4                                                                                                                                                                                                                                                                                                                                                  | Architecture #12 (Trained with balanced class weights)                   | 10     |          |         | 0.964562 / 0.943674                     |

# Architecture #1
``` python
inp = k.layers.Input(shape=(n_mfcc,
                            1 + int(np.floor(audio_length/512)),
                            1))
x = k.layers.BatchNormalization()(inp)

x = k.layers.Conv2D(32, (4, 10), padding='same')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.MaxPool2D()(x)

x = k.layers.Conv2D(32, (4, 10), padding='same')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.MaxPool2D()(x)

x = k.layers.Conv2D(32, (4, 10), padding='same')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.MaxPool2D()(x)

x = k.layers.Flatten()(x)

x = k.layers.Dense(64)(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.Dropout(0.5)(x)

out = k.layers.Dense(n_classes, activation='softmax')(x)

model = k.models.Model(inputs=inp, outputs=out)
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 40, 173, 1)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 40, 173, 1)        4         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 40, 173, 32)       1312      
_________________________________________________________________
batch_normalization_2 (Batch (None, 40, 173, 32)       128       
_________________________________________________________________
activation_1 (Activation)    (None, 40, 173, 32)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 20, 86, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 86, 32)        40992     
_________________________________________________________________
batch_normalization_3 (Batch (None, 20, 86, 32)        128       
_________________________________________________________________
activation_2 (Activation)    (None, 20, 86, 32)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 43, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 43, 32)        40992     
_________________________________________________________________
batch_normalization_4 (Batch (None, 10, 43, 32)        128       
_________________________________________________________________
activation_3 (Activation)    (None, 10, 43, 32)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 21, 32)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3360)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                215104    
_________________________________________________________________
batch_normalization_5 (Batch (None, 64)                256       
_________________________________________________________________
activation_4 (Activation)    (None, 64)                0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 41)                2665      
=================================================================
Total params: 301,709
Trainable params: 301,387
Non-trainable params: 322
_________________________________________________________________
```

# Architecture #2
``` python
inp = k.layers.Input(shape=(128,
                            1 + int(np.floor(audio_length/512)),
                            1))

x = k.layers.Conv2D(32, (4, 10), padding='same')(inp)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.MaxPool2D()(x)

x = k.layers.Conv2D(32, (4, 10), padding='same')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.MaxPool2D()(x)

x = k.layers.Conv2D(32, (4, 10), padding='same')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)
x = k.layers.MaxPool2D()(x)

x = k.layers.Flatten()(x)
x = k.layers.Dense(128)(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Activation('relu')(x)

out = k.layers.Dense(n_classes, activation='softmax')(x)

model = k.models.Model(inputs=inp, outputs=out)
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 128, 173, 1)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 128, 173, 32)      1312      
_________________________________________________________________
batch_normalization_1 (Batch (None, 128, 173, 32)      128       
_________________________________________________________________
activation_1 (Activation)    (None, 128, 173, 32)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 86, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 86, 32)        40992     
_________________________________________________________________
batch_normalization_2 (Batch (None, 64, 86, 32)        128       
_________________________________________________________________
activation_2 (Activation)    (None, 64, 86, 32)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 43, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 43, 32)        40992     
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 43, 32)        128       
_________________________________________________________________
activation_3 (Activation)    (None, 32, 43, 32)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 21, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 10752)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               1376384   
_________________________________________________________________
batch_normalization_4 (Batch (None, 128)               512       
_________________________________________________________________
activation_4 (Activation)    (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 41)                5289      
=================================================================
Total params: 1,465,865
Trainable params: 1,465,417
Non-trainable params: 448
_________________________________________________________________
```

# Architecture #3
``` python
inp = k.layers.Input(shape=(audio_length, 1))
x = k.layers.BatchNormalization()(inp)

x = k.layers.Conv1D(16, 9, padding='valid', activation='relu')(x)
x = k.layers.Conv1D(16, 9, padding='valid', activation='relu')(x)
x = k.layers.MaxPool1D(16)(x)
x = k.layers.Dropout(0.1)(x)

x = k.layers.Conv1D(32, 3, padding='valid', activation='relu')(x)
x = k.layers.Conv1D(32, 3, padding='valid', activation='relu')(x)
x = k.layers.MaxPool1D(4)(x)
x = k.layers.Dropout(0.1)(x)

x = k.layers.Conv1D(32, 3, padding='valid', activation='relu')(x)
x = k.layers.Conv1D(32, 3, padding='valid', activation='relu')(x)
x = k.layers.MaxPool1D(4)(x)
x = k.layers.Dropout(0.1)(x)

x = k.layers.Conv1D(256, 3, padding='valid', activation='relu')(x)
x = k.layers.Conv1D(256, 3, padding='valid', activation='relu')(x)
x = k.layers.GlobalMaxPool1D()(x)
x = k.layers.Dropout(0.2)(x)

x = k.layers.Dense(64, activation='relu')(x)
x = k.layers.Dense(1028, activation='relu')(x)

out = k.layers.Dense(n_classes, activation='softmax')(x)

model = k.models.Model(inputs=inp, outputs=out)
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 32000, 1)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 32000, 1)          4         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 31992, 16)         160       
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 31984, 16)         2320      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 1999, 16)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1999, 16)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 1997, 32)          1568      
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 1995, 32)          3104      
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 498, 32)           0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 498, 32)           0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 496, 32)           3104      
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 494, 32)           3104      
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 123, 32)           0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 123, 32)           0         
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 121, 256)          24832     
_________________________________________________________________
conv1d_8 (Conv1D)            (None, 119, 256)          196864    
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 256)               0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                16448     
_________________________________________________________________
dense_2 (Dense)              (None, 1028)              66820     
_________________________________________________________________
dense_3 (Dense)              (None, 41)                42189     
=================================================================
Total params: 360,517
Trainable params: 360,515
Non-trainable params: 2
_________________________________________________________________
```

# Architecture #4
``` python
inp = k.layers.Input(shape=(350,))
x = k.layers.BatchNormalization()(inp)

x = k.layers.Dense(256, activation='relu')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Dropout(0.6)(x)
x = k.layers.Dense(128, activation='relu')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Dropout(0.4)(x)

out = k.layers.Dense(41, activation='softmax')(x)

model = k.models.Model(inputs=inp, outputs=out)
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 350)               0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 350)               1400      
_________________________________________________________________
dense_1 (Dense)              (None, 256)               89856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
batch_normalization_3 (Batch (None, 128)               512       
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 41)                5289      
=================================================================
Total params: 130,977
Trainable params: 129,509
Non-trainable params: 1,468
_________________________________________________________________
```


# Architecture #5
``` python
model = Sequential()

model.add(Conv2D(48, 11,  input_shape=input_shape, strides=(2,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1,2)))
model.add(BatchNormalization())

model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=2))
model.add(BatchNormalization())

model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 134, 48)       5856      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 66, 48)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 66, 48)        192       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 22, 128)       153728    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 10, 128)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 7, 10, 128)        512       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 10, 192)        221376    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 10, 192)        331968    
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 7, 10, 128)        221312    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 4, 128)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 4, 128)         512       
_________________________________________________________________
flatten_1 (Flatten)          (None, 2560)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               655616    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 41)                10537     
=================================================================
Total params: 1,667,401
Trainable params: 1,666,793
Non-trainable params: 608
__________________________________
```

# Architecture #6
``` python
model = Sequential()

model.add(Conv2D(48, 11,  input_shape=input_shape, strides=(2,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1,2)))
model.add(BatchNormalization())

model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=2))
model.add(BatchNormalization())

model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 167, 48)       5856      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 83, 48)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 83, 48)        192       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 28, 128)       153728    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 13, 128)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 7, 13, 128)        512       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 13, 192)        221376    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 13, 192)        331968    
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 7, 13, 128)        221312    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 6, 128)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 6, 128)         512       
_________________________________________________________________
flatten_1 (Flatten)          (None, 3840)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               983296    
_________________________________________________________________
dense_2 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_3 (Dense)              (None, 41)                10537     
=================================================================
Total params: 1,995,081
Trainable params: 1,994,473
Non-trainable params: 608
```

# Architecture #7
``` python
model = Sequential()
model.add(Conv2D(64, (7, 3),  input_shape=input_shape, strides=(1,2), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 1), strides=(2, 1)))
model.add(BatchNormalization())
model.add(Conv2D(128, (7, 1),  strides=(1,1), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (5, 1),  input_shape=input_shape, strides=(1,1), activation='relu', padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(128, (1, 5),  input_shape=input_shape, strides=(1,1), activation='relu', padding='same'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 251, 64)       1408      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 251, 64)       0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 251, 64)       256       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 31, 251, 128)      57472     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 125, 128)      0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 125, 128)      512       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 125, 128)      82048     
_________________________________________________________________
batch_normalization_3 (Batch (None, 10, 125, 128)      512       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 125, 128)      82048     
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 128)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 128)               512       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
dropout_3 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 41)                2665      
=================================================================
Total params: 239,849
Trainable params: 238,953
Non-trainable params: 896
```

# Architecture #8
``` python
model = Sequential()
model.add(Conv2D(64, (7, 3),  input_shape=input_shape, strides=(1,2), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 1), strides=(2, 1)))
model.add(BatchNormalization())
model.add(Conv2D(128, (7, 1),  strides=(1,1), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (5, 1),  input_shape=input_shape, strides=(1,1), activation='relu', padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(128, (1, 5),  input_shape=input_shape, strides=(1,1), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 251, 64)       1408      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 251, 64)       0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 251, 64)       256       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 31, 251, 128)      57472     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 125, 128)      0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 125, 128)      512       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 125, 128)      82048     
_________________________________________________________________
batch_normalization_3 (Batch (None, 10, 125, 128)      512       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 125, 128)      82048     
_________________________________________________________________
flatten_1 (Flatten)          (None, 160000)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                10240064  
_________________________________________________________________
batch_normalization_4 (Batch (None, 64)                256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 41)                2665      
=================================================================
Total params: 10,471,401
Trainable params: 10,470,633
Non-trainable params: 768
______________________________
```

# Architecture #9
``` python
inp1 = Input(shape=input_shape, name='mel')

x = Conv2D(48, 11,  strides=(2,3), activation='relu', padding='same')(inp1)
x = MaxPooling2D(3, strides=(1,2))(x)
x = BatchNormalization()(x)

x = Conv2D(128, 5, strides=(2,3), activation='relu', padding='same')(x)
x = MaxPooling2D(3, strides=2)(x)
x = BatchNormalization()(x)

x = Conv2D(192, 3, strides=1, activation='relu', padding='same')(x)
x = Conv2D(192, 3, strides=1, activation='relu', padding='same')(x)
x = Conv2D(128, 3, strides=1, activation='relu', padding='same')(x)
x = MaxPooling2D(3, strides=(1,2))(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)

inp2 = Input(shape=(350,), name='pca')
y = BatchNormalization()(inp2)
y = Dense(256, activation='relu')(y)
y = BatchNormalization()(y)

x = concatenate([x, y], axis=-1)
x = Dropout(0.5)(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

out = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[inp1, inp2], outputs=out)
```
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
mel (InputLayer)                (None, 64, 501, 1)   0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 167, 48)  5856        mel[0][0]                        
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 30, 83, 48)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 30, 83, 48)   192         max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 15, 28, 128)  153728      batch_normalization_1[0][0]      
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 7, 13, 128)   0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 7, 13, 128)   512         max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 7, 13, 192)   221376      batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 7, 13, 192)   331968      conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 7, 13, 128)   221312      conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 5, 6, 128)    0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
pca (InputLayer)                (None, 350)          0                                            
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 5, 6, 128)    512         max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 350)          1400        pca[0][0]                        
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 3840)         0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 256)          89856       batch_normalization_4[0][0]      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          983296      flatten_1[0][0]                  
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 256)          1024        dense_2[0][0]                    
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 512)          0           dense_1[0][0]                    
                                                                 batch_normalization_5[0][0]      
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 512)          0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 256)          131328      dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 256)          0           dense_3[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 41)           10537       dropout_2[0][0]                  
==================================================================================================
Total params: 2,152,897
Trainable params: 2,151,077
Non-trainable params: 1,820
______________________________
```

# Architecture #10
``` python
mn = MobileNet(include_top=False)
mn.layers.pop(0)
inp = Input(shape=X_train.shape[1:])
x = BatchNormalization()(inp)
x = Conv2D(10, kernel_size = (1,1), padding = 'same', activation = 'relu')(x)
x = Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu')(x)
mn_out = mn(x)
x = GlobalAveragePooling2D()(mn_out)
x = Dense(1536, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(384, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(41, activation='softmax')(x)
model = Model(inputs=[inp], outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 64, 501, 1)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 501, 1)        4         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 501, 10)       20        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 501, 3)        33        
_________________________________________________________________
mobilenet_1.00_224 (Model)   multiple                  3228864   
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1536)              1574400   
_________________________________________________________________
batch_normalization_2 (Batch (None, 1536)              6144      
_________________________________________________________________
dense_2 (Dense)              (None, 384)               590208    
_________________________________________________________________
batch_normalization_3 (Batch (None, 384)               1536      
_________________________________________________________________
dense_3 (Dense)              (None, 41)                15785     
=================================================================
Total params: 5,416,994
Trainable params: 5,391,264
Non-trainable params: 25,730
```
# Architecture #11
``` python
inp = k.layers.Input(shape=(64, None, 1))
x = k.layers.BatchNormalization()(inp)
x = k.layers.Conv2D(10, kernel_size = (1,1), padding = 'same', activation = 'relu')(x)
x = k.layers.Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu')(x)
mn = k.applications.mobilenetv2.MobileNetV2(include_top=False)
mn.layers.pop(0)
mn_out = mn(x)
x = k.layers.GlobalAveragePooling2D()(mn_out)
x = k.layers.Dense(1536, activation='relu')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Dense(384, activation='relu')(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Dense(41, activation='softmax')(x)
model = k.models.Model(inputs=[inp], outputs=x)
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 64, 512, 1)        0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 64, 512, 1)        4         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 512, 10)       20        
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 64, 512, 3)        33        
_________________________________________________________________
mobilenetv2_1.00_224 (Model) multiple                  2257984   
_________________________________________________________________
global_average_pooling2d_2 ( (None, 1280)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 1536)              1967616   
_________________________________________________________________
batch_normalization_5 (Batch (None, 1536)              6144      
_________________________________________________________________
dense_5 (Dense)              (None, 384)               590208    
_________________________________________________________________
batch_normalization_6 (Batch (None, 384)               1536      
_________________________________________________________________
dense_6 (Dense)              (None, 41)                15785     
=================================================================
Total params: 4,839,330
Trainable params: 4,801,376
Non-trainable params: 37,954
```

# Architecture #12
``` python
inp1 = kr.layers.Input(shape=(64, None, 1), name='mel')
x = kr.layers.BatchNormalization()(inp1)
x = kr.layers.Conv2D(10, kernel_size=(1, 1), padding='same', activation='relu')(x)
x = kr.layers.Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu')(x)

mn = kr.applications.mobilenetv2.MobileNetV2(include_top=False)
mn.layers.pop(0)
mn_out = mn(x)
x = kr.layers.GlobalAveragePooling2D()(mn_out)

inp2 = kr.layers.Input(shape=(350,), name='pca')
y = kr.layers.BatchNormalization()(inp2)

x = kr.layers.concatenate([x, y], axis=-1)
x = kr.layers.Dense(1536, activation='relu')(x)
x = kr.layers.BatchNormalization()(x)
x = kr.layers.Dense(384, activation='relu')(x)
x = kr.layers.BatchNormalization()(x)
x = kr.layers.Dense(41, activation='softmax')(x)

model = kr.models.Model(inputs=[inp1, inp2], outputs=x)
```
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
mel (InputLayer)                (None, 64, None, 1)  0                                            
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, None, 1)  4           mel[0][0]                        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 64, None, 10) 20          batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, None, 3)  33          conv2d_1[0][0]                   
__________________________________________________________________________________________________
mobilenetv2_1.00_224 (Model)    multiple             2257984     conv2d_2[0][0]                   
__________________________________________________________________________________________________
pca (InputLayer)                (None, 350)          0                                            
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 1280)         0           mobilenetv2_1.00_224[1][0]       
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 350)          1400        pca[0][0]                        
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1630)         0           global_average_pooling2d_1[0][0]
                                                                 batch_normalization_2[0][0]      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1536)         2505216     concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 1536)         6144        dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 384)          590208      batch_normalization_3[0][0]      
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 384)          1536        dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 41)           15785       batch_normalization_4[0][0]      
==================================================================================================
Total params: 5,378,330
Trainable params: 5,339,676
Non-trainable params: 38,654
__________________________________________________________________________________________________
```
