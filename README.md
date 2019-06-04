This repository contains the final solution that I used for the Freesound General-Purpose Audio Tagging Challenge on Kaggle. The model achieved 8th position on the final leaderboard, with a score (MAP@3) of 0.943289.

# About the competition
The objective of the competition is to distinguish 41 different types of sounds using the provided wav files. Sounds in the dataset include things like musical instruments, human sounds, domestic sounds, and animals. Submissions were evaluated according to the Mean Average Precision @ 3 (MAP@3). For more information about the competition, refer to the [Kaggle home page for this competition.](https://www.kaggle.com/c/freesound-audio-tagging/overview)

# Acknowledgements
Thanks to [Amlan Praharaj](https://www.kaggle.com/amlanpraharaj), [Oleg Panichev](https://www.kaggle.com/opanichev) and [Aleksandrs Gehsbargs](https://www.kaggle.com/agehsbarg) for the kernels they have shared. The kernels have helped me get started on the competition. Special thanks to [@daisukelab](https://www.kaggle.com/daisukelab) for the sharing his observations about data preprocessing, augmentation and model architectures. His insights were crucial to my solution.

# Solution
Note that this document only describes the final approach. For the list of things that I tried, [please refer to this document](approaches_all.md).

##  Data preprocessing
Leading/trailing silence in the audio may not contain much information and thus not useful for the model. Hence, the very first preprocessing step is to remove this silence. `librosa.effects.trim` function was used to achieve this.

### Log Mel-Spectrograms
Often in speech recognition tasks, MFCC features are constructed from the raw audio data. Since the current data contains non-human sounds as well, using the Log Mel-Spectrogram data is better compared to the MFCC representation. Log Mel-Spectrogram for all train and test samples was pre-computed, so that compute time can be saved during training and prediction (Disk is cheaper when compared to GPU).

### Additional features
Inspired by few Kaggle kernels, summary statistics of multiple spectral and time based features were calculated. Since many of these features were correlated, these features were transformed using the Principle Component Analysis (PCA). Top 350 features were used while modeling (which amount to ~97% of the total variance).

## Architecture
The model at its core, uses the [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture with few modifications. The input Log Mel-Spec data is sent to the MobileNetV2 after first passing the input through two 2D convolution layers. This is so that the single channel input can be converted into a 3 channel input. (Thanks to the [FastAI forums for this tip](http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/12)) The output from the MobileNetV2 is then concatenated with the PCA features, and a series of Dense layers are used before the final softmax activation layer for output.

``` python
inp1 = Input(shape=(64, None, 1), name='mel')

x = BatchNormalization()(inp1)
x = Conv2D(10, kernel_size=(1, 1), padding='same', activation='relu')(x)
x = Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu')(x)

mn = MobileNetV2(include_top=False)
mn.layers.pop(0)

mn_out = mn(x)
x = GlobalAveragePooling2D()(mn_out)

inp2 = Input(shape=(350,), name='pca')
y = BatchNormalization()(inp2)

x = concatenate([x, y], axis=-1)
x = Dense(1536, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(384, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(41, activation='softmax')(x)

model = Model(inputs=[inp1, inp2], outputs=x)
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

## Train data generation with augmentation
Both the train and test audio files are of varied length. Model is designed to make use of this particular nature of the dataset. Use of Global average pooling before the Dense layers allows the model to accept inputs of various lengths. While training, at each batch generation, a random integer between the limits is chosen. 25th and 75th percentiles of train file lengths are used as min and max limits respectively. Shorter length samples than the chosen length are padded, while a random span of chosen length is extracted from longer length samples.

The common augmentation practices for Image classification such as horizontal/vertical shift, horizontal flip were used. In addition to this, Random erasing was also used. Random erasing or Cutout selects a random rectangle in the image, and replaces it with adjacent or random values. For more information about this data augmentation technique, refer to the [original paper](https://arxiv.org/abs/1708.04896).

Mixup is the final augmentation technique used while training. Mixup essentially takes pairs of data points, chosen randomly, and mixes them (both X and y) using a proportion chosen from Beta distribution.
> One intuition behind this is that by linearly interpolating between datapoints, we incentivize the network to act smoothly and kind of interpolate nicely between datapoints - without sharp transitions. (Quote from https://www.inference.vc/mixup-data-dependent-data-augmentation/)

While I haven't ran exhaustive trials to say for sure, anecdotally, each of the data augmentation have helped in improving the loss.

## Training
Ten folds (stratified split as there is class imbalance) were generated. For each fold, a model of similar architecture but that uses only Log Mel-Spectrogram data is trained. The weights from this model are loaded into the whole model (that uses both mel and pca features), and training process continues. Attempt to train the model without using this two-stage approach didn't result in as good a model as before.

## Predictions on test data
Six different lengths selected at equal intervals between the 25th and 75th percentile of train file lengths. To make use of the higher amount of information present in longer length samples, at each length, predictions are generated five times. Each time a random span of specified length is extracted from longer (than specified length) length samples. 10 (folds) x 6 (lengths) x 5 (tries) gives 300 sets of predictions for the test data. All of these predictions were combined using geometric mean, and top 3 predicted classes for each data point are selected for submission.

# Reproducing the results
Download and keep the data files in the `./data/` folder and run the bash script `./run_all.sh`. Script was tested on `Ubuntu 16.04`.
