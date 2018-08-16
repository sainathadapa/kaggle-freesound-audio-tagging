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

# Miscellaneous

## Git-LFS
Although I was initially skeptical about Git-LFS, I'm completely sold out on the idea and the tech after using it for this competition. With Git-LFS, I was able to able to easily transfer model weights and other data files between my desktop and cloud instances. I could overwrite files carefree, avoiding weird file names, knowing that I can always restore previous files using the familiar git commands.

## A note about using Mix-up
Mixup can implemented in the following fashion:

1. Take the train data {X_1, y_1}
2. Shuffle the train data order to create another set {X_2, y_2}
3. Sample values (as many as number of data points in the train set) from Beta distribution α
4. Mixup `X = α*X_1 + (1-α)*X_2`, `y = α*y_1 + (1-α)*y_2`
5. Do this for each epoch

I needed a Batch version of Mixup, one that does not assume all the training data is available while computing the Mixup. I altered the original Mixup version for batch use case, by replacing the training set with batch:

1. Take the current batch {X_b_1, y_b_1}
2. Shuffle the train data order to create another set {X_b_2, y_b_2}
3. Sample values (as many as data points in the batch) from Beta distribution α
4. Mixup `X = α*X_b_1 + (1-α)*X_b_2`, `y = α*y_b_1 + (1-α)*y_b_2`

Surprisingly, this method resulted in training loss reaching to unbelievably low values, while validation loss is similar to the values that I observed using the original Mixup algorithm. The model is being trained as before, as evidenced by validation losses being similar to original case's val losses, but training metrics are altered for some reason. My reasoning for the low training losses: After Mixup, the disimilarity between the data points in the batch is reduced, and so the training loss for each batch will also be lower than before.

Since the data is shuffled before each epoch, different sets of pairs of data points are being Mixed-up every epoch. Hence this way of doing Mixup within a batch is still valid, except for the fact that training losses and other metrics calculated on the training data should be ignored. The validation metrics are untouched, correct and can be used for training stoppage.

But if we want to use training metrics, we can resolve the issue by selecting a different set of points for Mixup (as opposed to points in the same batch).

1. At each epoch, pair each point in the training set with some random point chosen from the training set. Save this mapping: {(i_1, i_2)}
2. Take the current batch {X_b_1, y_b_1}, corresponding to indices i_b_1
3. Find the indices i_b_2 corresponding to i_b_1, using the mapping {(i_1, i_2)}. Create the batch {X_b_2, y_b_2} using the i_b_2 indices
4. Sample values (as many as data points in the batch) from Beta distribution α
5. Mixup `X = α*X_b_1 + (1-α)*X_b_2`, `y = α*y_b_1 + (1-α)*y_b_2`
