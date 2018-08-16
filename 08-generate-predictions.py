import numpy as np
import pandas as pd
import keras as kr
from tqdm import tqdm
from utils import uni_len, mel_0_1
from mel_and_pca_model_funcs import create_mel_and_pca_model


class TestGenerator(kr.utils.Sequence):
    def __init__(self,
                 mel_files,
                 pca_data,
                 batch_size=64,
                 mel_data=None,
                 req_mel_len=None):

        self.mel_files = mel_files
        self.batch_size = batch_size
        self.mel_data = mel_data
        self.pca_data = pca_data

        self.one_set_size = int(np.ceil(len(self.mel_files) / self.batch_size))

        self.req_mel_len = req_mel_len
        self.on_epoch_end()

    def load_one_mel(self, filename):
        x = self.mel_data[filename].copy()
        x = uni_len(x, self.req_mel_len)
        x = x[..., np.newaxis]
        return x

    def load_mels_for_batch(self, filelist):
        this_batch_data = [self.load_one_mel(x) for x in filelist]
        return np.array(this_batch_data)

    def __len__(self):
        return self.one_set_size

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        # initialize the indices
        self.indexes = np.arange(len(self.mel_files))

        # create pca array(s)
        tmp = []
        for _ in range(1):
            tmp.append(self.pca_data[self.indexes, :])
        self.pca_this_epoch = tmp

        # create mel array(s)
        tmp = []
        for one_req_len in [self.req_mel_len]:
            self.req_mel_len = one_req_len
            tmp.append(self.load_mels_for_batch([
                  self.mel_files[i] for i in np.arange(len(self.mel_files))
            ]))
        self.mel_this_epoch = tmp

    def __data_generation(self, batch_num):

        this_set = int(batch_num / self.one_set_size)
        this_index = batch_num % self.one_set_size
        this_indices = self.indexes[this_index*self.batch_size:(this_index+1)*self.batch_size]

        this_batch_mel = self.mel_this_epoch[this_set][this_indices, :, :]
        this_batch_pca = self.pca_this_epoch[this_set][this_indices, :]

        return {'mel': this_batch_mel, 'pca': this_batch_pca}


test_metadata = pd.read_csv('./data/sample_submission.csv')

# Removing null wavs
test_metadata['toremove'] = 0
test_metadata.loc[lambda x: x.fname.isin([
    '0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav'
]), 'toremove'] = 1
removed = test_metadata.loc[lambda x: x.toremove == 1, :]
test_metadata = test_metadata.loc[lambda x: x.toremove == 0, :]

# Pre-load all mel data
mel_test_all_data = {
    fname: mel_0_1(np.load('./data/mel_spec_test/' + fname + '.npy'))
    for fname in tqdm(test_metadata.fname.values)
}

# Load PCA data
pca_test = np.load('./data/test_pca_feats.npy')[:, :350]
pca_test = pd.DataFrame(pca_test,
                        index=pd.read_pickle('./data/test_tab_feats.pkl').fname.values)
pca_test = pca_test.loc[test_metadata.fname.values, :].values

# Create the DL model
model = create_mel_and_pca_model()
model.summary()

# Predictions
batch_size = 128
prediction = np.log(np.ones((9397, 41)))

for i in range(10):
    print('fold {}'.format(str(i)))
    model.load_weights('model_outs/mel_and_pca_model/fold{}/best_model_3.h5'.format(i))
    for req_mel_len in [263, 363, 463, 563, 663, 763]:
        print(req_mel_len)
        for _ in range(5):
            print(i)

            test_generator = TestGenerator(
                test_metadata.fname.values,
                pca_test,
                batch_size=batch_size,
                mel_data=mel_test_all_data,
                req_mel_len=req_mel_len)

            this_pred = model.predict_generator(
                test_generator,
                steps=len(test_generator),
                max_queue_size=1,
                workers=1,
                use_multiprocessing=False)

            prediction = prediction + np.log(this_pred + 1e-38)

            del test_generator, this_pred

print(np.min(prediction))
print(np.max(prediction))

# Geometric average of all the predictions
prediction = prediction / 300.
prediction = np.exp(prediction)

print(np.min(prediction))
print(np.max(prediction))

# Create the submission file
labels = pd.read_csv('./data/train.csv').label.tolist()
labels = list(sorted(list(set(labels))))

top_3 = np.array(labels)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]

test_metadata['label'] = pd.Series(predicted_labels, index=test_metadata.index)
submission = pd.concat([
    test_metadata.loc[:, ['fname', 'label']],
    removed.loc[:, ['fname', 'label']]
])
submission.to_csv('submission.csv', index=False)
