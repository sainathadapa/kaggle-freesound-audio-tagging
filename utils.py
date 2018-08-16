import os
import keras as kr
import numpy as np
from keras import backend as ktf
from pushbullet import Pushbullet


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def mel_0_1(x):
    min_val = -90.5
    max_val = 39.21
    return (x - min_val) / (max_val - min_val)


def uni_len(x, reqlen):
    x_len = x.shape[1]
    if reqlen < x_len:
        max_offset = x_len - reqlen
        offset = np.random.randint(max_offset)
        x = x[:, offset:(reqlen+offset)]
        return x
    elif reqlen == x_len:
        return x
    else:
        total_diff = reqlen - x_len
        offset = np.random.randint(total_diff)
        left_pad = offset
        right_pad = total_diff - offset
        return np.pad(x, (
            (0, 0), (left_pad, right_pad)
        ), 'symmetric')


class CyclicLR(kr.callbacks.Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            ktf.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            ktf.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        ktf.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(ktf.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


def pushbullet_callback(this_fold):

    print('Pushbullet api key found!')
    pb = Pushbullet(os.environ['PB_API_KEY'])

    def pb_func(epoch, logs):
        pb.push_note(
            "fold: " + str(this_fold) + " epoch: " + str(epoch),
            "val_loss: " +
            str(logs['val_loss']) +
            "    val_acc: " +
            str(logs['val_acc']))

    return kr.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: pb_func(epoch, logs))
