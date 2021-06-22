import os
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import cv2
import tensorflow as tf

from generator import SimilarityJudgementModelDataGenerator
from model import Model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.decay_step = epochs / 10
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 1 and (epoch - 1) % self.decay_step == 0:
            self.lr *= 0.5
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)


class LiveLossPlot(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.keys = ['loss', 'val_loss']
        self.vs = dict()
        for key in self.keys:
            self.vs[key] = []
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    def on_epoch_end(self, epoch, logs=None):
        from matplotlib import pyplot as plt
        for key in self.keys:
            self.vs[key].append(logs[key])
        for i in range(len(self.vs)):
            plt.plot(self.vs[self.keys[i]], color=self.colors[i])
        plt.xlabel('Epoch')
        plt.legend(self.keys)
        plt.draw()
        plt.pause(1e-7)


class SimilarityJudgementModel:
    def __init__(
            self,
            train_image_path,
            input_shape,
            lr,
            epochs,
            batch_size,
            pretrained_model_path='',
            validation_image_path=''):
        self.pool = ThreadPoolExecutor(8)
        self.train_dir_paths = glob(f'{train_image_path}/*')
        self.input_shape = input_shape
        self.validation_dir_paths = []
        if validation_image_path != '':
            self.validation_dir_paths = glob(f'{validation_image_path}/*')
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_type = cv2.IMREAD_COLOR
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        self.model = Model(
            input_shape=self.input_shape)
        if pretrained_model_path != '':
            self.model.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        self.train_data_generator = SimilarityJudgementModelDataGenerator(
            dir_paths=self.train_dir_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            img_type=self.img_type)
        self.validation_data_generator = SimilarityJudgementModelDataGenerator(
            dir_paths=self.validation_dir_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            img_type=self.img_type)

    def fit(self):
        self.model.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
        self.model.model.summary()
        print(f'\ntrain on {len(self.train_dir_paths)} samples')

        if not (os.path.exists('checkpoints') and os.path.isdir('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)
        callbacks = [
            LearningRateScheduler(self.lr, self.epochs),
            LiveLossPlot(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/sj_epoch_{epoch}_val_binary_accuracy_{val_binary_accuracy:.4f}.h5',
                monitor='val_binary_accuracy',
                mode='max',
                save_best_only=True)]

        if len(self.validation_dir_paths) > 0:
            print(f'validate on {len(self.validation_dir_paths)} samples')
            self.model.model.fit(
                x=self.train_data_generator,
                validation_data=self.validation_data_generator,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        else:
            self.model.model.fit(
                x=self.train_data_generator,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        cv2.destroyAllWindows()
