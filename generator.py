from concurrent.futures.thread import ThreadPoolExecutor

from glob import glob
import random
import cv2
import numpy as np
import tensorflow as tf


class SimilarityJudgementModelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_paths, input_shape, batch_size, img_type):
        self.dir_paths = dir_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.img_type = img_type
        self.random_indexes = np.arange(len(self.dir_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size
        fs = []
        for i in range(start_index, start_index + self.batch_size):
            cur_dir_path = self.dir_paths[self.random_indexes[i]]
            # same case
            if np.random.uniform() > 0.5:
                fs.append(self.pool.submit(self.__load_same_dir_image, cur_dir_path))
            # different case
            else:
                fs.append(self.pool.submit(self.__load_different_dir_image, cur_dir_path))

        for f in fs:
            img_a, img_b, is_same_dir = f.result()
            img_a = cv2.resize(img_a, (self.input_shape[1], self.input_shape[0]))
            img_b = cv2.resize(img_b, (self.input_shape[1], self.input_shape[0]))

            x_a = np.asarray(img_a).reshape(self.input_shape).astype('float32') / 255.0
            x_b = np.asarray(img_b).reshape(self.input_shape).astype('float32') / 255.0
            x_zero = np.zeros(shape=self.input_shape, dtype=np.float32)
            x = np.concatenate((x_a, x_b, x_zero), axis=2)
            y = np.asarray([is_same_dir]).astype('float32')
            batch_x.append(x)
            batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)

    def __load_same_dir_image(self, dir_path):
        dir_img_paths = glob(f'{dir_path}/*.jpg')
        img_path_a = dir_img_paths[random.randrange(0, len(dir_img_paths))]
        img_path_b = dir_img_paths[random.randrange(0, len(dir_img_paths))]
        img_a = cv2.imread(img_path_a, self.img_type)
        img_b = cv2.imread(img_path_b, self.img_type)
        return img_a, img_b, 1.0

    def __load_different_dir_image(self, dir_path):
        dir_img_paths = glob(f'{dir_path}/*.jpg')
        img_path_a = dir_img_paths[random.randrange(0, len(dir_img_paths))]
        dir_name = img_path_a.replace('\\', '/').split('/')[-2]
        while True:
            random_dir_path = self.dir_paths[random.randrange(0, len(self.dir_paths))]
            random_dir_name = random_dir_path.replace('\\', '/').split('/')[-1]
            if random_dir_name != dir_name:
                break
        random_dir_img_paths = glob(f'{random_dir_path}/*.jpg')
        img_path_b = random_dir_img_paths[random.randrange(0, len(random_dir_img_paths))]
        img_a = cv2.imread(img_path_a, self.img_type)
        img_b = cv2.imread(img_path_b, self.img_type)
        return img_a, img_b, 0.0

    def __len__(self):
        return int(np.floor(len(self.dir_paths) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)
