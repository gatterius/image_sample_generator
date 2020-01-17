import tensorflow as tf
from classes.ImageSampleGenerator import ImageSampleGenerator
import numpy as np


def main():
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    # data = np.dstack((train_data.reshape((28, 28, -1)), eval_data.reshape((28, 28, -1)))).reshape((-1, 28, 28))
    # labels = np.concatenate((train_labels, eval_labels))
    IG = ImageSampleGenerator('data/paper_photo', 'data/images/', 'data/labels/', train_data, train_labels, 28, 28,
                              sample_type='jpg', invert_samples=True, annotation_mode='box',
                              aug_back=False, aug_result=True)
    IG.generate_dataset(10, 10)


if __name__ == "__main__":
    main()

