import tensorflow as tf
from classes.ImageSampleGenerator import ImageSampleGenerator
import numpy as np


def main():
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    data = np.dstack((train_data.reshape((28, 28, -1)), eval_data.reshape((28, 28, -1)))).reshape((-1, 28, 28))
    labels = np.concatenate((train_labels, eval_labels))
    IG = ImageSampleGenerator('data/paper_photo', 'data/images/', 'data/labels/', data, labels, 28, 28)
    IG.generate_dataset(1000, 30, True)


if __name__ == "__main__":
    main()

