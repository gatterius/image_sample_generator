import tensorflow as tf
from classes.ImageSampleGenerator import ImageSampleGenerator


def main():
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    IG = ImageSampleGenerator('data/paper_photo', 'data/test_gen/', train_data, train_labels, 28, 28)
    IG.generate_aug_dataset(20, 20)


if __name__ == "__main__":
    main()

