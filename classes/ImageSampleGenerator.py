import cv2
import pandas as pd
from random import randint
import numpy as np
from itertools import product
import os
import imgaug as ia
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
ia.seed(42)

# TODO
# Update generation so samples can have arbitrary size


class ImageSampleGenerator:
    """
    This is a class for generating images by placing a number of samples randomly on given backgrounds.

    Attributes:
        back_dir: path to the directory where background images are stored
        output_img_dir: path to the directory where generated images are placed
        output_label_dir: path to the directory where generated labels are placed
        sample_data: a set of samples to be placed on background images
        sample_labels: a set of labels of sample images
        sample_x_size: width of samples
        sample_y_size: height of samples
    """
    def __init__(self, back_dir, output_img_dir, output_label_dir,
                 sample_data, sample_labels, sample_x_size, sample_y_size, sample_type, invert_samples):
        self.aug_seq = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=True)
        self.sample_data = sample_data
        self.sample_x_size = sample_x_size
        self.sample_y_size = sample_y_size
        self.output_img_dir = output_img_dir
        self.output_label_dir = output_label_dir
        self.sample_labels = sample_labels
        self.sample_type = sample_type
        self.invert_samples = invert_samples
        back_list = []
        back_filename_list = os.listdir(back_dir)
        for back_file in back_filename_list:
            back = cv2.imread(os.path.join(back_dir, back_file), 0)
            back = back[
                   0:back.shape[0] // self.sample_x_size * self.sample_x_size,
                   0:back.shape[1] // self.sample_y_size * self.sample_y_size]
            back_list.append(back)
        self.back_list = back_list

    def place_sample(self, back_img_part, sample_img, type='jpg', invert=False):
        """
        Places digit on a given image part, inverting original image and placing only non-white pixels,
        so background of digit is removed. Two types of samples can be used: jpg with 3 channels (only non-white
        pixels will be placed on background image) and png with 4 channels (only pixels with non-zero values in
        alpha channel will be placed on background image). In png case, samples have to have only 0 or 255 in the
        alpha channel, as it is simply removed before sample being placed on the background and values in range 1-254
        can lead to sample disruptions

        Parameters:
            -back_img_part: a rectangle from background image where sample will be placed
            -sample_img: a sample to be placed on the background
            -type: format of the sample, jpg or png can be used
            -invert: if sample should be inverted before being placed

        Returns:
            -back_img_part: a rectangle from background image with non-white pixels of sample placed over it
        """
        if invert and type == 'png':
            sample_img[:, :, 0:3] = cv2.bitwise_not(sample_img[:, :, 0:3])
        elif invert and type == 'jpg':
            sample_img = cv2.bitwise_not(sample_img)
        if type == 'jpg':
            non_white_pixels = np.where(sample_img != 255)
        elif type == 'png':
            non_white_pixels = np.where(sample_img[:, :, 3] != 0)
            sample_img = sample_img[:, :, 0:3]
        else:
            print('Given sample type is not defined, use \'jpg\' or \'png\'')
            exit()
        back_img_part[non_white_pixels] = sample_img[non_white_pixels]
        return back_img_part

    def generate_random_coordinates(self, img_shape, coord_num):
        """
        Generates a given number of random coordinate pairs by splitting whole back image into squares of size equal
        to sample size and than popping random squares into resulting list

        Parameters:
            -img_shape: shape of the background image
            -coord_num: quantity of pairs to be returned

        Returns:
            -chosen_coord_pairs: generated coordinate pairs
        """
        x_val = list(range(1, (img_shape[0] // self.sample_x_size)))
        y_val = list(range(1, (img_shape[1] // self.sample_y_size)))
        coord_pairs = list(product(x_val, y_val))
        chosen_coord_pairs = []
        for i in range(coord_num):
            coord_index = randint(0, len(coord_pairs) - 1)
            new_pair = coord_pairs.pop(coord_index)
            chosen_coord_pairs.append((new_pair[0] * self.sample_x_size, new_pair[1] * self.sample_y_size))
        return chosen_coord_pairs

    def generate_image(self, back_img, sample_list, label_list):
        """
        Places given number of digits on image using random coordinates to generate a image

        Parameters:
            -back_img: background image
            -sample_list: set of samples to be placed on background image
            -label_list: labels for samples

        Returns:
            -back_img: generated image
            -annotation_list: annotation labels for placed samples
        """
        coord_list = self.generate_random_coordinates(back_img.shape, len(sample_list))
        annotation_list = []
        for i in range(0, len(sample_list)):
            x = coord_list[i][0]
            y = coord_list[i][1]
            label = [
                label_list[i],
                (x + (self.sample_x_size // 2)) / back_img.shape[0],
                (y + (self.sample_y_size // 2)) / back_img.shape[1],
                self.sample_x_size / back_img.shape[0],
                self.sample_y_size / back_img.shape[1],
            ]
            back_img[x:x + self.sample_x_size, y:y + self.sample_y_size] = \
                self.place_sample(back_img[x:x + self.sample_x_size, y:y + self.sample_x_size],
                                  sample_list[i], self.sample_type, self.invert_samples)
            annotation_list.append(label)
        return back_img, annotation_list

    def aug_image(self, image):
        return self.aug_seq(images=image)

    def generate_dataset(self, img_count, sample_count, aug=True):
        """
        Generates a set of images of given size

        Parameters:
            -img_count: number of images to be generated
            -sample_count: number of samples to be placed on each image
            -aug: if generated images should be augmented

        Returns:
            nothing, generated images are written down into output directory
        """
        filename_list = []
        for i in tqdm(range(img_count)):
            new_img = self.back_list[randint(0, len(self.back_list) - 1)].copy()
            sample_index_list = [randint(0, self.sample_data.shape[0]-1) for a in range(sample_count)]
            sample_list = [self.sample_data[i] for i in sample_index_list]
            label_list = [self.sample_labels[i] for i in sample_index_list]
            gen_img, annotation_part = self.generate_image(new_img, sample_list, label_list)
            if aug:
                gen_img = self.aug_image(gen_img)
            filename = os.path.join(self.output_img_dir, f'_{i}.jpg')
            label_filename = os.path.join(self.output_label_dir, f'_{i}.txt')
            filename_list.append(filename)
            annotation_part = np.array(annotation_part)
            annotation_part_df = pd.DataFrame({
                'label': np.uint8(annotation_part[:, 0]),
                'x_center': np.around(annotation_part[:, 1], 5),
                'y_center': np.around(annotation_part[:, 2], 5),
                'width': np.around(annotation_part[:, 3], 5),
                'height': np.around(annotation_part[:, 4], 5),
            })
            annotation_part_df.to_csv(label_filename, index=False, header=False, sep=' ')
            cv2.imwrite(filename, gen_img)

        train_list, test_list = train_test_split(filename_list, test_size=0.33, random_state=42)
        train_out = ''
        test_out = ''
        for file in train_list:
            train_out += str(file) + '\n'
        for file in test_list:
            test_out += str(file) + '\n'

        text_file = open("data/train.txt", "w")
        res = text_file.write(train_out)
        text_file.close()
        text_file = open("data/test.txt", "w")
        res = text_file.write(test_out)
        text_file.close()


