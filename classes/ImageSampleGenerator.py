import cv2
import pandas as pd
from random import randint
import numpy as np
from itertools import product
import os

class ImageSampleGenerator:
    def __init__(self, back_dir, output_dir, sample_data, sample_labels, sample_x_size, sample_y_size):
        self.sample_data = sample_data
        self.sample_x_size = sample_x_size
        self.sample_y_size = sample_y_size
        self.output_dir = output_dir
        self.sample_labels = sample_labels
        back_list = []
        back_filename_list = os.listdir(back_dir)
        for back_file in back_filename_list:
            back = cv2.imread(os.path.join(back_dir, back_file), 0)
            back = back[0:back.shape[0] // self.sample_x_size * self.sample_x_size,
                   0:back.shape[1] // self.sample_y_size * self.sample_y_size]
            back_list.append(back)
        self.back_list = back_list

    # place digit on a given image part, inverting original image and placing only non-white pixels
    # (so background of digit is removed)
    def place_digit(self, back_img_part, digit_img):
        digit_img = cv2.bitwise_not(digit_img)
        non_white_pixels = np.where(digit_img != 255)
        for i in range(len(non_white_pixels[0])):
            back_img_part[non_white_pixels[0][i], non_white_pixels[1][i]] = digit_img[
                non_white_pixels[0][i], non_white_pixels[1][i]]
        return back_img_part

    def generate_random_coordinates(self, img_shape, coord_num):
        x_val = list(range(1, (img_shape[0] // self.sample_x_size)))
        y_val = list(range(1, (img_shape[1] // self.sample_y_size)))
        coord_pairs = list(product(x_val, y_val))
        chosen_coord_pairs = []
        for i in range(coord_num):
            coord_index = randint(0, len(coord_pairs) - 1)
            new_pair = coord_pairs.pop(coord_index)
            chosen_coord_pairs.append((new_pair[0] * self.sample_x_size, new_pair[1] * self.sample_y_size))
        return chosen_coord_pairs

    # place given number of digits on image using random coordinates
    def generate_photo(self, back_img, digit_list, label_list):
        coord_list = self.generate_random_coordinates(back_img.shape, len(digit_list))
        annotation_list = []
        for i in range(0, len(digit_list)):
            x = coord_list[i][0]
            y = coord_list[i][1]
            label = [label_list[i], x, y]
            back_img[x:x + self.sample_x_size, y:y + self.sample_y_size] = \
                place_digit(back_img[x:x + self.sample_x_size, y:y + self.sample_x_size], digit_list[i])
            annotation_list.append(label)
        return back_img, annotation_list

    def generate_dataset(self, img_count):
        annotation = pd.DataFrame(columns=['file', 'label', 'x_min', 'y_min'])
        for i in range(img_count):
            new_img = self.back_list[randint(0, len(self.back_list) - 1)].copy()
            sample_index_list = [randint(0, self.sample_data.shape[0]) for a in range(img_count)]
            sample_list = [self.sample_data[i] for i in sample_index_list]
            label_list = [self.sample_labels[i] for i in sample_index_list]
            gen_img, annotation_part = generate_photo(new_img, sample_list, label_list)
            filename = os.path.join(self.output_dir, f'{i}.bmp')
            annotation_part_df = pd.DataFrame({
                'file': filename,
                'label': np.uint8(np.array(annotation_part)[:, 0]),
                'x_min': np.uint8(np.array(annotation_part)[:, 1]),
                'y_min': np.uint8(np.array(annotation_part)[:, 2])})
            cv2.imwrite(filename, gen_img)
            annotation = annotation.append(annotation_part_df, ignore_index=True)
        annotation.to_csv('annotation.csv', index=False)