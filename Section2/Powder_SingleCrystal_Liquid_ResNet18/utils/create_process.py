import os
import csv
import random
import shutil
from shutil import copy2
from PIL import Image
import pandas as pd

class create_dataset_preprocessing:
    def __init__(self, file_folder_dir,original_image_dir, all_label_csv_dir) -> None:
        
        cropped_dir = file_folder_dir + r"/cropped/"
        if not os.path.exists(cropped_dir): os.makedirs(cropped_dir)

        cropped_resized_dir = file_folder_dir + r"/cropped_resized/"
        if not os.path.exists(cropped_resized_dir): os.makedirs(cropped_resized_dir)

        split_type_dir = file_folder_dir + r"/split_type/"
        if not os.path.exists(split_type_dir): os.makedirs(split_type_dir)

        split_set_dir = file_folder_dir + r"/dataset/"
        if not os.path.exists(split_set_dir): os.makedirs(split_set_dir)

        self.identify_and_delete_wasteimage(original_image_dir, all_label_csv_dir)
        self.crop_image(original_image_dir, cropped_dir)
        self.resize_and_pad_image(cropped_dir, cropped_resized_dir)
        self.split_type_folder(cropped_resized_dir, all_label_csv_dir, split_type_dir)
        self.data_set_split(split_type_dir, split_set_dir)
        self.create_set_labelcsv(split_set_dir, all_label_csv_dir)

    def identify_and_delete_wasteimage(self, all_image_dir, all_label_csv_dir):
        # identify and delete images not present in the CSV file
        dff = pd.read_csv(all_label_csv_dir, delimiter=',', header=None, names=['Image', 'Label'])
        csv_images = dff['Image'].tolist()
        for file_name in os.listdir(all_image_dir):
            if file_name not in csv_images:
                file_path = all_image_dir + file_name
                os.remove(file_path)
                print(f'Removed file: {file_path}')
        print(f"Image cleaning completed")
    def crop(self, lx, ly, rx, ry, input_dir, output_dir):
        image = Image.open(input_dir)
        rect = (lx, ly, rx, ry)
        crop_image = image.crop(rect)
        crop_image.save(output_dir)

    def crop_image(self, original_image_dir, cropped_dir):
        for name in os.listdir(original_image_dir):
            string_name = name[:name.find(".")]
            input_dir = os.path.join(original_image_dir, name)
            output_dir = cropped_dir + "/" + string_name + ".jpg"
            if "_1" in name:
                # lx, ly, rx ,ry = 0, 750, 3320, 2350 # vessel I (with QR code)
                lx, ly, rx, ry = 0, 750, 2000, 2350  # vessel I update (with QR code)
            elif "_2" in name:
                lx, ly, rx, ry = 2150, 650, 5470, 2250  # vessel II (without QR code)
            else:
                lx, ly, rx, ry = 1980, 760, 3600, 2630  # circle (of bottom)

            self.crop(lx, ly, rx, ry, input_dir, output_dir)
        print(f"Image cropping completed")

    def resize_and_pad(self, cropped_image_path, target_size):
        # open the image
        img = Image.open(cropped_image_path)

        # calculate the aspect ratio of the original image
        width_ratio = target_size[0] / img.width
        height_ratio = target_size[1] / img.height
        ratio = min(width_ratio, height_ratio)

        # calculate the adjusted size
        new_size = (int(img.width * ratio), int(img.height * ratio))

        # resize and pad the image
        resized_img = img.resize(new_size, Image.LANCZOS)
        padded_img = Image.new("RGB", target_size, (255, 255, 255))
        padded_img.paste(resized_img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))

        return padded_img

    def resize_and_pad_image(self, cropped_dir, cropped_resized_dir):
        for name in os.listdir(cropped_dir):
            cropped_image_path = cropped_dir + name
            output_image = self.resize_and_pad(cropped_image_path, target_size=(512, 512))
            # save the adjusted image
            out_path = cropped_resized_dir + name
            output_image.save(out_path)
        print(f"Image resizing and padding completed")

    def split_type_folder(self, cropped_resized_dir, all_label_csv_dir, split_type_dir):
        powder_folder = split_type_dir + r"/Powder/"
        if not os.path.exists(powder_folder): os.makedirs(powder_folder)

        single_crystal_folder = split_type_dir + r"/SingleCrystal/"
        if not os.path.exists(single_crystal_folder): os.makedirs(single_crystal_folder)

        liquid_folder = split_type_dir + r"/Liquid/"
        if not os.path.exists(liquid_folder): os.makedirs(liquid_folder)

        with open(all_label_csv_dir, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img_name = row[0]
                img_label = row[1]
                # categorize images based on their labels
                source_path = cropped_resized_dir + img_name

                if img_label == 'Powder':
                    dest_path = os.path.join(powder_folder, os.path.basename(img_name))
                elif img_label == 'SingleCrystal':
                    dest_path = os.path.join(single_crystal_folder, os.path.basename(img_name))
                elif img_label == 'Liquid':
                    dest_path = os.path.join(liquid_folder, os.path.basename(img_name))

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copyfile(source_path, dest_path)
        print(f"Classification labeling completed")

    def data_set_split(self, src_data_folder, target_data_folder, train_scale=0.6, val_scale=0.2, test_scale=0.2):
        '''
        read the source data folder
        create categorized folders, and split the dataset into train, validation, and test sets
        '''
        print("start dividing the dataset")
        class_names = os.listdir(src_data_folder)
        # start dividing the dataset
        split_names = ['train', 'val', 'test']
        for split_name in split_names:
            split_path = os.path.join(target_data_folder, split_name)
            if os.path.isdir(split_path):
                pass
            else:
                os.mkdir(split_path)
            # subsequently create class folders under the `split_path` directory
            for class_name in class_names:
                class_split_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_split_path):
                    pass
                else:
                    os.mkdir(class_split_path)

        # divide the dataset according to the ratios of train, valid, and test sets
        # traverse and classify
        train_set_num = 0
        val_set_num = 0
        test_set_num = 0
        for class_name in class_names:
            current_all_data = os.listdir(os.path.join(src_data_folder, class_name))
            current_data_length = len(current_all_data)
            current_data_index_list = list(range(current_data_length))
            random.shuffle(current_data_index_list)

            train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
            val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
            test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
            train_stop_flag = current_data_length * train_scale
            val_stop_flag = current_data_length * (train_scale + val_scale)
            current_idx = 0
            train_num = 0
            val_num = 0
            test_num = 0
            for i in current_data_index_list:
                src_img_path = os.path.join(src_data_folder, class_name, current_all_data[i])
                if current_idx <= train_stop_flag:
                    copy2(src_img_path, train_folder)
                    train_num = train_num + 1
                elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                    copy2(src_img_path, val_folder)
                    val_num = val_num + 1
                else:
                    copy2(src_img_path, test_folder)
                    test_num = test_num + 1

                current_idx = current_idx + 1

            train_set_num += train_num
            val_set_num += val_num
            test_set_num += test_num
            print(f"********************{class_name}*********************")
            print(f"{class_name} classes divided in the ratio of {train_scale}: {val_scale}: {test_scale}, {current_data_length} images in total")
            print(f"train set: {train_num} images")
            print(f"val set: {val_num} images")
            print(f"test set: {test_num} images")
        print("Dataset splitting completed")
        print(f"train set: {train_set_num} images, val set: {val_set_num} images, test set: {test_set_num} images")
            
    def create_set_labelcsv(self, split_set_dir, all_label_csv_dir):

        train_folder = split_set_dir + r"/train/"
        val_folder = split_set_dir + r"/val/"
        test_folder = split_set_dir + r"/test/"

        train_csv = split_set_dir + 'train.csv'
        val_csv = split_set_dir + 'val.csv'
        test_csv = split_set_dir + 'test.csv'

        # read label_cleaned.csv
        image_labels = {}
        with open(all_label_csv_dir, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                image_labels[row[0]] = row[1]

        # process the training set
        with open(train_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for type_folder in os.listdir(train_folder):
                for filename in os.listdir(train_folder + type_folder):
                    if filename in image_labels:
                        writer.writerow([filename, image_labels[filename]])

        # process the validation set
        with open(val_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for type_folder in os.listdir(val_folder):
                for filename in os.listdir(val_folder + type_folder):
                    if filename in image_labels:
                        writer.writerow([filename, image_labels[filename]])

        # process the test set
        with open(test_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for type_folder in os.listdir(test_folder):
                for filename in os.listdir(test_folder + type_folder):
                    if filename in image_labels:
                        writer.writerow([filename, image_labels[filename]])

        print(f'Dataset CSV creation completed. Files saved as {train_csv} ,  {val_csv} and {test_csv}.')




