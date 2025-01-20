from create_process import create_dataset_preprocessing

file_folder_dir = r""
all_label_csv_dir = r""
original_image_dir = file_folder_dir + r"/all_image_raw/"  # the raw data folder
create_dataset_preprocessing(file_folder_dir, original_image_dir, all_label_csv_dir)