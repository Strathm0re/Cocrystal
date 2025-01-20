import os
import torch
import torchvision
import shutil
import utils.augmentation as augmentation


class reader:
    def __init__(self, data_dir, batch_size) -> None:
        # organize dataset
        self.reorg_dataset_data(data_dir)
        self.batch_size = batch_size
        # read the dataset (include image augmentation)
        transform_train, transform_test = augmentation.agm()

        self.train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
        self.valid_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
        self.test_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)
        self.train_valid_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid'),transform=transform_train)

        self.train_iter = torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=True, drop_last=True)
        self.valid_iter = torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False, drop_last=True)
        self.test_iter = torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False, drop_last=False)
        self.train_valid_iter = torch.utils.data.DataLoader(self.train_valid_ds, batch_size, shuffle=True, drop_last=True)

        labels = self.read_csv_labels(os.path.join(data_dir, 'train.csv'))
        print('# training samples :', len(labels))
        print('# categories :', len(set(labels.values())))

        
    #-------------------------organize dataset----------------------------
    def read_csv_labels(self, fname):
        """read fname to return a filename to the label dictionary"""
        with open(fname, 'r') as f:
            # skip the header line (column names)
            # lines = f.readlines()[1:]
            lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))

    def copyfile(self, filename, target_dir):
        """copy the file to the target directory"""
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(filename, target_dir)

    def reorg_train_valid(self, data_dir):
        """integrate training and validation sets"""
        for dataset_type in ['train', 'val']:
            for class_type in os.listdir(os.path.join(data_dir, dataset_type)):
                for filename in os.listdir(os.path.join(data_dir, dataset_type, class_type)):
                    self.copyfile(os.path.join(data_dir, dataset_type, class_type, filename),
                            os.path.join(data_dir, 'train_valid', class_type))
        
        # merge train.csv and valid.csv into train_valid.csv
        train_labels = self.read_csv_labels(os.path.join(data_dir, 'train.csv'))
        valid_labels = self.read_csv_labels(os.path.join(data_dir, 'val.csv'))
        train_valid_labels = {**train_labels, **valid_labels}

        train_valid_csv_path = os.path.join(data_dir, 'train_valid.csv')
        with open(train_valid_csv_path, 'w') as f:
            for filename, label in train_valid_labels.items():
                f.write(f'{filename},{label}\n')


    def reorg_dataset_data(self, data_dir):
        self.reorg_train_valid(data_dir)
        # self.reorg_test(data_dir)
