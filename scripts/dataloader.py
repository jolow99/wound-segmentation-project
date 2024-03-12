import random 
import cv2 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import sys 
sys.path.append('..')
from utils.utils import get_png_filename_list

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        image = cv2.imread(self.image_paths[idx])
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations to the image and label
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

class DataGen:

    def __init__(self, path, split_ratio, x=224, y=224, color_space='rgb'):
        self.x = x
        self.y = y
        self.path = path
        self.color_space = color_space
        self.path_train_images = path + "train/images/"
        self.path_train_labels = path + "train/labels/"
        self.path_test_images = path + "test/images/"
        self.path_test_labels = path + "test/labels/"
        self.image_file_list = [self.path_train_images + item for item in get_png_filename_list(self.path_train_images)]
        self.label_file_list = [self.path_train_labels + item for item in get_png_filename_list(self.path_train_labels)]
        self.image_file_list[:], self.label_file_list[:] = self.shuffle_image_label_lists_together()
        self.split_index = int(split_ratio * len(self.image_file_list))
        # print('after combine',self.image_file_list, self.label_file_list)
        self.x_train_file_list = self.image_file_list[self.split_index:]
        self.y_train_file_list = self.label_file_list[self.split_index:]
        self.x_val_file_list = self.image_file_list[:self.split_index]
        self.y_val_file_list = self.label_file_list[:self.split_index]
        self.x_test_file_list = [self.path_test_images + item for item in get_png_filename_list(self.path_test_images)]
        self.y_test_file_list = [self.path_test_labels + item for item in get_png_filename_list(self.path_test_labels)]

    def shuffle_image_label_lists_together(self):
        combined = list(zip(self.image_file_list, self.label_file_list))
        random.shuffle(combined)
        return zip(*combined)

    # @staticmethod
    # def change_color_space(image, label, color_space):
    #     if color_space.lower() is 'hsi' or 'hsv':
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #         label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
    #     elif color_space.lower() is 'lab':
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #         label = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)
    #     return image, label

def create_dataloaders(data_gen, batch_size, device):
    # Assuming data_gen is an instance of the modified DataGen class or equivalent functionality
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        # Add any other transformations here (e.g., normalization, resizing)
    ])

    # Create datasets
    train_dataset = CustomImageDataset(data_gen.x_train_file_list, data_gen.y_train_file_list, transform=transform)
    valid_dataset = CustomImageDataset(data_gen.x_val_file_list, data_gen.y_val_file_list, transform=transform)
    test_dataset = CustomImageDataset(data_gen.x_test_file_list, data_gen.y_test_file_list, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
