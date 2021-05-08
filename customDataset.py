from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch
import numpy as np

class CustomDataset(Dataset):
    """Custom Dataset for loading IMDB-WIKI face images"""

    def __init__(self, csv_path, img_dir, bins_inner_edges, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['img_paths'].values
        self.ages = df['ages'].values
        self.genders = df['genders'].values
        self.transform = transform
        self.M = bins_inner_edges.shape[0]
        self.L = bins_inner_edges.shape[1] + 1
        self.bins_inner_edges = bins_inner_edges

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        if self.transform is not None:
            img = self.transform(img)

        age_label = self.ages[index]
        age_label_matrix = torch.zeros(size=(self.M, self.L), dtype=torch.int8)

        for i in range(self.bins_inner_edges.shape[0]):
            age_label_matrix[i, np.digitize(age_label, self.bins_inner_edges[i])] = 1

        return img, age_label, age_label_matrix

    def __len__(self):
        return self.ages.shape[0]