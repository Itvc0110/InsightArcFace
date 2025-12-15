import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class InferenceDataset(Dataset):
    def __init__(self, input_dir):
        self.image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]  

def get_inference_dataloader(input_dir, batch_size=32, num_workers=4):
    dataset = InferenceDataset(input_dir)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)