import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
from torch.autograd import Variable

class NamesTrainingData(Dataset):
    """Face Landmarks dataset."""
    def findFiles(self, path): return glob.glob(path)

    def __init__(self, file_paths = 'data/names/*.txt', dataset_type = 'train', split = 0.80):
        """
        Args:
            
        """
        all_files = self.findFiles(file_paths)
        if 'train' in dataset_type:
            files = all_files[:13]
        if 'test' in dataset_type:
            files = all_files[13:]
        char_vocab = {}
        name_data = {}
        
        MAX_NAME_LENGTH = 0
        for file in files:
            with open(file) as f:
                names = f.read().split("\n")[0:-1]
                name_data[file] = names
                for name in names:
                    if len(name) > MAX_NAME_LENGTH: MAX_NAME_LENGTH = len(name)
                    for ch in name: char_vocab[ch] = True
        
        idx_to_char = [char for char in char_vocab]
        idx_to_char = ['end'] + idx_to_char
        char_to_idx = {idx_to_char[i]:i for i in range(len(idx_to_char))}
        
        class_no = 0
        data = []
        classes = []
        for class_name in name_data:
            names = name_data[class_name]
            for name in names:
                name_np = np.zeros(MAX_NAME_LENGTH)
                for idx, ch in enumerate(name):
                    name_np[idx] = char_to_idx[ch]
                    data.append((name_np, class_no))

            classes.append(class_name)
            class_no += 1

        random.shuffle(data)

        val_split_idx = int(len(data) * split)
        print val_split_idx
        if 'val' in dataset_type:
            data = data[val_split_idx:]
        else:
            data = data[:val_split_idx]

        # print data
        self.classes = classes
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.x = np.array([row[0] for row in data],dtype = 'int64' )
        self.y = np.array([row[1] for row in data], dtype = 'int64')
        print len(self.x), len(self.y)
        # print self.y
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # sample = {'x': self.x[idx], 'y': self.y[idx]}
        return self.x[idx], self.y[idx]

def main():
    ndset = NamesTrainingData()

if __name__ == '__main__':
    main()

