import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
from torch.autograd import Variable
import os
import re

class NamesTrainingData(Dataset):

    """Face Landmarks dataset."""
    def findFiles(self, path):
        return glob.glob(path)

    def __init__(self, file_paths = 'data/names/*.txt', dataset_type = 'train', split = 0.80):
        all_files = self.findFiles(file_paths)

        files = ['data/names/Czech.txt', 'data/names/German.txt', 'data/names/Arabic.txt', 
        'data/names/Japanese.txt', 'data/names/Chinese.txt', 
        'data/names/Vietnamese.txt', 'data/names/Russian.txt', 
        'data/names/French.txt', 'data/names/Irish.txt', 
        'data/names/English.txt', 'data/names/Spanish.txt', 
        'data/names/Greek.txt', 'data/names/Italian.txt', 'data/names/Portuguese.txt', 'data/names/Scottish.txt', 
        'data/names/Dutch.txt', 'data/names/Korean.txt', 'data/names/Polish.txt']
        # if 'test' in dataset_type: ######### COMMENTED so we can train on entire names dataset ############
        #     files = ['data/names/Portuguese.txt', 'data/names/Scottish.txt', 
        #     'data/names/Dutch.txt', 'data/names/Korean.txt', 'data/names/Polish.txt']

        print files

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
        idx_to_char.sort()
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
        self.seq_length = MAX_NAME_LENGTH
        print len(self.x), len(self.y)
        # print self.y
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # sample = {'x': self.x[idx], 'y': self.y[idx]}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy( self.x[idx]).to(device), torch.from_numpy(self.y[idx:idx+1]).to(device)[0]

class SubNamesTrainingData(Dataset):

    """Face Landmarks dataset."""
    def findFiles(self, path):
        return glob.glob(path)

    def __init__(self, file_paths = 'data/names/*.txt', dataset_type = 'train', split = 0.80):
        all_files = self.findFiles(file_paths)

        
        files = ['data/names/Portuguese.txt', 'data/names/Scottish.txt', 
        'data/names/Dutch.txt', 'data/names/Korean.txt', 'data/names/Polish.txt']

        print files

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
        idx_to_char.sort()
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
        self.seq_length = MAX_NAME_LENGTH
        print len(self.x), len(self.y)
        # print self.y
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # sample = {'x': self.x[idx], 'y': self.y[idx]}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy( self.x[idx]).to(device), torch.from_numpy(self.y[idx:idx+1]).to(device)[0]
        

class QuestionLabels(Dataset):
    """Face Landmarks dataset."""
    def findFiles(self, path): return glob.glob(path)

    def __init__(self, file_path = 'data/train_5500.label.txt', dataset_type = 'train', split = 0.80):
        """
        Args:
            
        """
        with open(file_path) as f:
            lines = f.read().split("\n")
        
        class_names = []
        tokenized_lines = []
        vocab_count = {}
        class_count = {}
        MAX_LINE_LENGTH = 0
        for line in lines:
            if len(line) > 1:
                colon_index = line.find(":")
                class_name = line[:colon_index]
                line_words = line[colon_index + 1:].split()
                if len(line_words) > MAX_LINE_LENGTH:
                    MAX_LINE_LENGTH = len(line_words)
                tokenized_lines.append((line_words, class_name))
                for word in line_words:
                    if word in vocab_count:
                        vocab_count[word] += 1
                    else:
                        vocab_count[word] = 0
                if class_name not in class_names:
                    class_names.append(class_name)
                    class_count[class_name] = 1
                else:
                    class_count[class_name] += 1

        new_vocab = {word : vocab_count[word] for word in vocab_count if vocab_count[word] > 3}
        vocab_count_pairs = [(-new_vocab[word], word) for word in new_vocab]
        vocab_count_pairs.sort()
        idx_to_char = [pair[1] for pair in vocab_count_pairs]
        idx_to_char = ["<END>", "<UNK>"] + idx_to_char
        char_to_idx = {idx_to_char[i]:i for i in range(len(idx_to_char))}

        class_names.sort()
        class_to_idx = {class_name : idx for idx, class_name in enumerate(class_names)}

        x = []
        y = []

        val_split_idx = int(split * len(tokenized_lines))
        if not "val" in dataset_type:
            tokenized_lines = tokenized_lines[:val_split_idx]
            print "Train Split"
        else:
            tokenized_lines = tokenized_lines[val_split_idx:]
            print "Val split"

        for tokenized_line in tokenized_lines:
            word_list = tokenized_line[0]
            line_np = np.zeros(MAX_LINE_LENGTH)
            for widx, word in enumerate(word_list[:MAX_LINE_LENGTH]):
                if word in char_to_idx:
                    line_np[widx] = char_to_idx[word]
                else:
                    line_np[widx] = char_to_idx["<UNK>"]

            x.append(line_np)
            y.append(class_to_idx[tokenized_line[1]])


        self.x = np.array(x, dtype = 'int64')
        self.y = np.array(y, dtype = 'int64')
        # print self.y



        # for widx in range(len(self.x[5])):
        #     print idx_to_char[self.x[5][widx]]

        # print self.y[5], class_names[self.y[5]], class_to_idx[class_names[self.y[5]]]

        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.classes = class_names
        self.seq_length = MAX_LINE_LENGTH

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # sample = {'x': self.x[idx], 'y': self.y[idx]}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy( self.x[idx]).to(device), torch.from_numpy(self.y[idx:idx+1]).to(device)[0]

class TwitterArabic(Dataset):
    def findFiles(self, path): return glob.glob(path)

    def __init__(self, dir_path = 'data/Twitter', dataset_type = 'train', split = 0.80):
        
        all_tweets = []
        

        for idx in range(1000):
            with open(os.path.join( os.path.join(dir_path, "Positive"), "positive{}.txt".format(idx+1) )) as f:
                all_tweets.append((f.read().split(), 0))
            with open(os.path.join( os.path.join(dir_path, "Negative"), "negative{}.txt".format(idx+1) )) as f:
                all_tweets.append((f.read().split(), 1))
        
        vocab_count = {}
        MAX_LINE_LENGTH = 0
        val_split_idx = int(split * len(all_tweets))
        for tweet in all_tweets[:val_split_idx]:
            for word in tweet[0]:
                if word in vocab_count:
                    vocab_count[word] += 1
                else:
                    vocab_count[word] = 0
            if len(tweet[0]) > MAX_LINE_LENGTH:
                MAX_LINE_LENGTH = len(tweet[0])
        
        new_vocab = {word : vocab_count[word] for word in vocab_count if vocab_count[word] > 1}
        vocab_count_pairs = [(-new_vocab[word], word) for word in new_vocab]
        vocab_count_pairs.sort()
        idx_to_char = [pair[1] for pair in vocab_count_pairs]
        idx_to_char = ["<END>", "<UNK>"] + idx_to_char
        char_to_idx = {idx_to_char[i]:i for i in range(len(idx_to_char))}
        # print MAX_LINE_LENGTH
        
        if not "val" in dataset_type:
            all_tweets = all_tweets[:val_split_idx]
            print "Train Split"
        else:
            all_tweets = all_tweets[val_split_idx:]
            print "Val split"


        MAX_LINE_LENGTH = min(MAX_LINE_LENGTH, 40)
        x = []
        y = []
        for tweet in all_tweets:
            word_list = tweet[0]
            word_list.reverse()
            line_np = np.zeros(MAX_LINE_LENGTH)
            for widx, word in enumerate(word_list[:MAX_LINE_LENGTH]):
                if word in char_to_idx:
                    line_np[widx] = char_to_idx[word]
                else:
                    line_np[widx] = char_to_idx["<UNK>"]
            
            x.append(line_np)
            y.append(tweet[1])

        self.x = np.array(x, dtype = 'int64')
        self.y = np.array(y, dtype = 'int64')
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.classes = ["positive", "negative"]
        self.seq_length = MAX_LINE_LENGTH

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # sample = {'x': self.x[idx], 'y': self.y[idx]}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy( self.x[idx]).to(device), torch.from_numpy(self.y[idx:idx+1]).to(device)[0]



class IMDB(Dataset):
    def findFiles(self, path): 
        #print glob.glob(path)
        return glob.glob(path)

    def normalizeString(self, s):
        s = s.lower().strip()
        s = re.sub(r"<br />",r" ",s)
        s = re.sub(r'(\W)(?=\1)', '', s)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        
        return s

    def __init__(self, dir_path = 'data/aclImdb/train', dataset_type = 'train', split = 0.8):
        
        all_sentiments = []
        #print self.findFiles(dir_path + "/pos/*.txt")
        positive_files = self.findFiles(dir_path + "/pos/*.txt")
        negative_files = self.findFiles(dir_path + "/neg/*.txt")

        for file in positive_files:
            with open(file) as f:
                all_sentiments.append(( self.normalizeString(f.read()).strip().split(' '), 0))

        for file in negative_files:
            with open(file) as f:
                all_sentiments.append(( self.normalizeString(f.read()).strip().split(' '), 1))

        #print all_sentiments
        random.shuffle(all_sentiments)

        all_sentiments_test = []
        positive_test_files = self.findFiles('data/aclImdb/test' + "/pos/*.txt")
        negative_test_files = self.findFiles('data/aclImdb/test' + "/neg/*.txt")

        for file in positive_test_files:
            with open(file) as f:
                all_sentiments_test.append(( self.normalizeString(f.read()).strip().split(' '), 0))

        for file in negative_test_files:
            with open(file) as f:
                all_sentiments_test.append(( self.normalizeString(f.read()).strip().split(' '), 1))

        random.shuffle(all_sentiments_test)
        vocab_count = {}
        MAX_LINE_LENGTH = 0
        AVG_LINE_LENGTH = 0.0
        
        
        
        for sentiment in all_sentiments:
            for word in sentiment[0]:
                if word in vocab_count:
                    vocab_count[word] += 1
                else:
                    vocab_count[word] = 0
            AVG_LINE_LENGTH += len(sentiment[0])
            if len(sentiment[0]) > MAX_LINE_LENGTH:
                MAX_LINE_LENGTH = len(sentiment[0])

        AVG_LINE_LENGTH = AVG_LINE_LENGTH/len(all_sentiments)
        print MAX_LINE_LENGTH, AVG_LINE_LENGTH
        print len(vocab_count)
        
        
        vocab_count_pairs = [(-vocab_count[word], word) for word in vocab_count]
        vocab_count_pairs.sort()
        vocab_count_pairs = vocab_count_pairs[0:10000]
        idx_to_char = [pair[1] for pair in vocab_count_pairs]
        idx_to_char = ["<END>", "<UNK>"] + idx_to_char
        char_to_idx = {idx_to_char[i]:i for i in range(len(idx_to_char))}
        # print MAX_LINE_LENGTH
        
        val_split_idx = len(all_sentiments)
        all_sentiments = all_sentiments + all_sentiments_test
        if not "val" in dataset_type:
            all_sentiments = all_sentiments[:val_split_idx]
            print "Training Length", len(all_sentiments)
            print "Train Split"
        else:
            all_sentiments = all_sentiments[val_split_idx:]
            print "Val split"


        MAX_LINE_LENGTH = min(MAX_LINE_LENGTH, 500)
        x = []
        y = []
        for sentiment in all_sentiments:
            word_list = sentiment[0]
            line_np = np.zeros(MAX_LINE_LENGTH)
            for widx, word in enumerate(word_list[:MAX_LINE_LENGTH]):
                if word in char_to_idx:
                    line_np[widx] = char_to_idx[word]
                else:
                    line_np[widx] = char_to_idx["<UNK>"]
            
            x.append(line_np)
            y.append(sentiment[1])

        self.x = np.array(x, dtype = 'int64')
        self.y = np.array(y, dtype = 'int64')
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.classes = ["positive", "negative"]
        self.seq_length = MAX_LINE_LENGTH

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
    #     # sample = {'x': self.x[idx], 'y': self.y[idx]}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy( self.x[idx]).to(device), torch.from_numpy(self.y[idx:idx+1]).to(device)[0]


def get_dataset(dataset_name, dataset_type):
    if dataset_name == "Names":
        return NamesTrainingData(dataset_type = dataset_type)
    elif dataset_name == "SubNames":
        return SubNamesTrainingData(dataset_type = dataset_type)
    elif dataset_name == "QuestionLabels":
        return QuestionLabels(dataset_type = dataset_type)
    elif dataset_name == "TwitterArabic":
        return TwitterArabic(dataset_type = dataset_type)
    elif dataset_name == "IMDB":
        return IMDB(dataset_type = dataset_type)

def main():
    ndset = IMDB()

if __name__ == '__main__':
    main()


