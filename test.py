from io import open
import glob
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import model_lstm, seq_rewriter
import random

USE_CUDA = False

def findFiles(path): return glob.glob(path)

def evaluate(model, data_x, data_y, options):
    BATCH_SIZE = options['BATCH_SIZE']
    avg_acc = 0.0
    count = 0
    correct_preds = 0
    for bi in range(len(data_x)/BATCH_SIZE):
        batch_x = np.array(data_x[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
        batch_y = np.array(data_y[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
        batch_x = Variable(torch.from_numpy(batch_x))
        batch_y = Variable(torch.from_numpy(batch_y))
        if USE_CUDA:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        # print batch_x.shape
        pred_logits, _ = model(batch_x)
        _, predictions = torch.max(pred_logits, 1)
        # print predictions
        correct_preds += torch.sum((predictions == batch_y).double())
        # del 
        
        count += batch_y.size(0)



    avg_acc = correct_preds/count
    print "ACCCCCCCCCCCCCCCCCCCC", avg_acc.data[0]
    print "*********************************************************************************"

def evaluate_rewriter(lstm_model, seq_model, data_x, data_y, options):
    BATCH_SIZE = options['BATCH_SIZE']
    avg_acc = 0.0
    count = 0
    correct_preds = 0
    for bi in range(len(data_x)/BATCH_SIZE):
        batch_x = np.array(data_x[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
        batch_y = np.array(data_y[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
        batch_x = Variable(torch.from_numpy(batch_x))
        batch_y = Variable(torch.from_numpy(batch_y))
        if USE_CUDA:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        rewritten_x = seq_model(batch_x)

        # print batch_x.shape
        pred_logits, _ = lstm_model(rewritten_x)
        _, predictions = torch.max(pred_logits, 1)
        # print predictions
        correct_preds += torch.sum((predictions == batch_y).double())
        # print correct_preds
        # print correct_preds
        count += batch_y.size(0)

    del seq_model.saved_log_probs[:]
    avg_acc = correct_preds/count
    print "ACCCCCCCCCCCCCCCCCCCC", avg_acc.data[0]
    print "*********************************************************************************"


def train_seq_rewriter(data_x, data_y, val_x, val_y, options = {}):
    eps = np.finfo(np.float32).eps.item()

    seq_model = seq_rewriter.seq_rewriter({
        'vocab_size' : options['vocab_size'],
        'target_size' : options['vocab_size']
        })

    BATCH_SIZE = options['BATCH_SIZE']
    MAX_EPOCHS = options['MAX_EPOCHS']
    LR = options['LR']

    lstm_model = model_lstm.charRNN({
        'vocab_size' : options['vocab_size'],
        'hidden_size' : 200,
        'target_size' : options['target_size']
        })
    lstm_model.load_state_dict(torch.load('CKPTS/lstm_epoch_40.ckpt'))
    lstm_loss_criterion = nn.CrossEntropyLoss()

    if USE_CUDA:
        lstm_model = lstm_model.cuda()
        seq_model = seq_model.cuda()

    parameters = filter(lambda p: p.requires_grad, seq_model.parameters())
    optimizer = optim.Adam(parameters, lr=LR)

    print "Initital"
    # print_rewriting(data_x[0:10, )
    # evaluate_rewriter(lstm_model, seq_model, data_x, data_y, options)

    for epoch in range(MAX_EPOCHS):
        epch_loss = 0
        loss_avg = None
        for bi in range(len(data_x)/BATCH_SIZE):
            batch_x = np.array(data_x[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
            batch_y = np.array(data_y[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
            batch_x = Variable(torch.from_numpy(batch_x))
            batch_y = Variable(torch.from_numpy(batch_y))

            if USE_CUDA:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            rewritten_x = seq_model(batch_x)

            pred_logits, _ = lstm_model(rewritten_x)
            _, predictions = torch.max(pred_logits, 1)
            pred_correctness = (predictions == batch_y).float()
            # print pred_correctness
            pred_correctness[pred_correctness == 0.0] = -1.0
            # print "***"
            # print pred_correctness
            # print hkjh
            # rewards = (pred_correctness - pred_correctness.mean()) / (pred_correctness.std() + eps)
            rewards = pred_correctness
            

            lstm_loss = lstm_loss_criterion(pred_logits, batch_y)
            lstm_loss_np = lstm_loss.data[0]

            seq_rewriter_loss = 0
            for idx, log_prob in enumerate(seq_model.saved_log_probs):
                if (idx % (batch_x.size()[1])) < min(epoch + 1, 10):
                    # seq_rewriter_loss += log_prob * lstm_loss_np
                    # print rewards
                    seq_rewriter_loss += (-log_prob * rewards[idx/batch_x.size()[1]])

            if bi == 0:
                loss_avg = lstm_loss_np
            else:
                loss_avg -= loss_avg/100.0
                loss_avg += lstm_loss_np/100.0

            

            optimizer.zero_grad()
            seq_rewriter_loss.backward()
            optimizer.step()
            # seq_model.saved_log_probs = []
            del seq_model.saved_log_probs[:]

            if bi % 30 == 0:
                print loss_avg, bi,len(data_x)/BATCH_SIZE,  epoch, torch.sum(rewards)
                print "Rewriting"
                print_rewriting(batch_x, rewritten_x, options['idx_to_char'])


        evaluate_rewriter(lstm_model, seq_model, data_x, data_y, options)
        evaluate_rewriter(lstm_model, seq_model, val_x, val_y, options)






def print_rewriting(x, rewritten_x, idx_to_char):
    for i in range(x.size(0)):
        if USE_CUDA:
            x_row = x[i].data.cpu().numpy()
            re_x_row = rewritten_x[i].cpu().numpy()
        else:
            x_row = x[i].data.numpy()
            re_x_row = rewritten_x[i].numpy()

        original_str = ""
        rewritten_str = ""
        for j in range(len(x_row)):
            original_str += ( idx_to_char[x_row[j]] )
            rewritten_str += ( idx_to_char[re_x_row[j]] )

        print original_str, rewritten_str



def train_lstm(training_x, training_y, val_x, val_y, options = {}):
    model = model_lstm.charRNN({
        'vocab_size' : options['vocab_size'],
        'hidden_size' : 200,
        'target_size' : options['target_size']
        })

    BATCH_SIZE = options['BATCH_SIZE']
    MAX_EPOCHS = options['MAX_EPOCHS']
    LR = options['LR']
    loss_criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=LR)



    for epoch in range(MAX_EPOCHS):
        epch_loss = 0
        loss_avg = None
        for bi in range(len(training_x)/BATCH_SIZE):
            batch_x = np.array(training_x[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
            batch_y = np.array(training_y[bi * BATCH_SIZE : (bi+1) * BATCH_SIZE ], dtype = 'int64')
            batch_x = Variable(torch.from_numpy(batch_x))
            batch_y = Variable(torch.from_numpy(batch_y))

            # print batch_x.shape
            pred_logits, _ = model(batch_x)
            # _, predictions = torch.max(pred_logits, 1)
            # print predictions
            
            # break
            # print pred_logits
            loss = loss_criterion(pred_logits, batch_y)
            if loss_avg == None:
                loss_avg = loss.data[0]
            else:
                loss_avg -= loss_avg/100.0
                loss_avg += loss.data[0]/100.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if bi % 10 == 0:
                print loss_avg, bi,len(training_x)/BATCH_SIZE,  epoch

        evaluate(model, val_x, val_y, options)
        torch.save(model.state_dict(), 'CKPTS/lstm_epoch_{}.ckpt'.format(epoch))

def main():
    MAX_NAME_LENGTH = 0
    MAX_EPOCHS = 100
    BATCH_SIZE = 16
    LR = 0.01

    all_files = findFiles('data/names/*.txt')

    char_vocab = {}
    name_data = {}
    
    for file in all_files:
        with open(file) as f:
            names = f.read().split("\n")[0:-1]
            name_data[file] = names
            for name in names:
                if len(name) > MAX_NAME_LENGTH: MAX_NAME_LENGTH = len(name)
                for ch in name: char_vocab[ch] = True


    # char_vocab['end'] = True
    print MAX_NAME_LENGTH
    idx_to_char = [char for char in char_vocab]
    idx_to_char = ['end'] + idx_to_char
    char_to_idx = {idx_to_char[i]:i for i in range(len(idx_to_char))}


    class_no = 0
    training_data = []
    testing_data = []
    training_classes = []
    testing_classes = []
    for class_name in name_data:
        names = name_data[class_name]
        for name in names:
            name_np = np.zeros(MAX_NAME_LENGTH)
            for idx, ch in enumerate(name):
                name_np[idx] = char_to_idx[ch]
            
            if class_no < 13:
                training_data.append((name_np, class_no))
            else:
                testing_data.append((name_np, class_no-13))
        
        if class_no < 13:
            training_classes.append(class_name)
        else:
            testing_classes.append(class_name)
        class_no += 1

    random.shuffle(training_data)
    random.shuffle(testing_data)

    # print len(training_data), "check"
    # sdhfjkshd
    val_data = training_data[:-len(training_data)/5]
    training_data = training_data[0:(len(training_data)*4)/5]

    training_x = [tr[0] for tr in training_data]
    training_y = [tr[1] for tr in training_data]

    testing_x = [tr[0] for tr in testing_data[0:(len(testing_data)*4)/5]]
    testing_y = [tr[1] for tr in testing_data[0:(len(testing_data)*4)/5]]

    testing_val_x = [tr[0] for tr in testing_data[(len(testing_data)*4)/5 :]]
    testing_val_y = [tr[1] for tr in testing_data[(len(testing_data)*4)/5 :]]


    val_x = [tr[0] for tr in val_data]
    val_y = [tr[1] for tr in val_data]

    options = {
        'vocab_size' : len(idx_to_char),
        'target_size' : len(training_classes),
        'MAX_EPOCHS' : MAX_EPOCHS, 
        'BATCH_SIZE' : 16,
        'LR' : LR
    }
    # train_lstm(training_x, training_y, val_x, val_y, options)

    options = {
        'vocab_size' : len(idx_to_char),
        'target_size' : len(training_classes),
        'MAX_EPOCHS' : MAX_EPOCHS, 
        'BATCH_SIZE' : 4,
        'LR' : 0.01,
        'idx_to_char' : idx_to_char
    }

    train_seq_rewriter(testing_x, testing_y, testing_val_x, testing_val_y, options)

if __name__ == '__main__':
    main()




