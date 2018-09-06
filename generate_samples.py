import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model_classifier, seq_rewriter_gumbel
import argparse
import datasets
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers import EarlyStopping
from torch.utils.data import DataLoader
import os
import json
import time

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Output filename')
    parser.add_argument('--temp_min', type=float, default=0.01,
                        help='Temp Min')
    parser.add_argument('--epochs_to_anneal', type=float, default=15.0,
                        help='Epoch Number upto which length will be progressed to full length')
    parser.add_argument('--temp_max', type=float, default=2.0,
                        help='Temp Max')
    parser.add_argument('--reg', type=float, default=0.01,
                        help='Output filename')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Output filename')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Max Epochs')
    parser.add_argument('--log_every_batch', type=int, default=50,
                        help='Log every batch')
    parser.add_argument('--save_ckpt_every', type=int, default=20,
                        help='Save Checkpoint Every')
    parser.add_argument('--dataset', type=str, default="QuestionLabels",
                        help='Output filename')
    parser.add_argument('--base_dataset', type=str, default="Names",
                        help='Output filename')
    parser.add_argument('--checkpoints_directory', type=str, default="CKPTS",
                        help='Check Points Directory')
    parser.add_argument('--adv_directory', type=str, default="ADVERSARIAL_GUMBEL",
                        help='Check Points Directory')
    parser.add_argument('--continue_training', type=str, default="False",
                        help='Continue Training')
    parser.add_argument('--filter_width', type=int, default=5,
                        help='Filter Width')
    parser.add_argument('--hidden_units', type=int, default=256,
                        help='hidden_units')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='embedding_size')
    parser.add_argument('--resume_run', type=int, default=-1,
                        help='Which run to resume')
    parser.add_argument('--random_network', type=str, default="False",
                        help='Random Network')
    parser.add_argument('--classifier_type', type=str, default="charRNN",
                        help='rnn type')
    parser.add_argument('--print_prob', type=str, default="False",
                        help='Probs')
    parser.add_argument('--progressive', type=str, default="True",
                        help='Progressively increase length for back prop')
    

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    base_train_dataset = datasets.get_dataset(args.base_dataset, dataset_type = 'train')

    
    train_dataset = datasets.get_dataset(args.dataset, dataset_type = 'train')
    val_dataset = datasets.get_dataset(args.dataset, dataset_type = 'val')

    if args.classifier_type == "charRNN":
        lstm_model = model_classifier.uniRNN({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })
        print "char RNN"

    if args.classifier_type == "biRNN":
        lstm_model = model_classifier.biRNN({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })
        print "BI RNN"

    if args.classifier_type == "CNN":
        lstm_model = model_classifier.CnnTextClassifier({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })
        print "CnnTextClassifier"

    lstm_ckpt_dir = "{}/{}_classifer_{}".format(args.checkpoints_directory, args.base_dataset, args.classifier_type)
    lstm_ckpt_name = "{}/best_model.pth".format(lstm_ckpt_dir)
    if args.random_network != "True":
        lstm_model.load_state_dict(torch.load(lstm_ckpt_name))
    else:
        print "Random LSTM network.."
    lstm_model.eval()
    lstm_loss_criterion = nn.CrossEntropyLoss()

    seq_model = seq_rewriter_gumbel.seq_rewriter({
        'vocab_size' : len(train_dataset.idx_to_char),
        'target_size' : len(base_train_dataset.idx_to_char),
        'filter_width' : args.filter_width,
        'target_sequence_length' : base_train_dataset.seq_length
    })

    new_classifier = nn.Sequential(seq_model, lstm_model)

    lstm_model.to(device)
    seq_model.to(device)
    new_classifier.to(device)

    parameters = filter(lambda p: p.requires_grad, seq_model.parameters())
    for parameter in parameters:
        print "PARAMETERS", parameter.size()

    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    evaluator = create_supervised_evaluator(new_classifier,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                        })
    
    # CHECKPOINT DIRECTORY STUFF.......
    checkpoints_dir = "{}/{}".format(args.checkpoints_directory, args.adv_directory)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_suffix = "lr_{}_tmin_{}_fw_{}_bs_{}_rand_{}_classifer_{}".format(args.learning_rate, args.temp_min, args.filter_width, 
        args.batch_size, args.random_network,args.classifier_type)

    checkpoints_dir = "{}/{}_adversarial_base_{}_{}".format(checkpoints_dir, args.dataset, 
        args.base_dataset, checkpoint_suffix)
    
    
    if args.resume_run == -1:
        run_index = len(os.listdir(checkpoints_dir)) - 1
        print "CHeck ", run_index
    else:
        run_index = args.resume_run
    checkpoints_dir = "{}/{}".format(checkpoints_dir, run_index)
    if not os.path.exists(checkpoints_dir):
        print checkpoints_dir
        raise Exception("Coud not find checkpoints_dir")

    with open("{}/training_log.json".format(checkpoints_dir)) as tlog_f:
        print "CHECKSSSSSS"
        training_log = json.load(tlog_f)

    seq_model.load_state_dict(torch.load("{}/best_model.pth".format(checkpoints_dir)))
        # running_reward = training_log['running_reward'][-1]
    seq_model.eval()
    lstm_model.eval()
    new_classifier.eval()
    

    for batch_idx, batch in enumerate(val_loader):
        original_sentences = batch_to_sentenes(batch[0], val_dataset.idx_to_char )
        rewritten_x = seq_model(batch[0], temp = 1.0)
        new_sentences = batch_to_sentenes(rewritten_x, base_train_dataset.idx_to_char, spaces = True )

        pred_logits = lstm_model(seq_model.probs)
        _, predictions = torch.max(pred_logits, 1)

        results = []
        for i in range(batch[0].size()[0]):
            print "ORIG", original_sentences[i]
            print "REWR", new_sentences[i]
            print "CLAS", base_train_dataset.classes[int(predictions[i])]
            print "MAPP", val_dataset.classes[ int(predictions[i])]
            print "TARG", val_dataset.classes[int(batch[1][i])]
            print "***************"

        


def batch_to_sentenes(batch, idx_to_char, spaces = False):
    sentences = []
    for sen_no in range(batch.size()[0]):
        sent = ""
        for char_no in range(batch.size()[1]):
            if idx_to_char[batch[sen_no][char_no]] == "end":
                continue
            if not spaces:
                sent += str(idx_to_char[batch[sen_no][char_no]])
            else:
                sent += " " + str(idx_to_char[batch[sen_no][char_no]])
        sentences.append(sent)

    return sentences


if __name__ == '__main__':
    main()