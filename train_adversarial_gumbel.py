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
                        help='learning_rate')
    parser.add_argument('--temp_min', type=float, default=0.01,
                        help='Temp Min')
    parser.add_argument('--epochs_to_anneal', type=float, default=15.0,
                        help='epochs_to_anneal')
    parser.add_argument('--temp_max', type=float, default=2.0,
                        help='Temp Max')
    parser.add_argument('--reg', type=float, default=0.01,
                        help='regularizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Max Epochs')
    parser.add_argument('--log_every_batch', type=int, default=50,
                        help='Log every batch')
    parser.add_argument('--save_ckpt_every', type=int, default=20,
                        help='Save Checkpoint Every')
    parser.add_argument('--dataset', type=str, default="QuestionLabels",
                        help='dataset')
    parser.add_argument('--base_dataset', type=str, default="Names",
                        help='base_dataset')
    parser.add_argument('--checkpoints_directory', type=str, default="CKPTS",
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
    checkpoints_dir = "{}/ADVERSARIAL_GUMBEL".format(args.checkpoints_directory)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_suffix = "lr_{}_tmin_{}_fw_{}_bs_{}_rand_{}_classifer_{}".format(args.learning_rate, args.temp_min, args.filter_width, 
        args.batch_size, args.random_network,args.classifier_type)

    checkpoints_dir = "{}/{}_adversarial_base_{}_{}".format(checkpoints_dir, args.dataset, 
        args.base_dataset, checkpoint_suffix)
    
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    start_epoch = 0
    training_log = {
        'log' : [],
        'best_epoch' : 0,
        'best_accuracy' : 0.0,
        'running_reward' : []
    }
    running_reward = -args.batch_size
    
    lstm_loss_criterion = nn.CrossEntropyLoss()

    if args.continue_training == "True":
        if args.resume_run == -1:
            run_index = len(os.listdir(checkpoints_dir)) - 1
        else:
            run_index = args.resume_run
        checkpoints_dir = "{}/{}".format(checkpoints_dir, run_index)
        if not os.path.exists(checkpoints_dir):
            raise Exception("Coud not find checkpoints_dir")

        with open("{}/training_log.json".format(checkpoints_dir)) as tlog_f:
            print "CHECKSSSSSS"
            training_log = json.load(tlog_f)

        seq_model.load_state_dict(torch.load("{}/best_model.pth".format(checkpoints_dir)))
        start_epoch = training_log['best_epoch']
        # running_reward = training_log['running_reward'][-1]
    else:
        run_index = len(os.listdir(checkpoints_dir))
        checkpoints_dir = "{}/{}".format(checkpoints_dir, run_index)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
    
    temp_min = args.temp_min
    temp_max = args.temp_max

    for epoch in range(start_epoch, args.max_epochs):
        new_classifier.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            slope = (temp_max - temp_min)/args.epochs_to_anneal
            temp = max( temp_max - (slope*epoch), temp_min)
            rewritten_x = seq_model(batch[0], temp = temp)
            pred_logits = lstm_model(seq_model.probs)
            # print seq_model.probs
            _, predictions = torch.max(pred_logits, 1)

            pred_correctness = (predictions == batch[1]).float()
            pred_correctness[pred_correctness == 0.0] = -1.0
            rewards = pred_correctness
            batch_reward = torch.sum(rewards)
            # print batch_reward

            loss = lstm_loss_criterion(pred_logits, batch[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print running_reward/(args.log_every_batch * 1.0)
            # print batch_reward/(args.log_every_batch * 1.0)
            running_reward -= running_reward/(args.log_every_batch * 1.0)
            running_reward += batch_reward/(args.log_every_batch * 1.0)

            if batch_idx % args.log_every_batch == 0:
                if args.print_prob == "True":
                    print "Temp", temp, seq_model.probs
                print ("Epoch[{}] Iteration[{}] RunningLoss[{}] Reward[{}] Temp[{}]".format(
                    epoch, batch_idx, loss, running_reward, temp))

        evaluator.run(train_loader)
        training_metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(epoch, training_metrics['accuracy']))

        evaluator.run(val_loader)
        evaluation_metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(epoch, evaluation_metrics['accuracy']))

        training_log['log'].append({
            'training_metrics' : training_metrics,
            'evaluation_metrics' : evaluation_metrics,
            'temp' : temp
        })

        if evaluation_metrics['accuracy'] > training_log['best_accuracy']:
            torch.save(seq_model.state_dict(), "{}/best_model.pth".format(checkpoints_dir))
            training_log['best_accuracy'] = evaluation_metrics['accuracy']
            training_log['best_epoch'] = epoch

        if epoch % args.save_ckpt_every == 0:
            torch.save(seq_model.state_dict(), "{}/model_{}.pth".format(checkpoints_dir, epoch))

        print "BEST", training_log['best_epoch'], training_log['best_accuracy']
        with open("{}/training_log.json".format(checkpoints_dir), 'w') as f:
            f.write(json.dumps(training_log))

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    


if __name__ == '__main__':
    main()