import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model_classifier, seq_rewriter
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
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--reg', type=float, default=0.01,
                        help='Regularizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Max Epochs')
    parser.add_argument('--log_every_batch', type=int, default=10,
                        help='Log every batch')
    parser.add_argument('--save_ckpt_every', type=int, default=20,
                        help='Save Checkpoint Every')
    parser.add_argument('--dataset', type=str, default="Names",
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
    parser.add_argument('--progressive', type=str, default="True",
                        help='Progressively increase length for back prop')
    parser.add_argument('--progress_up_to', type=float, default=30.0,
                        help='Epoch Number upto which length will be progressed to full length')

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

    seq_model = seq_rewriter.seq_rewriter({
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
    checkpoints_dir = "{}/ADVERSARIAL".format(args.checkpoints_directory)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_suffix = "lr_{}_rg_{}_fw_{}_bs_{}_rd_{}_classifer_{}".format(args.learning_rate, args.reg, args.filter_width, 
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
        running_reward = training_log['running_reward'][-1]
    else:
        run_index = len(os.listdir(checkpoints_dir))
        checkpoints_dir = "{}/{}".format(checkpoints_dir, run_index)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
    
    for epoch in range(start_epoch, args.max_epochs):
        new_classifier.train()
        for batch_idx, batch in enumerate(train_loader):
            rewritten_x = seq_model(batch[0])
            pred_logits = lstm_model(rewritten_x)
            _, predictions = torch.max(pred_logits, 1)

            pred_correctness = (predictions == batch[1]).float()
            pred_correctness[pred_correctness == 0.0] = -1.0
            rewards = pred_correctness
            # lstm_loss = lstm_loss_criterion(pred_logits, batch[1])
            seq_rewriter_loss = 0
            max_length_to_update = train_dataset.seq_length + args.filter_width + 1
            if args.progressive == "True":
                max_length_to_update = min( int( (epoch/args.progress_up_to) * max_length_to_update  ) + 1, max_length_to_update )
            for idx, log_prob in enumerate(seq_model.saved_log_probs):
                if (idx % (batch[0].size()[1])) < max_length_to_update:
                    seq_rewriter_loss += (-log_prob * rewards[idx/rewritten_x.size()[1]])

            # seq_rewriter_loss /= (args.batch_size * max_length_to_update)
            # seq_rewriter_loss += (- args.reg * seq_model.entropy)

            l2_reg = None
            for W in seq_model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            
            # reg_loss = args.reg * l2_reg
            reg_loss = 0
            seq_rewriter_loss_combined = seq_rewriter_loss + reg_loss
            optimizer.zero_grad()
            seq_rewriter_loss_combined.backward()
            optimizer.step()
            seq_model.saved_log_probs = None

            batch_reward = torch.sum(rewards)
            running_reward -= running_reward/(args.log_every_batch * 1.0)
            running_reward += batch_reward/(args.log_every_batch * 1.0)

            if batch_idx % args.log_every_batch == 0:
                print ("Epoch[{}] Iteration[{}] Running Reward[{}] LossBasic[{}] RegLoss[{}] max_length_to_update[{}]".format(
                    epoch, batch_idx, running_reward, seq_rewriter_loss, reg_loss, max_length_to_update))
                training_log['running_reward'].append(float(running_reward.cpu().numpy()))

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