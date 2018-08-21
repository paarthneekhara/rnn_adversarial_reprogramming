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

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Output filename')
    parser.add_argument('--reg', type=float, default=1.0,
                        help='Output filename')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Output filename')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Max Epochs')
    parser.add_argument('--log_every_batch', type=int, default=10,
                        help='Log every batch')
    parser.add_argument('--save_ckpt_every', type=int, default=10,
                        help='Save Checkpoint Every')
    parser.add_argument('--dataset', type=str, default="Names",
                        help='Output filename')
    parser.add_argument('--base_dataset', type=str, default="Names",
                        help='Output filename')
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
    parser.add_argument('--random_network', type=str, default="False",
                        help='Random Network')
    parser.add_argument('--classifier_type', type=str, default="charRNN",
                        help='rnn type')

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
    
    checkpoint_suffix = "{}_{}_{}_random_{}_classifer_{}".format(args.learning_rate, args.filter_width, 
        args.batch_size, args.random_network,args.classifier_type)

    checkpoints_dir = "{}/{}_adversarial_base_{}_{}".format(args.checkpoints_directory, args.dataset, 
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
        with open("{}/training_log.json".format(checkpoints_dir)) as tlog_f:
            training_log = json.load(tlog_f)

        seq_model.load_state_dict(torch.load("{}/best_model.pth".format(checkpoints_dir)))
        start_epoch = training_log['best_epoch']
        # Not really but oh well
        running_reward = training_log['running_reward'][-1]

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
            max_length_to_update = 100
            for idx, log_prob in enumerate(seq_model.saved_log_probs):
#                 print idx, log_prob
                
                if (idx % (batch[0].size()[1])) < max_length_to_update:
                    seq_rewriter_loss += (-log_prob * rewards[idx/rewritten_x.size()[1]])

            seq_rewriter_loss /= (args.batch_size * max_length_to_update)
            print "Without reg", seq_rewriter_loss
            seq_rewriter_loss += (- args.reg * seq_model.entropy)
            print "With reg", seq_rewriter_loss

            optimizer.zero_grad()
            seq_rewriter_loss.backward()
            optimizer.step()
#             print "done backward"
            seq_model.saved_log_probs = None
#             print "Deleted"
            batch_reward = torch.sum(rewards)
            running_reward -= running_reward/(args.log_every_batch * 1.0)
            running_reward += batch_reward/(args.log_every_batch * 1.0)

            if batch_idx % args.log_every_batch == 0:
                print ("Epoch[{}] Iteration[{}] Running Reward[{}]".format(epoch, batch_idx, running_reward))
                training_log['running_reward'].append(float(running_reward.cpu().numpy()))

        evaluator.run(train_loader)
        training_metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(epoch, training_metrics['accuracy']))

        evaluator.run(val_loader)
        evaluation_metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f}"
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