import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model_classifier
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
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Output filename')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Output filename')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs')
    parser.add_argument('--dataset', type=str, default="Names",
                        help='Output filename')
    parser.add_argument('--checkpoints_directory', type=str, default="CKPTS",
                        help='Check Points Directory')
    parser.add_argument('--hidden_units', type=int, default=256,
                        help='hidden_units')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='embedding_size')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience')
    parser.add_argument('--classifier_type', type=str, default="charRNN",
                        help='rnn type')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    train_dataset = datasets.get_dataset(args.dataset, dataset_type = 'train')
    val_dataset = datasets.get_dataset(args.dataset, dataset_type = 'train_val')


    if args.classifier_type == "charRNN":
        model_options = {
            'vocab_size' : len(train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(train_dataset.classes),
            'embedding_size' : args.embedding_size
        }
        model = model_classifier.uniRNN(model_options)
        print "char RNN"

    if args.classifier_type == "biRNN":
        model_options = {
            'vocab_size' : len(train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(train_dataset.classes),
            'embedding_size' : args.embedding_size
        }
        model = model_classifier.biRNN(model_options)
        print "BI RNN"

    if args.classifier_type == "CNN":
        model_options = {
            'vocab_size' : len(train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(train_dataset.classes),
            'embedding_size' : args.embedding_size
        }
        model = model_classifier.CnnTextClassifier(model_options)
        print "CnnTextClassifier"

    print device
    model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    loss_criterion = nn.CrossEntropyLoss()

    print "check", torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)

    trainer = create_supervised_trainer(model, optimizer, loss_criterion)
    evaluator = create_supervised_evaluator(model,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                            'nll': Loss(loss_criterion)
                                        })

    checkpoints_dir = "{}/{}_classifer_{}".format(args.checkpoints_directory, args.dataset, args.classifier_type)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    training_log = {
        'model_options' : model_options,
        'log' : [],
        'best_epoch' : 0,
        'best_accuracy' : 0.0
    }

    

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        total_batches =  int(len(train_dataset)/args.batch_size)
        if trainer.state.iteration % 100 == 0:
            print("Epoch[{}] Iteration[{}] Total Iterations[{}] Loss: {:.2f}".format(
                trainer.state.epoch, trainer.state.iteration, total_batches, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        
        evaluator.run(train_loader)
        training_metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, training_metrics['accuracy'], training_metrics['nll']))

        evaluator.run(val_loader)
        evaluation_metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, evaluation_metrics['accuracy'], evaluation_metrics['nll']))

        out_path = "{}/model_epoch_{}.pth".format(checkpoints_dir, trainer.state.epoch)
        torch.save(model.state_dict(), out_path)

        training_log['log'].append({
            'training_metrics' : training_metrics,
            'evaluation_metrics' : evaluation_metrics,
        })

        # if (trainer.state.epoch - training_log['best_epoch']) > args.patience and (evaluation_metrics['accuracy'] < training_log['best_accuracy']):
        #     trainer.terminate()

        if evaluation_metrics['accuracy'] > training_log['best_accuracy']:
            torch.save(model.state_dict(), "{}/best_model.pth".format(checkpoints_dir))
            training_log['best_accuracy'] = evaluation_metrics['accuracy']
            training_log['best_epoch'] = trainer.state.epoch

        print "BEST", training_log['best_epoch'], training_log['best_accuracy']
        with open("{}/training_log.json".format(checkpoints_dir), 'w') as f:
            f.write(json.dumps(training_log))
    
    trainer.run(train_loader, max_epochs=args.epochs)
    
if __name__ == '__main__':
    main()