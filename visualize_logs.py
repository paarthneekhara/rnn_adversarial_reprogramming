import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import json
# from tkinter.filedialog import askopenfilename
import glob
import os
import shutil

def findFiles(path):
    return glob.glob(path)

def main():
    parser = argparse.ArgumentParser(description='Visualise')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log File')
    parser.add_argument('--adv_dir', type=str, default="ADVERSARIAL_GUMBEL",
                        help='adv_dir')
    args = parser.parse_args()

    if args.log_file == None:
        log_files = findFiles("CKPTS/{}/*/*/*.json".format(args.adv_dir))
    else:
        log_files = [args.log_file]


    plot_dir = "PLOTS"
    
    try:
        shutil.rmtree(plot_dir)
    except:
        pass

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    

    args = parser.parse_args()

    for filename in log_files:
        print filename
        try:
            with open(filename) as f:
                training_logs = json.load(f)
        except:
            continue

        training_accuracy = []
        validation_accuracy = []
        epochs = []

        for epoch, iter_details in enumerate(training_logs['log']):
            training_accuracy.append(iter_details['training_metrics']['accuracy'] * 100.0)
            validation_accuracy.append(iter_details['evaluation_metrics']['accuracy'] * 100.0)
            epochs.append(epoch)

        plt.plot(epochs, training_accuracy, label = "Training Accuracy")
        plt.plot(epochs, validation_accuracy, label = "Validation Accuracy")
        plt.title(chart_title(filename, args.adv_dir))
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc = 4)

        exp_name = filename.split("/")[2] + "_" + filename.split("/")[3]
        plt.savefig("{}/{}_{}".format(plot_dir, exp_name, "ACC.pdf"))
        plt.clf()
        if 'running_reward' in training_logs:
            minibatch_steps = []
            running_rewards = []
            for minibatch_step, running_reward in enumerate(training_logs['running_reward']):
                minibatch_steps.append(minibatch_step)
                running_rewards.append(running_reward/8.0)

            plt.plot(minibatch_steps, running_rewards, label = "Reward")
            plt.legend(loc = 4)
            plt.title(chart_title(filename, args.adv_dir))
            plt.ylabel("Reward")
            plt.xlabel("Mini-Batch Step")
            plt.savefig("{}/{}_{}".format(plot_dir, exp_name, "REW.pdf"))
            plt.clf()

def chart_title(filename, adv_dir):
    name = filename.split("/")[2]
    name_elements = name.split("_")
    adv_task = name_elements[0]
    original_task = name_elements[3]
    model = name_elements[-1]

    name_map  = {
        'SubNames' : 'Names-5 Classification',
        'QuestionLabels' : 'Question Classification',
        'charRNN' : 'LSTM',
        'biRNN' : 'Bi-LSTM',
        'CNN' : 'CNN',
        'TwitterArabic' : 'Arabic Tweets Classification',
        'IMDB' : 'IMDB Review Sentiment Classification',
        'Names' : 'Names-18 Classification'
    }

    task_type = "White-Box"
    if 'GUMBEL' not in adv_dir:
        task_type = "Black-Box"
    title = "{} Adversarial Reprogramming of {} model \n trained on {} for {}".format(
        task_type, name_map[model], name_map[original_task], name_map[adv_task])

    return title

            
if __name__ == '__main__':
    main()