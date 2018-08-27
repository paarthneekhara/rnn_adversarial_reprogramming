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
    parser.add_argument('--adv_dir', type=str, default="ADVERSARIAL",
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
        with open(filename) as f:
            training_logs = json.load(f)

        training_accuracy = []
        validation_accuracy = []
        epochs = []

        for epoch, iter_details in enumerate(training_logs['log']):
            training_accuracy.append(iter_details['training_metrics']['accuracy'])
            validation_accuracy.append(iter_details['evaluation_metrics']['accuracy'])
            epochs.append(epoch)

        plt.plot(epochs, training_accuracy, label = "training accuracy")
        plt.plot(epochs, validation_accuracy, label = "validation accuracy")
        plt.legend()

        exp_name = filename.split("/")[2] + "_" + filename.split("/")[3]
        plt.savefig("{}/{}_{}".format(plot_dir, exp_name, "ACC.png"))
        plt.clf()
        if 'running_rewards' in training_logs:
            minibatch_steps = []
            running_rewards = []
            for minibatch_step, running_reward in enumerate(training_logs['running_reward']):
                minibatch_steps.append(minibatch_step)
                running_rewards.append(running_reward)

            plt.plot(minibatch_steps, running_rewards, label = "running reward")
            plt.legend()
            plt.savefig("{}/{}_{}".format(plot_dir, exp_name, "REW.png"))
            plt.clf()

if __name__ == '__main__':
    main()