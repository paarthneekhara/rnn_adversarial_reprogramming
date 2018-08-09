import matplotlib.pyplot as plt
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log File')
    
    args = parser.parse_args()

    with open(args.log_file) as f:
        training_logs = json.load(f)

    training_accuracy = []
    validation_accuracy = []
    epochs = []

    for epoch, iter_details in enumerate(training_logs['log']):
        training_accuracy.append(iter_details['training_metrics']['accuracy'])
        validation_accuracy.append(iter_details['evaluation_metrics']['accuracy'])
        epochs.append(epoch)

    minibatch_steps = []
    running_rewards = []
    for minibatch_step, running_reward in enumerate(training_logs['running_reward']):
        minibatch_steps.append(minibatch_step)
        running_rewards.append(running_reward)

    plt.plot(epochs, training_accuracy, label = "training accuracy")
    plt.plot(epochs, validation_accuracy, label = "validation accuracy")
    plt.legend()
    plt.show()

    plt.plot(minibatch_steps, running_rewards, label = "running reward")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()