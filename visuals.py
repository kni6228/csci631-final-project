import matplotlib.pyplot as plt
import numpy as np
import os
from constants import LEARNING_CURVE_PATH


def plot_learning_curve(train_acc_history, val_acc_history, num_epochs=20):
    if not os.path.exists("./graphs"):
       os.mkdir("./graphs")

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for phase, history in phase_histories.items():
        plt.plot(range(1, num_epochs + 1), history, label=phase)
    plt.ylim((0, 1.))
    plt.yticks(np.arange(0.1, 1.0, 0.1))
    plt.xticks(np.arange(1, num_epochs + 1, np.ceil(num_epochs / 20)))
    plt.savefig(LEARNING_CURVE_PATH)
    plt.show()
