import matplotlib.pyplot as plt
import numpy as np

from constants import LEARNING_CURVE_PATH


def plot_learning_curve(phase_histories, num_epochs=20):
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for phase, history in phase_histories.items():
        plt.plot(range(1, num_epochs + 1), history, label=phase)
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, np.ceil(num_epochs / 20)))
    plt.savefig(LEARNING_CURVE_PATH)
    plt.show()
