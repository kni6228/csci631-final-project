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
    plt.plot(range(1, num_epochs + 1), train_acc_history, label='Train')
    plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation')
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, np.ceil(num_epochs / 20)))
    plt.legend()
    plt.savefig(LEARNING_CURVE_PATH)
    plt.show()
