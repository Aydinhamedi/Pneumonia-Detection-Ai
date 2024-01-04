# libs
import numpy as np
import matplotlib.pyplot as plt


# funcs
def _noise(scale):
    return np.random.uniform(low=-1, high=1) * scale


def simulate_fitting(noise_level: float,
                     mode: int,
                     power: float,
                     max_acc: float = 0.99,
                     epochs: int = 600,
                     sim_slope: float = 25):
    sim_train_acc = []
    sim_train_loss = []
    sim_val_acc = []
    sim_val_loss = []
    for epoch_raw in range(1, epochs):
        # calculating raw_sim_slope
        epoch = epoch_raw / 100
        sim_pram1 = sim_slope
        raw_sim_slope = -((2 / (epoch + 1) ** (epoch + sim_pram1)) / 2) + 1 + _noise(noise_level)
        raw_sim_slope_of = ((2 / ((epoch + 1) ** epoch) + (epoch - (epoch**2)) / 2) + 1 + _noise(noise_level)) - 4
        # Setting train metrics
        sim_train_acc.append(raw_sim_slope)
        sim_train_loss.append(-raw_sim_slope + 1)
        # calculating val
        if mode == 0:
            sim_val_loss_pre = -raw_sim_slope_of + 1
        elif mode == 1:
            sim_val_loss_pre = -raw_sim_slope + 1
        else:
            raise ValueError('Unknown mode')

        sim_val_loss.append(sim_val_loss_pre)
        sim_val_acc.append(raw_sim_slope)
    return sim_train_loss, sim_val_loss, sim_train_acc, sim_val_acc


def plot_simulation(sim_train_loss, sim_val_loss, sim_train_acc, sim_val_acc):
    epochs = range(1, len(sim_train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, sim_train_loss, 'bo', label='Training loss')
    plt.plot(epochs, sim_val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, sim_train_acc, 'bo', label='Training accuracy')
    plt.plot(epochs, sim_val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Simulate overfitting
train_loss, val_loss, train_acc, val_acc = simulate_fitting(noise_level=0.1, mode=0, power=5,
                                                            sim_slope=0.5, max_acc=0.9)
print("Overfitting simulation:")
plot_simulation(train_loss, val_loss, train_acc, val_acc)

# Simulate normal training
train_loss, val_loss, train_acc, val_acc = simulate_fitting(noise_level=0.01, mode=1, power=0,
                                                            sim_slope=1, max_acc=0.9)
print("\nNormal training simulation:")
plot_simulation(train_loss, val_loss, train_acc, val_acc)
