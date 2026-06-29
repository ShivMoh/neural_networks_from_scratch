import matplotlib.pyplot as plt
import math

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.show()

def plot_predictions(images, true_labels, pred_labels, cols=5):
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(images[i].squeeze(), cmap="gray")
            correct = true_labels[i] == pred_labels[i]
            color = "green" if correct else "red"
            ax.set_title(f"T:{true_labels[i]} P:{pred_labels[i]}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Predictions  (green=correct, red=wrong)", fontsize=11)
    plt.tight_layout()
    plt.show()
