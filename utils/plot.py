import matplotlib.pyplot as plt

# 从日志文件中提取到的数据
epochs = [0.28, 0.56, 0.83, 1.11, 1.39, 1.67, 2.22, 2.5, 3.06, 3.34, 3.62, 3.89, 4.17, 4.45, 4.73]
train_losses = [0.0444, 0.0165, 0.0126, 0.0097, 0.0074, 0.0069, 0.0064, 0.0045, 0.0040, 0.0036, 0.0021, 0.0022, 0.0017, 0.0013, 0.0013]
eval_epochs = [1.0, 2.0, 3.0, 4.0, 5.0]
eval_losses = [0.014805164188146591, 0.013287968002259731, 0.014198029413819313, 0.016179021447896957, 0.018241053447127342]
eval_accuracies = [0.9957497958850094, 0.996405634140164, 0.9966742810256809, 0.9968090090571222, 0.9968802166653316]

def plot_loss(epochs, train_losses, eval_epochs, eval_losses):
    plt.figure(figsize=(7, 6))
    plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss')
    plt.plot(eval_epochs, eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss_over_epochs.png")
    plt.close()

def plot_accuracy(eval_epochs, eval_accuracies):
    plt.figure(figsize=(7, 6))
    plt.plot(eval_epochs, eval_accuracies, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Eval Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("accuracy_over_epochs.png")
    plt.close()

def main():
    plot_loss(epochs, train_losses, eval_epochs, eval_losses)
    plot_accuracy(eval_epochs, eval_accuracies)

if __name__ == "__main__":
    main()
