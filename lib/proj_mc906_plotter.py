import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_loss_plots(table, plots_path):
    """
    Cria gráfico com os valores de loss do modelo por época por treino e validação

    param table: DataFrame pandas com os valores de loss do modelo
    param plots_path: Caminho para salvar o gráfico

    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5.5))
    fig.suptitle("Perdas do modelo no treino e validação por época")

    # plot graph of the train/box_loss
    train_box_loss = table["train/box_loss"]
    axs[0].plot(range(1, len(train_box_loss) + 1), train_box_loss, zorder=3, label="train/box_loss")
    axs[0].scatter(range(1, len(train_box_loss) + 1), train_box_loss, c='royalblue', zorder=4)

    # plot graph of the validation/box_loss
    val_box_loss = table["val/box_loss"]
    axs[0].plot(range(1, len(val_box_loss) + 1), val_box_loss, zorder=3, label="val/box_loss", c='red')
    axs[0].scatter(range(1, len(val_box_loss) + 1), val_box_loss, c='red', zorder=4)

    # ticks
    min_train_box_loss = table["train/box_loss"].min()
    max_train_box_loss = table["train/box_loss"].max()
    min_val_box_loss = table["val/box_loss"].min()
    max_val_box_loss = table["val/box_loss"].max()
    min_val = min(min_train_box_loss, min_val_box_loss)
    max_val = max(max_train_box_loss, max_val_box_loss)
    axs[0].set_yticks(np.arange(min_val * 0.95, max_val * 1.05, (max_val - min_val) / 15))

    axs[0].grid(zorder=0, alpha=0.55)
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel("epochs")
    axs[0].set_ylabel("box_loss")
    axs[0].set_title("box_loss X epochs")

    # plot graph of the train/cls_loss
    train_cls_loss = table["train/cls_loss"]
    axs[1].plot(range(1, len(train_cls_loss) + 1), train_cls_loss, zorder=3, label="train/cls_loss")
    axs[1].scatter(range(1, len(train_cls_loss) + 1), train_cls_loss, c='royalblue', zorder=4)

    # plot graph of the validation/cls_loss
    val_cls_loss = table["val/cls_loss"]
    axs[1].plot(range(1, len(val_cls_loss) + 1), val_cls_loss, zorder=3, label="val/cls_loss", c='red')
    axs[1].scatter(range(1, len(val_cls_loss) + 1), val_cls_loss, c='red', zorder=4)

    # ticks
    min_train_cls_loss = table["train/cls_loss"].min()
    max_train_cls_loss = table["train/cls_loss"].max()
    min_val_cls_loss = table["val/cls_loss"].min()
    max_val_cls_loss = table["val/cls_loss"].max()
    min_val = min(min_train_cls_loss, min_val_cls_loss)
    max_val = max(max_train_cls_loss, max_val_cls_loss)
    axs[1].set_yticks(np.arange(min_val * 0.95, max_val * 1.05, (max_val - min_val) / 15))

    axs[1].grid(zorder=0, alpha=0.55)
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel("epochs")
    axs[1].set_ylabel("cls_loss")
    axs[1].set_title("cls_loss X epochs")

    # plot graph of the train/dfl_loss
    train_dfl_loss = table["train/dfl_loss"]
    axs[2].plot(range(1, len(train_dfl_loss) + 1), train_dfl_loss, zorder=3, label="train/dfl_loss")
    axs[2].scatter(range(1, len(train_dfl_loss) + 1), train_dfl_loss, c='royalblue', zorder=4)

    # plot graph of the validation/cls_loss
    val_dfl_loss = table["val/dfl_loss"]
    axs[2].plot(range(1, len(val_dfl_loss) + 1), val_dfl_loss, zorder=3, label="val/dfl_loss", c='red')
    axs[2].scatter(range(1, len(val_dfl_loss) + 1), val_dfl_loss, c='red', zorder=4)

    # ticks
    min_train_dfl_loss = table["train/dfl_loss"].min()
    max_train_dfl_loss = table["train/dfl_loss"].max()
    min_val_dfl_loss = table["val/dfl_loss"].min()
    max_val_dfl_loss = table["val/dfl_loss"].max()
    min_val = min(min_train_dfl_loss, min_val_dfl_loss)
    max_val = max(max_train_dfl_loss, max_val_dfl_loss)
    axs[2].set_yticks(np.arange(min_val * 0.95, max_val * 1.05, (max_val - min_val) / 15))

    axs[2].grid(zorder=0, alpha=0.55)
    axs[2].legend(loc='upper right')
    axs[2].set_xlabel("epochs")
    axs[2].set_ylabel("dfl_loss")
    axs[2].set_title("dfl_loss X epochs")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(plots_path)

if __name__ == '__main__':
    print("Esse script não deve ser executado diretamente.")
