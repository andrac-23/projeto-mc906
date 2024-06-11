import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_loss_plots(table, plots_path, display=False):
    """
    Cria gráfico com os valores de loss do modelo por época por treino e validação

    param table: DataFrame pandas com os valores de loss do modelo
    param plots_path: Caminho para salvar o gráfico

    """
    plt.ioff()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 6.65))
    fig.suptitle("Perdas do modelo no treino e validação por época")

    def plot_loss_graph(ax, train_loss, val_loss, loss_name):
        ax.plot(range(1, len(train_loss) + 1), train_loss, zorder=3, label=f"train/{loss_name}")
        ax.scatter(range(1, len(train_loss) + 1), train_loss, c='royalblue', zorder=4)
        ax.plot(range(1, len(val_loss) + 1), val_loss, zorder=3, label=f"val/{loss_name}", c='red')
        ax.scatter(range(1, len(val_loss) + 1), val_loss, c='red', zorder=4)

        min_train_loss = train_loss.min()
        max_train_loss = train_loss.max()
        min_val_loss = val_loss.min()
        max_val_loss = val_loss.max()
        min_val = min(min_train_loss, min_val_loss)
        max_val = max(max_train_loss, max_val_loss)
        ax.set_yticks(np.arange(min_val * 0.95, max_val * 1.05, (max_val - min_val) / 15))

        ax.grid(zorder=0, alpha=0.55)
        ax.legend(loc='upper right')
        ax.set_xlabel("epochs")
        ax.set_ylabel(loss_name)
        ax.set_title(f"{loss_name} X epochs")

    plot_loss_graph(axs[0], table["train/box_loss"], table["val/box_loss"], "box_loss")
    plot_loss_graph(axs[1], table["train/cls_loss"], table["val/cls_loss"], "cls_loss")
    plot_loss_graph(axs[2], table["train/dfl_loss"], table["val/dfl_loss"], "dfl_loss")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    if display:
        plt.show()
    else:
        plt.savefig(plots_path)

def plot_graph(ax, metric_column, metric_name):
    ax.plot(range(1, len(metric_column) + 1), metric_column, zorder=3, label=metric_name)
    ax.scatter(range(1, len(metric_column) + 1), metric_column, c="royalblue", zorder=4)

    min_metric = metric_column.min()
    max_metric = metric_column.max()
    ax.set_yticks(np.arange(min_metric * 0.95, max_metric * 1.05, (max_metric - min_metric) / 11))

    ax.grid(zorder=0, alpha=0.55)
    ax.legend(loc='upper left')
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} X epochs")

def create_metrics_plots(table, plots_path, display=False):
    plt.ioff()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 6.65))
    fig.suptitle("Métricas gerais do modelo por época (validação)")

    plot_graph(axs[0, 0], table["metrics/precision(B)"], "metrics/precision(B)")
    plot_graph(axs[0, 1], table["metrics/recall(B)"], "metrics/recall(B)")
    plot_graph(axs[1, 0], table["metrics/mAP50(B)"], "metrics/mAP50(B)")
    plot_graph(axs[1, 1], table["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)

    if display:
        plt.show()
    else:
        plt.savefig(plots_path)

if __name__ == '__main__':
    results_csv_path = r"C:\Users\User\Desktop\v2_P\results.csv"
    table = pd.read_csv(results_csv_path)
    table.columns = table.columns.str.strip()
    #create_loss_plots(table, "plots.png", display=True)
    create_metrics_plots(table, "plots.png", display=True)
