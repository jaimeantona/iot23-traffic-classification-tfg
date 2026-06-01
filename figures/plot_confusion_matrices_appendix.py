import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

OUTPUT_DIR = "img/inkscape/pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BINARY_LABELS = ["Attack", "Benign"]

MULTICLASS_LABELS = [
    "Benign",
    "C&C",
    "DDoS",
    "Okiru",
    "PortScan"
]


# ============================================================
# MATRICES DE CONFUSIÓN
# Filas = clase real
# Columnas = predicción
# ============================================================

CONFUSION_MATRICES = {
    # --------------------------------------------------------
    # Escenario binario
    # Orden: Attack, Benign
    # --------------------------------------------------------
    "cm_binario_rf.pdf": {
        "matrix": np.array([
            [19910,    90],
            [   77, 19923]
        ]),
        "labels": BINARY_LABELS,
        "title": "Random Forest"
    },

    "cm_binario_lgbm.pdf": {
        "matrix": np.array([
            [19860,   140],
            [    2, 19998]
        ]),
        "labels": BINARY_LABELS,
        "title": "LightGBM"
    },

    "cm_binario_mlp.pdf": {
        "matrix": np.array([
            [19792,   208],
            [    8, 19992]
        ]),
        "labels": BINARY_LABELS,
        "title": "MLP"
    },

    # --------------------------------------------------------
    # Escenario multiclase con histogramas
    # Orden:
    # Benign, C&C, DDoS, Okiru, PortScan
    # --------------------------------------------------------
    "cm_multiclase_hist_rf.pdf": {
        "matrix": np.array([
            [18872,   373,   753,     0,     2],
            [  207, 19785,     8,     0,     0],
            [  671,    16, 19123,   142,    48],
            [    5,     0,    10, 19984,     1],
            [    2,     1,     5, 13383,  6609]
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "Random Forest"
    },

    "cm_multiclase_hist_lgbm.pdf": {
        "matrix": np.array([
            [17802,   485,  1711,     0,     2],
            [  185, 19805,    10,     0,     0],
            [    8,    12, 19782,   139,    59],
            [    3,     1,    11, 19984,     1],
            [    1,     1,     8, 13384,  6606]
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "LightGBM"
    },

    "cm_multiclase_hist_mlp.pdf": {
        "matrix": np.array([
            [17116,  1125,  1750,     1,     8],
            [ 6018, 13929,    53,     0,     0],
            [   14,     8, 19791,   139,    48],
            [    5,     6,    12, 19977,     0],
            [    8,    57,    26, 13381,  6528]
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "MLP"
    },

    # --------------------------------------------------------
    # Escenario multiclase sin histogramas
    # Orden:
    # Benign, C&C, DDoS, Okiru, PortScan
    # --------------------------------------------------------
    "cm_multiclase_nohist_rf.pdf": {
        "matrix": np.array([
            [18871,   374,   754,     0,     1],
            [  218, 19766,    16,     0,     0],
            [  667,    21, 19122,   142,    48],
            [    5,     0,    10, 19984,     1],
            [    2,     1,     6, 13383,  6608]
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "Random Forest"
    },

    "cm_multiclase_nohist_lgbm.pdf": {
        "matrix": np.array([
            [17796,   493,  1710,     0,     1],
            [  186, 19800,    14,     0,     0],
            [    6,    13, 19782,   139,    60],
            [    3,     1,    11, 19985,     0],
            [    2,     1,     6, 13384,  6607]
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "LightGBM"
    },

    "cm_multiclase_nohist_mlp.pdf": {
        "matrix": np.array([
            [16931,  1296,  1735,    33,     5],
            [ 6141, 13799,    40,    20,     0],
            [   22,   101, 19379,   294,   204],
            [    2,    69,    11, 19903,    15],
            [   91,   280,   110, 13378,  6141]
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "MLP"
    },
}


# ============================================================
# FUNCIÓN DE REPRESENTACIÓN
# ============================================================

def plot_confusion_matrix(matrix, labels, title, output_path):
    n_classes = len(labels)

    if n_classes == 2:
        figsize = (4.2, 3.8)
        rotation = 0
        fontsize_values = 9
    else:
        figsize = (5.8, 5.0)
        rotation = 35
        fontsize_values = 8

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="viridis")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicción", fontsize=10)
    ax.set_ylabel("Clase real", fontsize=10)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor"
    )

    threshold = matrix.max() / 2.0

    for i in range(n_classes):
        for j in range(n_classes):
            value = int(matrix[i, j])

            # Fondo amarillo para valores altos: texto negro.
            # Fondo violeta oscuro para valores bajos: texto blanco.
            color = "black" if value > threshold else "white"

            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize_values
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# GENERACIÓN DE FIGURAS
# ============================================================

def main():
    for filename, config in CONFUSION_MATRICES.items():
        output_path = os.path.join(OUTPUT_DIR, filename)

        plot_confusion_matrix(
            matrix=config["matrix"],
            labels=config["labels"],
            title=config["title"],
            output_path=output_path
        )

        print(f"Generado: {output_path}")

    print("\nTodas las matrices se han generado correctamente.")


if __name__ == "__main__":
    main()