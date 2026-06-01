import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Cambia esta ruta por la de tu dataset multiclase final con histogramas
INPUT_CSV = "pipeline_hist/out/dataset_balanced_100k_per_class_withproto_fixed_norm.csv"

# Carpeta donde se guardarán las figuras para LaTeX
OUTPUT_DIR = "img/pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clases a comparar
CLASS_A = "Okiru"
CLASS_B = "PartOfAHorizontalPortScan"

# Nombres abreviados para las leyendas
DISPLAY_NAMES = {
    "Okiru": "Okiru",
    "PartOfAHorizontalPortScan": "PartOfAHorizontalPortScan"
}

# Columnas de histogramas
HIST_GROUPS = {
    "Forward Packet Size Histogram": [f"fwd_size_bin{i}" for i in range(1, 9)],
    "Backward Packet Size Histogram": [f"bwd_size_bin{i}" for i in range(1, 9)],
    "Forward IPT Histogram": [f"fwd_ipt_bin{i}" for i in range(1, 9)],
    "Backward IPT Histogram": [f"bwd_ipt_bin{i}" for i in range(1, 9)],
}

# Variables agregadas para ECDF
ECDF_COLUMNS = [
    "pkts_fwd",
    "pkts_bwd",
    "bytes_fwd",
    "bytes_bwd"
]


# ============================================================
# FUNCIONES
# ============================================================

def load_dataset(path):
    """
    Carga el dataset detectando automáticamente si usa coma o punto y coma.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el fichero: {path}")

    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path, sep=";")

    if "label" not in df.columns:
        raise ValueError("El dataset debe contener una columna llamada 'label'.")

    return df


def check_columns(df, columns):
    """
    Comprueba que existan las columnas necesarias.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas en el dataset:\n" + "\n".join(missing)
        )


def plot_histograms_okiru_portscan(df):
    """
    Genera una figura 2x2 con los histogramas medios de Okiru y PortScan.
    """

    all_hist_cols = []
    for cols in HIST_GROUPS.values():
        all_hist_cols.extend(cols)

    check_columns(df, all_hist_cols)

    df_pair = df[df["label"].isin([CLASS_A, CLASS_B])].copy()

    if df_pair.empty:
        raise ValueError("No se encontraron muestras de Okiru o PartOfAHorizontalPortScan.")

    bins = np.arange(1, 9)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    axes = axes.flatten()

    bar_width = 0.35
    offsets = {
        CLASS_A: -bar_width / 2,
        CLASS_B: bar_width / 2
    }

    for ax, (title, cols) in zip(axes, HIST_GROUPS.items()):
        for cls in [CLASS_A, CLASS_B]:
            values = df_pair[df_pair["label"] == cls][cols].mean().values

            ax.bar(
                bins + offsets[cls],
                values,
                width=bar_width,
                label=DISPLAY_NAMES[cls]
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Bin", fontsize=10)
        ax.set_ylabel("Average probability", fontsize=10)
        ax.set_xticks(bins)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = os.path.join(OUTPUT_DIR, "hist_okiru_portscan.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada: {output_path}")


def compute_ecdf(values):
    """
    Calcula la ECDF de un vector de valores.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    values = np.sort(values)

    if len(values) == 0:
        return np.array([]), np.array([])

    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def plot_ecdf_okiru_portscan(df):
    """
    Genera una figura 2x2 con las ECDF de variables agregadas.
    """

    check_columns(df, ECDF_COLUMNS)

    df_pair = df[df["label"].isin([CLASS_A, CLASS_B])].copy()

    if df_pair.empty:
        raise ValueError("No se encontraron muestras de Okiru o PartOfAHorizontalPortScan.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    axes = axes.flatten()

    for ax, col in zip(axes, ECDF_COLUMNS):
        for cls in [CLASS_A, CLASS_B]:
            values = df_pair[df_pair["label"] == cls][col].values
            x, y = compute_ecdf(values)

            # Se evita log(0) usando solo valores positivos para escala logarítmica
            positive_mask = x > 0
            x = x[positive_mask]
            y = y[positive_mask]

            ax.plot(
                x,
                y,
                label=DISPLAY_NAMES[cls],
                linewidth=1.6
            )

        ax.set_xscale("log")
        ax.set_title(f"ECDF de {col}", fontsize=11)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("ECDF", fontsize=10)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = os.path.join(OUTPUT_DIR, "ecdf_okiru_portscan.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Cargando dataset...")
    df = load_dataset(INPUT_CSV)

    print("Generando histogramas Okiru vs PortScan...")
    plot_histograms_okiru_portscan(df)

    print("Generando ECDF Okiru vs PortScan...")
    plot_ecdf_okiru_portscan(df)

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()