# Detección de anomalías en redes IoT mediante técnicas estadísticas y de aprendizaje automático

Repositorio asociado al Trabajo Fin de Grado de **Jaime Antona Díaz**, desarrollado en la Escuela Politécnica Superior de la Universidad Autónoma de Madrid.

El objetivo del proyecto es estudiar la detección y clasificación de tráfico malicioso en redes IoT a partir del conjunto de datos **IoT-23**. Para ello, se construye un *pipeline* de procesamiento a partir de capturas de tráfico en formato PCAP, se extraen características agregadas e histogramas a nivel de flujo y se evalúan distintos modelos de aprendizaje automático.

## Descripción del proyecto

El trabajo considera dos escenarios de clasificación:

* **Clasificación binaria**: tráfico benigno frente a tráfico de ataque.
* **Clasificación multiclase**: diferenciación entre cinco clases de tráfico:

  * `Benign`
  * `C&C`
  * `DDoS`
  * `Okiru`
  * `PartOfAHorizontalPortScan`

Los modelos evaluados son:

* `Random Forest`
* `LightGBM`
* `Multilayer Perceptron (MLP)`

Además, se compara el rendimiento de los modelos en el escenario multiclase utilizando características con y sin histogramas de tamaño de paquete e *inter-packet time*.

## Dataset

El proyecto utiliza el conjunto de datos público **IoT-23**, publicado por Stratosphere Laboratory:

https://www.stratosphereips.org/datasets-iot23

Las capturas PCAP originales y los datasets procesados no se incluyen en este repositorio debido a su tamaño. Los scripts permiten reproducir las principales etapas del procesamiento partiendo de una copia local del dataset original.

## Estructura del repositorio

```text
iot23-traffic-classification-tfg/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_processing/
│   ├── split_dir_por_tamano_editcap.sh
│   ├── pcap_to_flow_features.sh
│   ├── packets_to_flow_features.awk
│   ├── run_one_scenario_chunks.sh
│   ├── run_all_pcaps_perfile.sh
│   ├── merge_all_per_pcap_to_one.sh
│   ├── label_per_pcap_by_scenario_streaming.py
│   ├── build_reduced_dataset_streaming.py
│   ├── build_binary_labeled_streaming.py
│   ├── build_binary_reduced_before_smote.py
│   ├── apply_smote_with_proto.py
│   ├── apply_smote_binary_with_proto.py
│   ├── fix_bins_consistency.py
│   └── normalize_histograms_replace.py
│
├── training/
│   ├── train_models_binary.py
│   ├── train_models_5class.py
│   └── train_models_5class_nohist.py
│
└── figures/
    ├── plot_confusion_matrices_appendix.py
    └── plot_appendix_d_hist_ecdf.py
```

## Pipeline de procesamiento

El procesamiento seguido en el trabajo se organiza en las siguientes etapas:

### 1. División de capturas grandes

Las capturas PCAP de gran tamaño pueden dividirse en fragmentos más pequeños mediante:

```bash
./data_processing/split_dir_por_tamano_editcap.sh \
    data/IoTScenarios \
    1000000 \
    25000 \
    chunks_25mb
```

### 2. Extracción de características a nivel de flujo

La extracción se realiza a partir de los paquetes de cada captura utilizando `tshark` y `awk`.

Para procesar un escenario concreto:

```bash
./data_processing/run_one_scenario_chunks.sh \
    data/IoTScenarios/CTU-IoT-Malware-Capture-52-1 \
    "C&C" \
    results/flow_extraction
```

Para procesar varios escenarios definidos en un fichero de mapeo:

```bash
./data_processing/run_all_pcaps_perfile.sh \
    data/IoTScenarios \
    data/mapping_manual.csv \
    results/flow_extraction
```

Las características extraídas incluyen:

* Protocolo.
* Duración del flujo.
* Número de paquetes y bytes en ambos sentidos.
* Histogramas de tamaño de paquete en ambos sentidos.
* Histogramas de *inter-packet time* en ambos sentidos.

### 3. Fusión y etiquetado

Los CSV generados para cada captura se fusionan mediante:

```bash
./data_processing/merge_all_per_pcap_to_one.sh \
    results/flow_extraction/per_pcap \
    results/datasets/flows_merged.csv
```

El etiquetado por escenario se realiza con:

```bash
python data_processing/label_per_pcap_by_scenario_streaming.py \
    --input-dir results/flow_extraction/per_pcap_by_scenario \
    --assigned-labels data/assigned_labels.csv \
    --output results/datasets/dataset_labeled.csv \
    --report results/datasets/labeling_report.csv
```

### 4. Construcción y balanceo de datasets

El dataset multiclase reducido se genera mediante:

```bash
python data_processing/build_reduced_dataset_streaming.py \
    --input results/datasets/dataset_labeled.csv \
    --output results/datasets/dataset_reduced_before_smote.csv \
    --target 100000
```

Posteriormente se aplica balanceo mediante `SMOTE`:

```bash
python data_processing/apply_smote_with_proto.py \
    --input results/datasets/dataset_reduced_before_smote.csv \
    --output results/datasets/dataset_balanced_100k_per_class_withproto.csv
```

Para el escenario binario:

```bash
python data_processing/build_binary_labeled_streaming.py \
    --input results/datasets/dataset_labeled.csv \
    --output results/datasets/dataset_binary_labeled.csv

python data_processing/build_binary_reduced_before_smote.py \
    --input results/datasets/dataset_binary_labeled.csv \
    --output results/datasets/dataset_binary_reduced_before_smote.csv \
    --target-attack 100000

python data_processing/apply_smote_binary_with_proto.py \
    --input results/datasets/dataset_binary_reduced_before_smote.csv \
    --output results/datasets/dataset_binary_balanced_100k_per_class_withproto.csv
```

### 5. Consistencia y normalización de histogramas

La consistencia de los histogramas se corrige mediante:

```bash
python data_processing/fix_bins_consistency.py \
    --input results/datasets/dataset_balanced_100k_per_class_withproto.csv \
    --output results/datasets/dataset_balanced_100k_per_class_withproto_fixed.csv
```

Después se normalizan los histogramas:

```bash
python data_processing/normalize_histograms_replace.py \
    --input results/datasets/dataset_balanced_100k_per_class_withproto_fixed.csv \
    --output results/datasets/dataset_balanced_100k_per_class_withproto_fixed_norm.csv
```

## Entrenamiento de modelos

### Clasificación binaria

```bash
python training/train_models_binary.py \
    --dataset data/dataset_binary_balanced_100k_per_class_withproto_fixed_norm.csv \
    --output-dir results/binary
```

### Clasificación multiclase con histogramas

```bash
python training/train_models_5class.py \
    --dataset data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv \
    --output-dir results/5class
```

### Clasificación multiclase sin histogramas

```bash
python training/train_models_5class_nohist.py \
    --dataset data/dataset_balanced_100k_per_class_nohist.csv \
    --output-dir results/5class_nohist
```

## Generación de figuras

Las matrices de confusión del apéndice de resultados pueden generarse mediante:

```bash
python figures/plot_confusion_matrices_appendix.py \
    --output-dir results/figures/confusion_matrices
```

Las figuras complementarias de histogramas y funciones ECDF se generan mediante:

```bash
python figures/plot_appendix_d_hist_ecdf.py \
    --dataset data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv \
    --output-dir results/figures/appendix_d
```

## Requisitos

### Dependencias de Python

Las dependencias principales pueden instalarse mediante:

```bash
pip install -r requirements.txt
```

### Herramientas externas

La fase de procesamiento de capturas requiere disponer de:

* `bash`
* `awk`
* `tshark`
* `editcap`
* `capinfos`

Las herramientas `tshark`, `editcap` y `capinfos` forman parte del entorno de Wireshark.

## Resultados principales

Los experimentos realizados muestran que la clasificación binaria entre tráfico benigno y tráfico de ataque alcanza un rendimiento muy elevado.

En el escenario multiclase, `Random Forest` obtiene el mejor equilibrio entre rendimiento y coste computacional. La principal dificultad observada se concentra en la separación entre las clases `Okiru` y `PartOfAHorizontalPortScan`, cuyos flujos presentan características estadísticas similares.

La comparación con y sin histogramas muestra que estas características enriquecen el análisis del comportamiento interno de los flujos, aunque la mejora directa en las métricas globales es reducida para los modelos basados en árboles.

## Autor

**Jaime Antona Díaz**
Grado en Ingeniería de Tecnologías de Telecomunicación
Escuela Politécnica Superior
Universidad Autónoma de Madrid

## Contexto académico

Este repositorio recoge el código desarrollado para el Trabajo Fin de Grado titulado:

**Detección de anomalías en redes IoT mediante la combinación de técnicas estadísticas y de aprendizaje automático**
