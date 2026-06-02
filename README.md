# Clasificación de tráfico IoT a partir del dataset IoT-23

Este repositorio contiene el código desarrollado para mi Trabajo Fin de Grado, realizado en la Escuela Politécnica Superior de la Universidad Autónoma de Madrid.

El trabajo parte del dataset público [IoT-23](https://www.stratosphereips.org/datasets-iot23) y estudia la clasificación de tráfico benigno y malicioso en entornos IoT. El procesamiento se realiza a partir de las capturas PCAP originales, extrayendo características de flujo e histogramas de tamaños de paquete y tiempos entre paquetes. Sobre los datasets construidos se evalúan tres modelos: Random Forest, LightGBM y Multilayer Perceptron.

## Contenido del repositorio

El código está dividido en tres carpetas:

```text
data_processing/   Procesamiento de capturas y construcción de datasets
training/          Entrenamiento y evaluación de modelos
figures/           Generación de figuras utilizadas en la memoria
```

La carpeta `data_processing/` contiene los scripts utilizados para:

* dividir capturas grandes en fragmentos manejables;
* extraer características desde PCAP mediante `tshark` y `awk`;
* fusionar y etiquetar los flujos obtenidos;
* construir los datasets binario y multiclase;
* aplicar balanceo y normalizar los histogramas.

La carpeta `training/` incluye los experimentos correspondientes a:

* clasificación binaria: `Benign` frente a `Attack`;
* clasificación multiclase con histogramas;
* clasificación multiclase sin histogramas.

Por último, `figures/` contiene los scripts empleados para generar las matrices de confusión y las comparaciones de histogramas y funciones ECDF incluidas en los apéndices de la memoria.

## Datos utilizados

Las capturas originales de IoT-23 y los datasets intermedios no se incluyen en este repositorio debido a su tamaño. Para reproducir el procesamiento es necesario descargar previamente el dataset desde su página oficial y disponer de una copia local de las capturas PCAP.

Las cinco clases consideradas en el escenario multiclase son:

```text
Benign
C&C
DDoS
Okiru
PartOfAHorizontalPortScan
```

## Dependencias

Las dependencias de Python utilizadas en los scripts están indicadas en `requirements.txt` y pueden instalarse mediante:

```bash
pip install -r requirements.txt
```

Para la extracción de características desde las capturas también es necesario disponer de:

```text
bash
awk
tshark
editcap
capinfos
```

`tshark`, `editcap` y `capinfos` forman parte de Wireshark.

## Procesamiento de las capturas

Las capturas especialmente grandes pueden dividirse antes del procesamiento:

```bash
./data_processing/split_dir_por_tamano_editcap.sh \
    data/IoTScenarios \
    1000000 \
    25000 \
    chunks_25mb
```

La extracción de características para un escenario concreto puede ejecutarse mediante:

```bash
./data_processing/run_one_scenario_chunks.sh \
    data/IoTScenarios/CTU-IoT-Malware-Capture-52-1 \
    "C&C" \
    results/flow_extraction
```

También se incluye un script para procesar varios escenarios a partir de un fichero de mapeo:

```bash
./data_processing/run_all_pcaps_perfile.sh \
    data/IoTScenarios \
    data/mapping_manual.csv \
    results/flow_extraction
```

La extracción genera, para cada flujo, variables agregadas de volumen y duración, junto con histogramas en ambos sentidos de la comunicación.

## Construcción de los datasets

A partir de los ficheros obtenidos en la extracción, los scripts de esta carpeta permiten crear las dos variantes utilizadas en los experimentos:

* dataset binario;
* dataset multiclase de cinco clases.

En ambos casos se incluyen etapas de reducción de clases mayoritarias, balanceo mediante SMOTE y validación de consistencia de los histogramas.

Ejemplo de balanceo del dataset multiclase:

```bash
python data_processing/apply_smote_with_proto.py \
    --input results/datasets/dataset_reduced_before_smote.csv \
    --output results/datasets/dataset_balanced_100k_per_class_withproto.csv
```

Ejemplo de corrección y normalización de histogramas:

```bash
python data_processing/fix_bins_consistency.py \
    --input results/datasets/dataset_balanced_100k_per_class_withproto.csv \
    --output results/datasets/dataset_balanced_100k_per_class_withproto_fixed.csv

python data_processing/normalize_histograms_replace.py \
    --input results/datasets/dataset_balanced_100k_per_class_withproto_fixed.csv \
    --output results/datasets/dataset_balanced_100k_per_class_withproto_fixed_norm.csv
```

## Entrenamiento

Los experimentos principales pueden ejecutarse con los siguientes scripts:

```bash
python training/train_models_binary.py \
    --dataset data/dataset_binary_balanced_100k_per_class_withproto_fixed_norm.csv \
    --output-dir results/binary
```

```bash
python training/train_models_5class.py \
    --dataset data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv \
    --output-dir results/5class
```

```bash
python training/train_models_5class_nohist.py \
    --dataset data/dataset_balanced_100k_per_class_nohist.csv \
    --output-dir results/5class_nohist
```

Los scripts generan los resultados de evaluación, los tiempos de entrenamiento e inferencia y los ficheros auxiliares utilizados para comparar los modelos.

## Figuras de la memoria

Las matrices de confusión del apéndice pueden regenerarse con:

```bash
python figures/plot_confusion_matrices_appendix.py \
    --output-dir results/figures/confusion_matrices
```

La comparación complementaria entre las clases `Okiru` y `PartOfAHorizontalPortScan` se genera mediante:

```bash
python figures/plot_appendix_d_hist_ecdf.py \
    --dataset data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv \
    --output-dir results/figures/appendix_d
```

## Trabajo Fin de Grado

**Título:** Detección de anomalías en redes IoT mediante la combinación de técnicas estadísticas y de aprendizaje automático
**Autor:** Jaime Antona Díaz
**Titulación:** Grado en Ingeniería de Tecnologías de Telecomunicación
**Universidad:** Universidad Autónoma de Madrid
**Curso académico:** 2025--2026
