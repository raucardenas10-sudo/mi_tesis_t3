# 🛡️ Detección Híbrida Multilingüe de Phishing mediante Transformers y SVM (FISI-UNMSM)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-INT8_Quantized-orange.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/XLM--RoBERTa-Base-green.svg)](https://huggingface.co/xlm-roberta-base)
[![Classifier](https://img.shields.io/badge/SVM-Kernel_RBF-purple.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-UNMSM_Academic-lightgrey.svg)]()

> **Proyecto de Tesis II — Avance PI 11 + PI 12 (Semana 11-12)**  
> **Escuela Profesional de Ingeniería de Sistemas — Grupo A (Aporte Conceptual / Teórico)**  
> **Facultad de Ingeniería de Sistemas e Informática (FISI)**  
> **Universidad Nacional Mayor de San Marcos (UNMSM)**  

---

## 📌 Datos de la Investigación

* **Título de la Tesis:** Modelo híbrido multilingüe basado en XLM-RoBERTa y Máquinas de Vectores de Soporte para la detección de phishing en entornos universitarios.
* **Tesista:** Quinto Pazce, Daniel
* **Asesor de Tesis:** Mg. Daniel Quinto Pazce
* **Sección de Clase:** G6 (2026-II)
* **Comité Evaluador de Expertos (Juicio de Expertos):**
  1. **Dr. Cesar Augusto Angulo Calderon** (FISI – UNMSM, DACC)
  2. **Mg. Heddy Liliana Colca García** (FISI – UNMSM)
  3. **Ing. Gelber Christian Uscuchagua Flores** (FISI – UNMSM)

---

## 📋 Resumen de la Propuesta

Esta investigación propone un pipeline híbrido para la clasificación predictiva de correos electrónicos de *phishing* e hipervínculos maliciosos en idioma español e inglés. El artefacto combina la potencia de representación semántica contextual de la red neuronal Transformer **XLM-RoBERTa** (vía vectorización del token de clasificación `[CLS]` cuantizado a precisión INT8) con la capacidad de separación no lineal de una **Máquina de Vectores de Soporte (SVM)** optimizada con Kernel RBF.

El sistema resuelve la brecha de transferencia *cross-lingual* al adaptar conocimiento entrenado masivamente en inglés ($18,000$ muestras) hacia el dominio institucional en español con un volumen acotado de datos ($k = 250$ ejemplos sintéticos FISI).

---

## 🏗️ Arquitectura del Pipeline Extremo a Extremo
+-------------------+      +---------------------------------+      +------------------------+
| Correo Entrante   | ---> | Tokenización XLM-RoBERTa (INT8) | ---> | Extracción Embedding   |
| (Asunto + Cuerpo) |      | (max_length = 128)              |      | Token [CLS] (768 dim)  |
+-------------------+      +---------------------------------+      +------------------------+
|
v
+-------------------+      +---------------------------------+      +------------------------+
| Diagnóstico Final | <--- | Clasificación Supervisada       | <--- | Estandarización        |
| (Phishing / HAM)  |      | SVM (Kernel RBF, C=1.0)         |      | StandardScaler         |
+-------------------+      +---------------------------------+      +------------------------+
---

## 📊 Cuadros Comparativos de Evidencia Empírica

### 1. Rendimiento Global del Modelo Seleccionado vs. Líneas Base (Conjunto Ciego)

Evaluación realizada sobre una prueba ciega independiente de $600$ correos en español ($300$ Phishing / $300$ HAM) post-deduplicación global MD5.

| Modelo / Estrategia | Datos de Entrenamiento | F1-Score | Precision (FP ↓) | Recall | ROC-AUC | Estado / Rol |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot Real (Directo)** | 18,000 EN (0 ES) | 0.3339 | 0.3619 | 0.3496 | 0.3496 | Punto de partida sin adaptación |
| **Línea Base A (TF-IDF + Reg. Logística)** | 1,400 ES (0 EN) | 0.9299 | 0.9544 | 0.9067 | 0.9727 | Baseline Monolingüe Clásico |
| **Línea Base B (XLM-R + SVM Mono-idioma)** | 1,400 ES (0 EN) | 0.9273 | 0.9640 | 0.8933 | 0.9765 | Baseline Transformer Mono-ES |
| **Propuesta Tesis (Cross-Lingual)** | **18,000 EN + 1,400 ES** | **0.9343** | **0.9712** | **0.9000** | **0.9756** | **Modelo Seleccionado (Mayor Precisión)** |

### 2. Curva de Transferencia Few-Shot Cross-Lingual ($k$-Samples / 5 Semillas)

Evaluación multisemilla (semillas: `42, 123, 777, 999, 2024`) variando el número de muestras $k$ en español agregadas al entrenamiento base en inglés:

| $k$-Samples en Español | Accuracy Promedio (%) | F1-Score Weighted Promedio (%) | Observación Metrológica |
| :---: | :---: | :---: | :--- |
| **$k = 0$ (Zero-Shot)** | $38.17\% \pm 0.00\%$ | $37.85\% \pm 0.00\%$ | Brecha severa de transferencia pura |
| **$k = 50$** | $67.87\% \pm 2.77\%$ | $67.53\% \pm 2.77\%$ | Inicio de adaptación semántica |
| **$k = 100$** | $82.67\% \pm 1.93\%$ | $82.52\% \pm 1.95\%$ | Incremento acelerado de efectividad |
| **$k = 250$** | **$90.80\% \pm 0.45\%$** | **$90.78\% \pm 0.46\%$** | **Punto de Convergencia ($F1 \ge 90\%$)** |
| **$k = 500$** | $92.80\% \pm 0.07\%$ | $92.79\% \pm 0.07\%$ | Zona de estabilidad asintótica |
| **$k = 1000$** | $93.73\% \pm 0.43\%$ | $93.73\% \pm 0.43\%$ | Alto desempeño multilingüe |
| **$k = 1400$** | $93.67\% \pm 0.00\%$ | $93.66\% \pm 0.00\%$ | Saturación del pool de entrenamiento |

### 3. Benchmarking de Latencia de Extremo a Extremo (RNF-01)

Medición sobre $100$ ejecuciones individuales (sin *minibatches*) en procesador Intel Core i5-11400H @ 2.70GHz (CPU pura):

| Indicador Metrológico | Medición Inicial (App) | Escenario Base (FP32) | Escenario Optimizado (INT8 + max_len=128) | Meta RNF | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mediana Latencia ($P_{50}$)** | ~404.84 ms | 130.23 ms | **44.55 ms** | $< 50\text{ ms}$ | **CUMPLE** |
| **Percentil 95 ($P_{95}$)** | No reportado | 207.43 ms | **67.95 ms** | N/A | Evaluado |
| **Latencia GUI Streamlit (Warm State)** | 400.00 ms (Cold) | N/A | **50 ms - 130 ms** | $< 300\text{ ms}$ | **JUSTIFICADA (Asíncrona)** |

---

## 📁 Estructura del Repositorio
mi_tesis_t3/
├── data/
│   ├── processed/              # Corpus unificado deduplicado (20,000 registros)
│   │   └── corpus_unificado_20k.csv
│   └── raw/                    # Fuentes primarias (Nazario, Enron, SpearPhishMX, FISI)
│       ├── dataset_espanol_mendeley.csv
│       ├── dataset_ingles_28k.csv
│       └── fisi_synthetic_200.csv
├── models/                     # Artefactos entrenados serializados
│   ├── clasificador_phishing.joblib (SVM Kernel RBF)
│   └── scaler.joblib           (StandardScaler estandarizado)
├── reports/                    # Gráficas y reportes cuantitativos exportados
│   ├── curva_k_samples.png
│   ├── curva_roc_auc.png
│   └── resultados_curva_k.csv
├── scripts/                    # Tubería de código ejecutable y auditorías
│   ├── unificar_datasets.py   (Deduplicación MD5 y saneamiento)
│   ├── generar_fisi_sintetico.py (Generación sintética única)
│   ├── extraer_embeddings.py  (Vectorización Token [CLS] XLM-RoBERTa)
│   ├── entrenar_pipeline.py   (Entrenamiento y evaluación del pipeline)
│   ├── auditoria_curva_k.py   (Benchmark multisemilla 5 seeds)
│   ├── auditoria_crosslingual.py (Comparativa vs Líneas Base)
│   ├── auditoria_latencia_optimizada.py (Benchmarking RNF-01)
│   ├── auditoria_duplicados.py (Auditoría de unicidad)
│   └── auditoria_few_vs_zero.py (Demostración de brecha Zero-Shot)
├── app.py                      # Prototipo visual interactivo (Streamlit GUI)
├── requirements.txt            # Dependencias del proyecto
├── .gitignore                  # Reglas de exclusión de Git
└── README.md                   # Documentación principal

---

## 🚀 Instrucciones para Reproducir el Proyecto

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone [https://github.com/TU_USUARIO/mi_tesis_t3.git](https://github.com/TU_USUARIO/mi_tesis_t3.git)
cd mi_tesis_t3
python -m venv venv
.\venv\Scripts\Activate.ps1   # En Windows PowerShell
pip install -r requirements.txt


```
### 2. Ejecutar auditorías de validación Auditoría de Curva Cross-Lingual
```bash

- Auditoría de Curva Cross-Lingual ($k$-samples):
python scripts/auditoria_curva_k.py

- Benchmarking Metrológico de Latencia (100 Correos):
python scripts/auditoria_latencia_optimizada.py

- Evaluación Comparativa contra Líneas Base:
python scripts/auditoria_crosslingual.py

-  Desplegar la Interfaz Gráfica Demostrativa
streamlit run app.py

```
### 3. Video Demostrativo
```bash
Video Demostrativo
Puedes visualizar la demostración en vivo del prototipo funcional, incluyendo la explicación técnica de las subsanaciones y la evaluación interactiva en tiempo real en la interfaz de Streamlit, en el siguiente enlace:

🎥 Ver Video de Demostración del Prototipo en Google Drive(Nota: https://drive.google.com/file/d/1drTNcVc0jmHbe5qzodSNt57JXcnlXlYB/view?usp=sharing).
