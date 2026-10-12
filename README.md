# 🛡️ Detector Híbrido Cross-Lingual de Phishing (FISI-UNMSM)

Este repositorio contiene la implementación y pipeline experimental del prototipo de detección de correos de phishing en español e inglés mediante **XLM-RoBERTa (Mean Pooling)** y **Regresión Logística**.

## 📊 Resultados de Evaluación

| Modelo | F1-Score | Precisión | Recall | Accuracy | Latencia | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | **0.9128** | **0.9189** | **0.9067** | **0.9133** | **0.00 ms** | **SELECCIONADO (Cumple RNF01/02)** |
| Random Forest | 0.9100 | 0.9273 | 0.8933 | 0.9117 | 0.10 ms | CUMPLE |
| SVM (RBF) | 0.8255 | 0.8311 | 0.8200 | 0.8267 | 1.34 ms | DESCARTADO |
| Gaussian Naive Bayes | 0.5870 | 0.5374 | 0.6467 | 0.5450 | 0.01 ms | DESCARTADO |

## 📁 Estructura del Repositorio

* `data/raw/`: Corpus original de correos (Inglés + Español + Sintéticos FISI).
* `data/processed/`: Vectores numéricos densos extraídos mediante XLM-RoBERTa (Mean Pooling vs [CLS]).
* `models/`: Modelo entrenado y serializado (`clasificador_phishing.joblib`).
* `scripts/`: Scripts reproducibles de preprocesamiento, extracción y entrenamiento.
* `app.py`: Interfaz gráfica interactiva desarrollada en Streamlit.

## 🚀 Instrucciones de Ejecución Local

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt