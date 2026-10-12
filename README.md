#  Detector Híbrido Cross-Lingual de Phishing (FISI-UNMSM)

Este repositorio contiene la implementación y pipeline experimental del prototipo de detección de correos de phishing en español e inglés mediante **XLM-RoBERTa (Mean Pooling)** y **Regresión Logística**.

### Comparativa de Extracción Semántica (XLM-RoBERTa)

| Estrategia de Vectorización | F1-Score | Precisión | Recall | Representación Contextual |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Pooling (Seleccionado)** | **0.9128** | **0.9189** | **0.9067** | **Promedio ponderado de la secuencia completa (Captura correos largos)** |
| Token `[CLS]` (Baseline) | 0.8845 | 0.8910 | 0.8780 | Vector del primer token (Pérdida de contexto en mensajes extensos) |


##  Resultados de Evaluación

| Modelo | F1-Score | Precisión | Recall | Accuracy | Latencia | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | **0.9128** | **0.9189** | **0.9067** | **0.9133** | **0.00 ms** | **SELECCIONADO (Cumple RNF01/02)** |
| Random Forest | 0.9100 | 0.9273 | 0.8933 | 0.9117 | 0.10 ms | CUMPLE |
| SVM (RBF) | 0.8255 | 0.8311 | 0.8200 | 0.8267 | 1.34 ms | DESCARTADO |
| Gaussian Naive Bayes | 0.5870 | 0.5374 | 0.6467 | 0.5450 | 0.01 ms | DESCARTADO |

##  Estructura del Repositorio

* `data/raw/`: Corpus original de correos (Inglés + Español + Sintéticos FISI).
* `data/processed/`: Vectores numéricos densos extraídos mediante XLM-RoBERTa (Mean Pooling vs [CLS]).
* `models/`: Modelo entrenado y serializado (`clasificador_phishing.joblib`).
* `scripts/`: Scripts reproducibles de preprocesamiento, extracción y entrenamiento.
* `app.py`: Interfaz gráfica interactiva desarrollada en Streamlit.

##  Instrucciones de Ejecución Local

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt

---

## 📹 Video Demostrativo

Puedes visualizar la demostración en vivo del prototipo funcional (duración 2:48 min), incluyendo la explicación técnica y la evaluación interactiva en tiempo real, en el siguiente enlace:

🎥 **[Ver Video de Demostración del Prototipo en Google Drive](https://drive.google.com/file/d/1eKhYWYmdec4A5qzcgEIGPHwU9BPPSIeY/view?usp=sharing)**   