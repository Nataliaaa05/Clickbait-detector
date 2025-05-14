# Clickbait-detector

# 📢 Clickbait Detection - Practice IV

## 📘 Overview

"El clickbait es un método para generar titulares, especialmente en línea, que omite deliberadamente parte de la información con el objetivo de generar curiosidad al crear una brecha informativa, atrayendo así la atención de los lectores y haciendo que hagan clic."

Este proyecto tiene como objetivo construir un modelo de **Machine Learning** capaz de detectar automáticamente si un teaser (titular breve) es clickbait o no.

---

## 🎯 Objective

Desarrollar un sistema de clasificación automática que, basado en un texto breve (teaser), determine si se trata de un ejemplo de clickbait utilizando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático.

---

## 🧩 Dataset

- **Archivo de entrenamiento:** `TA1C_dataset_detection_train.csv`
- **Archivo de prueba:** `TA1C_dataset_detection_dev.csv`
- **Características:**
  - `Teaser Text`: Texto del titular (característica de entrada).
  - `Tag Value`: Etiqueta binaria (0 = No clickbait, 1 = Clickbait).

---

## ⚙️ Methodology

### 1. **Preprocesamiento y Normalización del texto**
- Tokenización
- Limpieza de texto (eliminación de signos, minúsculas, etc.)
- Eliminación de stopwords
- Lematización

### 2. **Representación del texto**
Se utilizaron varias combinaciones de:
- N-gramas: unigramas, bigramas, trigramas
- Técnicas: frecuencia, binaria, TF-IDF
- Reducción de dimensionalidad opcional con **TruncatedSVD**

### 3. **Modelado y entrenamiento**
Se probaron varios algoritmos de machine learning:
- Regresión logística
- Naive Bayes
- Random Forest
- Gradient Boosting
- Multilayer Perceptron

### 4. **Validación y ajuste**
- División del dataset: 75% entrenamiento, 25% validación (`train_test_split`)
- Validación cruzada estratificada de 5 pliegues (`StratifiedKFold`)
- Métrica principal: **F1-Score Macro**

### 5. **Manejo de desbalanceo de clases**
- Distribución: 71% No clickbait vs. 29% Clickbait
- Métodos aplicados:
  - Oversampling
  - Undersampling  
  *(utilizando la librería [imbalanced-learn](https://imbalanced-learn.org/stable/))*

---

## 📈 Resultados

Para cada experimento se registró un informe de clasificación, destacando las combinaciones de:
- Métodos de normalización
- Técnicas de representación
- Algoritmos y sus hiperparámetros

El modelo con mejor desempeño fue retrainado usando todo el dataset de entrenamiento y utilizado para predecir el conjunto de prueba.

---

## 📁 Output

El archivo final de predicciones se genera como:

- **Nombre:** `detection.csv`
- **Formato:** CSV separado por comas
- **Columnas:** `Tweet ID`, `Tag Value`

---

## 📑 Entregables

- `detection.csv`: archivo de predicciones sobre el test set
- Código fuente en Python
- Reporte en PDF con:
  - Descripción del problema
  - Técnicas aplicadas
  - Resultados de los experimentos
  - Tabla comparativa de modelos

---

## 🧪 Reproducibilidad

Asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install pandas scikit-learn nltk imbalanced-learn
