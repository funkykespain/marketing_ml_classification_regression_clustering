# 📊 **Marketing ML: Customer Personality Analysis**  
**Machine Learning aplicado a Marketing** (Clasificación, Regresión y próximamente Clustering)  

![Banner](assets/banner.png)  

## 📌 **Descripción del Proyecto**  
Este repositorio contiene un **análisis de datos y modelos predictivos** basados en el dataset **[Customer Personality Analysis de Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/)**.  
El objetivo es predecir comportamientos de clientes (como respuesta a campañas o gasto) usando:  
- **🔵 Clasificación**: ¿El cliente aceptará una futura oferta?  
- **🔴 Regresión**: ¿Cuánto gastará el cliente en el negocio conociendo sus datos personales?  
- **🟢 (En progreso) Clustering**: Segmentación de clientes por comportamiento.  

---

## 📂 **Estructura del Repositorio** 
```
marketing_ml_classification_regression_clustering/
├── app/
│ ├── classification/
│ │ └── 01_classification_model.py # App Streamlit (Clasificación)
│ ├── regression/
│ │ └── 02_regression_model.py # App Streamlit (Regresión)
│ └── assets/ # Imágenes para visualización
├── notebooks/
│ ├── 01_load_and_clean_data.ipynb # Carga y exploración inicial de los datos
│ ├── 02_data_cleaning.ipynb # Limpieza de datos
│ ├── 03_feature_engineering.ipynb # Ingeniería de características
│ ├── 04_explore_data.ipynb # Análisis exploratorio de datos (EDA)
│ ├── 05_classification_model_training_evaluation.ipynb # Modelado y evaluación (Clasificación)
│ ├── 06_regression_model_training_evaluation.ipynb # Modelado y evaluación (Regresión)
├── assets/
│ ├── banner.png
│ ├── eda_income_vs_spent.png
│ ├── curva_roc.png
│ └── real_vs_predicho.png
└── README.md
```

---

## 🧠 **Análisis y Modelos**  

### 1. **Carga, Limpieza y Análisis Exploratorio de Datos (EDA)**  
Se realiza una exploración inicial de los datos para comprender patrones clave como ingresos, gastos y respuestas a campañas.  
📌 [Ver notebook de EDA](notebooks/04_explore_data.ipynb)  

**Hallazgos clave**:  
✔️ Relación entre **gasto en vino** y **nivel educativo**.  
✔️ Clientes con hijos tienden a responder menos a campañas.  

![EDA](assets/eda_income_vs_spent.png) 

---

### 2. **Modelo de Clasificación**  
**Objetivo**: Predecir si un cliente aceptará una campaña de marketing.  
📌 [Ver notebook de clasificación](notebooks/05_classification_model_training_evaluation.ipynb)  

**Modelos probados:**  
✔️ Logistic Regression  
✔️ KNN  
✔️ Decision Tree  
✔️ Random Forest (Modelo ganador: Accuracy **84.2%**)  

**Variables clave**:  
📊 Gasto total y en 2 productos específicos, compras online, compras por catálogo, ingresos y número de hijos.  

```python
# Ejemplo en Streamlit (app/classification/)
st.write("Probabilidad de aceptación: ", model.predict_proba(input_data)[0][1])
```

![Classification](assets/curva_roc.png) 

[![Abrir en Streamlit](https://img.shields.io/badge/Abrir_en_Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://marketingmlclassificationregressionclustering-2apkcvnbir7q4iuc.streamlit.app) 

---

### 3. **Modelo de Regresión**  
**Objetivo**: Predecir el gasto total de un cliente en los próximos 2 años en el negocio.
📌 [Ver notebook de regresión](notebooks/06_regression_model_training_evaluation.ipynb)

**Modelos probados:**  
✔️ Linear Regression 
✔️ Ridge  
✔️ Lasso 
✔️ Random Forest (Modelo ganador: **R² Score 0.8214**)

**Variables clave**:  
📊 Ingresos, antigüedad como cliente, edad, hijos y nivel de estudios.

```python
# Ejemplo en Streamlit (app/regression/)
st.metric("Gasto predicho", f"${prediction[0]:.2f}")
```

![Regression](assets/real_vs_predicho.png) 

[![Abrir en Streamlit](https://img.shields.io/badge/Abrir_en_Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://funkykespain-marketing--appregression02-regression-model-rjzlvk.streamlit.app) 

---

### 4. **Próximos Pasos: Clustering**  
**Objetivo**: Segmentar clientes en grupos con comportamientos similares. 
📌 [Notebook próximamente disponible] 
- **Dataset**: El mismo ([Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/)).  
- **Técnicas a probar**: 
   - K-Means
   - DBSCAN
   - Modelos jerárquicos
- **Variables candidatas**:
✔️ Todas las usadas en classification y regression.   

---

## 🚀 **Cómo Ejecutar el Proyecto**  
1. **Apps Streamlit**:  
   ```bash
   streamlit run app/classification/01_classification_model.py
   streamlit run app/regression/02_regression_model.py
   ```  
2. **Notebooks**: Abrir en Jupyter o Google Colab.  

---

## 🌟 **Conclusión**  
Este proyecto demuestra cómo **el Machine Learning puede optimizar estrategias de marketing**, desde predecir respuestas hasta segmentar clientes. 
🚀 ¡Próximamente: clustering!  
💡 **¿Preguntas?** ¡Abre un *issue* o contáctame!  

--- 

### 🔗 **Enlaces**  
- [Dataset en Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/)  
- [Repositorio](https://github.com/funkykespain/marketing_ml_classification_regression_clustering/)  
- [Predicción de Aceptación de Campañas de Marketing](https://marketingmlclassificationregressionclustering-2apkcvnbir7q4iuc.streamlit.app/) 
- [Predicción de Gasto de Clientes](https://funkykespain-marketing--appregression02-regression-model-rjzlvk.streamlit.app/) 

--- 

✨ **¡Si te gusta el proyecto, déjale una ⭐!**  

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
