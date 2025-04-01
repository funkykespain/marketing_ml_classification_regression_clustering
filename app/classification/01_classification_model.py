import os
import streamlit as st
import joblib
import numpy as np

# Carga del modelo con ruta segura
try:
    model_path = os.path.join(os.path.dirname(__file__), 'models/RandomForestClassifier.pkl')
    model, threshold = joblib.load(model_path)
except Exception as e:
    st.error(f"Error cargando el modelo: {str(e)}")
    st.stop()

# Título y descripción de la aplicación
st.title("📊 Predicción de Aceptación de Campañas de Marketing")
st.write("""
Ingrese los valores de las características del cliente para predecir si aceptará al menos una campaña de marketing.
""")

# Sidebar con instrucciones
st.sidebar.header("ℹ️ Instrucciones")
st.sidebar.write("""
1️⃣ Introduzca los valores en los campos correspondientes.  
2️⃣ Haga clic en **Predecir** para obtener el resultado.  
3️⃣ Se mostrará la probabilidad de aceptación y la predicción final.  
""")

# Diccionario con los valores por defecto (media de las variables)
default_values = {
    "MntWines": 305.03,
    "Spent": 606.71,
    "Income": 51954.46,
    "NumCatalogPurchases": 2,
    "MntMeatProducts": 65.5,
    "NumWebPurchases": 5,
    "Kidhome": 0,
    "Child_Home": 0
}

# Inputs del usuario
MntWines = st.number_input("🍷 Importe gastado en vino (últimos 2 años)", min_value=0.0, max_value=1500.0, value=default_values["MntWines"])
Spent = st.number_input("💳 Importe total gastado (últimos 2 años)", min_value=0.0, max_value=3000.0, value=default_values["Spent"])
Income = st.number_input("💰 Ingresos anuales", min_value=0.0, max_value=200000.0, value=default_values["Income"])
NumCatalogPurchases = st.slider("📖 Compras realizadas por catálogo", min_value=0, max_value=20, value=default_values["NumCatalogPurchases"])
MntMeatProducts = st.number_input("🥩 Importe gastado en carne (últimos 2 años)", min_value=0.0, max_value=500.0, value=default_values["MntMeatProducts"])
NumWebPurchases = st.slider("🛒 Compras realizadas online", min_value=0, max_value=25, value=default_values["NumWebPurchases"])
Kidhome = st.selectbox("👶 Número de hijos en el hogar", [0, 1, 2], index=default_values["Kidhome"])
Child_Home = st.radio(
    "🏡 ¿Hay niños en casa?",
    options=[("No", 0), ("Sí", 1)],
    index=default_values["Child_Home"]
)[1]

# Convertir los datos en un array para el modelo
user_data = np.array([[MntWines, Spent, Income, NumCatalogPurchases, MntMeatProducts, NumWebPurchases, Kidhome, Child_Home]])

# Botón para hacer la predicción
if st.button("🔮 Predecir"):
    try:
        # Obtener la probabilidad de aceptación de la campaña
        proba = model.predict_proba(user_data)[:, 1][0]
        
        # Aplicar el threshold para la clasificación final
        prediction = 1 if proba >= threshold else 0

        # Mostrar resultados
        st.subheader("📢 Resultado de la Predicción")
        st.write(f"**Probabilidad de aceptar una campaña:** `{proba:.2%}`")
        
        if prediction == 1:
            st.success("✅ El cliente probablemente ACEPTARÁ una campaña de marketing.")
        else:
            st.error("❌ El cliente probablemente NO aceptará ninguna campaña de marketing.")
    except Exception as e:
        st.error(f"Error al hacer la predicción: {str(e)}")