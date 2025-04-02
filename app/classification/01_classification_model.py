import sys
import os
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Manejo especial para urllib3 en Python 3.12
try:
    import urllib3
except ModuleNotFoundError:
    os.system(f"{sys.executable} -m pip install urllib3>=2.0.0")
    import urllib3

# Configuración inicial de la página
st.set_page_config(
    page_title="Predicción de Marketing",
    page_icon="📊",
    layout="wide"
)

# Configuración de paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Sube 2 niveles desde "app/classification"
MODEL_PATH = BASE_DIR / "models" / "classification" / "RandomForestClassifier.pkl"

@st.cache_resource
def load_model():
    """Carga el modelo con manejo de errores"""
    try:
        model, threshold = joblib.load(MODEL_PATH)
        return model, threshold
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        st.stop()

def get_default_values():
    """Valores por defecto para los inputs"""
    return {
        "MntWines": 305.00,
        "Spent": 607.00,
        "Income": 51954.00,
        "NumCatalogPurchases": 2,
        "MntMeatProducts": 66.0,
        "NumWebPurchases": 5,
        "Kidhome": 0,
        "Child_Home": 0
    }

def user_input_features(defaults):
    """Crea los controles para entrada de usuario"""
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            MntWines = st.number_input("🍷 Importe gastado en vino (últimos 2 años)", 0.0, 1500.0, defaults["MntWines"], step=1.0)
            MntMeatProducts = st.number_input("🥩 Gastado en carne(últimos 2 años)", 0.0, 500.0, defaults["MntMeatProducts"], step=1.0)
            Spent = st.number_input("💳 Importe total gastado en todos los productos (últimos 2 años)", 0.0, 3000.0, defaults["Spent"], step=1.0)
        
        with col2:
            NumCatalogPurchases = st.slider("📖 Compras por catálogo", 0, 20, defaults["NumCatalogPurchases"])
            NumWebPurchases = st.slider("🛒 Compras online", 0, 25, defaults["NumWebPurchases"])
            Income = st.number_input("💰 Ingresos anuales", 0.0, 200000.0, defaults["Income"], step=1.0)
        
        Kidhome = st.selectbox("👶 Número de hijos", [0, 1, 2], index=defaults["Kidhome"])
        # Definir "¿Hay niños en casa?" automáticamente en función de "Número de hijos"
        Child_Home = 1 if Kidhome > 0 else 0
        # opciones = ["No", "Sí"]
        # Child_Home = st.radio("🏡 ¿Hay niños en casa?", opciones, index=defaults["Child_Home"])
        # Child_Home = 1 if Child_Home == "Sí" else 0

        submitted = st.form_submit_button("🔮 Predecir")
        
    return submitted, np.array([[MntWines, Spent, Income, NumCatalogPurchases, MntMeatProducts, NumWebPurchases, Kidhome, Child_Home]])

def display_results(proba, prediction):
    """Muestra los resultados de la predicción"""
    st.subheader("📢 Resultado de la Predicción")
    st.metric("Probabilidad de aceptar", f"{proba:.2%}")
    
    if prediction == 1:
        st.success("✅ El cliente probablemente ACEPTARÁ una campaña de marketing.")
    else:
        st.error("❌ El cliente probablemente NO aceptará ninguna campaña de marketing.")

def main():
    st.title("📊 Predicción de Aceptación de Campañas de Marketing")
    st.write("Ingrese los valores para predecir si aceptará al menos una campaña.")
    
    # Cargar el modelo
    if not MODEL_PATH.exists():
        st.error(f"❌ Error: El modelo no se encuentra en: {MODEL_PATH}")
        st.stop()
    
    model, threshold = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Instrucciones")
        st.markdown("""
        1. Introduzca los valores en los campos  
        2. Haga clic en **Predecir**  
        3. Verá la probabilidad y predicción  
        """)
        st.header("⚙️ Configuración del Modelo")
        st.write(f"Threshold actual: {threshold:.2f}")
    
    # Entrada de usuario
    defaults = get_default_values()
    submitted, user_data = user_input_features(defaults)
    
    # Predicción
    if submitted:
        try:
            proba = model.predict_proba(user_data)[:, 1][0]
            prediction = 1 if proba >= threshold else 0
            display_results(proba, prediction)
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")

if __name__ == "__main__":
    main()