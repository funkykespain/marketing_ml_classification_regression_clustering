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

# # Configuración de paths
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# MODEL_PATH = BASE_DIR / "models" / "classification" / "RandomForestClassifier.pkl"

# # Obtener la raíz del repositorio desde la variable de entorno de Streamlit Cloud
# REPO_ROOT = Path(os.getenv("STREAMLIT_REPO_ROOT", Path(__file__).resolve().parent.parent))
# MODEL_PATH = REPO_ROOT / "models" / "classification" / "RandomForestClassifier.pkl"

# Determinar la ruta base correctamente
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Sube 2 niveles desde "app/classification"

MODEL_PATH = BASE_DIR / "models" / "classification" / "RandomForestClassifier.pkl"

# Verificar si el modelo existe antes de cargarlo
if not MODEL_PATH.exists():
    st.error(f"❌ Error: El modelo no se encuentra en: {MODEL_PATH}")
else:
    st.success(f"✅ Modelo encontrado en: {MODEL_PATH}")

# # Definir la raíz del repositorio manualmente en función del entorno de ejecución
# if "STREAMLIT_ENV" in os.environ:  # Esto significa que está en Streamlit Cloud
#     BASE_DIR = Path("/mount/src/marketing_ml_classification_regression_clustering")
# else:
#     BASE_DIR = Path(__file__).resolve().parent.parent  # En local, mantiene la estructura original

# MODEL_PATH = BASE_DIR / "models" / "classification" / "RandomForestClassifier.pkl"

# # Comprobar si el archivo realmente existe
# if not MODEL_PATH.exists():
#     st.error(f"El modelo no se encuentra en: {MODEL_PATH}")
# else:
#     st.success(f"Modelo encontrado en: {MODEL_PATH}")

@st.cache_resource
def load_model(MODEL_PATH):
    """Carga el modelo con manejo de errores"""
    try:
        model, threshold = joblib.load(MODEL_PATH)
        return model, threshold
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        st.stop()

def get_default_values():
    """Devuelve valores por defecto para los inputs"""
    return {
        "MntWines": 305.03,
        "Spent": 606.71,
        "Income": 51954.46,
        "NumCatalogPurchases": 2,
        "MntMeatProducts": 65.5,
        "NumWebPurchases": 5,
        "Kidhome": 0,
        "Child_Home": 0
    }

def user_input_features(defaults):
    """Crea los controles para entrada de usuario"""
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            MntWines = st.number_input("🍷 Importe gastado en vino", 
                                     min_value=0.0, max_value=1500.0, 
                                     value=defaults["MntWines"])
            Spent = st.number_input("💳 Importe total gastado", 
                                   min_value=0.0, max_value=3000.0, 
                                   value=defaults["Spent"])
            Income = st.number_input("💰 Ingresos anuales", 
                                   min_value=0.0, max_value=200000.0, 
                                   value=defaults["Income"])
        
        with col2:
            NumCatalogPurchases = st.slider("📖 Compras por catálogo", 
                                          min_value=0, max_value=20, 
                                          value=defaults["NumCatalogPurchases"])
            MntMeatProducts = st.number_input("🥩 Gastado en carne", 
                                            min_value=0.0, max_value=500.0, 
                                            value=defaults["MntMeatProducts"])
            NumWebPurchases = st.slider("🛒 Compras online", 
                                      min_value=0, max_value=25, 
                                      value=defaults["NumWebPurchases"])
        
        Kidhome = st.selectbox("👶 Número de hijos", [0, 1, 2], 
                             index=defaults["Kidhome"])
        Child_Home = st.radio("🏡 ¿Hay niños en casa?", 
                            options=[("No", 0), ("Sí", 1)],
                            index=defaults["Child_Home"])[1]
        
        submitted = st.form_submit_button("🔮 Predecir")
        
    return submitted, np.array([[MntWines, Spent, Income, NumCatalogPurchases, 
                               MntMeatProducts, NumWebPurchases, Kidhome, Child_Home]])

def display_results(proba, prediction):
    """Muestra los resultados de la predicción"""
    st.subheader("📢 Resultado de la Predicción")
    st.metric("Probabilidad de aceptar", f"{proba:.2%}")
    
    if prediction == 1:
        st.success("✅ El cliente probablemente ACEPTARÁ una campaña de marketing.")
    else:
        st.error("❌ El cliente probablemente NO aceptará ninguna campaña de marketing.")

def main():
    # Configuración inicial
    st.title("📊 Predicción de Aceptación de Campañas de Marketing")
    st.write("Ingrese los valores para predecir si aceptará al menos una campaña.")

    # Depuración
    st.write(f"Directorio actual: {os.getcwd()}")
    st.write(f"Ruta esperada del modelo: {MODEL_PATH}")
    
    # Cargar el modelo
    MODEL_PATH = Path(__file__).parent.parent / "models" / "classification" / "RandomForestClassifier.pkl"
    model, threshold = load_model(MODEL_PATH)
    
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