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
    page_title="Predicción de Gasto de Clientes",
    page_icon="💰",
    layout="wide"
)

# Configuración de paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Subimos 2 niveles desde "app/regression"
MODEL_PATH = BASE_DIR / "models" / "regression" / "RandomForestRegressor.pkl"

@st.cache_resource
def load_model():
    """Carga el modelo con manejo de errores"""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        st.stop()

def get_default_values():
    """Valores por defecto para los inputs"""
    return {
        "Income": 51955.00,
        "Seniority": 0.50,
        "Age": 47.0,
        "Teenhome": 0.0,
        "Kidhome": 0.0,
        "Education": 1
    }

def user_input_features(defaults):
    """Crear los controles para entrada de usuario"""
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            Income = st.number_input("💰 Ingresos anuales del hogar", 
                                   min_value=0.0, max_value=200000.0, 
                                   value=defaults["Income"], step=1000.0)
            Seniority = st.slider("🏆 Antigüedad del cliente (0=nuevo, 1=veterano)", 
                                min_value=0.0, max_value=1.0, 
                                value=defaults["Seniority"], step=0.01)
            Age = st.number_input("👨 Edad del cliente", 
                                min_value=18, max_value=100, 
                                value=int(defaults["Age"]))
        
        with col2:
            Teenhome = st.selectbox("🧑 Número de adolescentes en el hogar", 
                                  [0, 1, 2], index=int(defaults["Teenhome"]))
            Kidhome = st.selectbox("👶 Número de niños en el hogar", 
                                 [0, 1, 2], index=int(defaults["Kidhome"]))
            Education = st.selectbox("🎓 Nivel educativo", 
                                   ["Primaria", "Segundo ciclo", "Licenciatura", "Máster", "Doctorado"],
                                   index=defaults["Education"])
        
        # Mapear educación a valores numéricos
        education_map = {
            "Primaria": 0,
            "Segundo ciclo": 1,
            "Licenciatura": 2,
            "Máster": 3,
            "Doctorado": 4
        }
        Education_encoded = education_map[Education]
        
        # Determinar Child_Home automáticamente
        Child_Home = 1 if (Teenhome > 0 or Kidhome > 0) else 0

        submitted = st.form_submit_button("🔮 Predecir Gasto")
        
    return submitted, np.array([[Income, Seniority, Age, Teenhome, Kidhome, Child_Home, Education_encoded]])

def display_results(prediction):
    """Muestra los resultados de la predicción"""
    st.subheader("📢 Resultado de la Predicción")
    st.metric("Gasto total estimado (últimos 2 años)", f"${prediction[0]:,.2f}")
    
    # Interpretación del resultado
    if prediction[0] > 1000:
        st.success("💰 Cliente de alto valor - Recomendado para campañas premium")
    elif prediction[0] > 500:
        st.info("💳 Cliente de valor medio - Buen candidato para promociones")
    else:
        st.warning("🛒 Cliente de bajo gasto - Considerar estrategias de retención")

def main():
    st.title("💰 Predicción de Gasto de Clientes")
    st.header("Predice el gasto total de un cliente en los próximos 2 años usando IA")
    st.write("Ingrese las características del cliente para predecir su gasto potencial.")
    
    # Cargar el modelo
    if not MODEL_PATH.exists():
        st.error(f"❌ Error: El modelo no se encuentra en: {MODEL_PATH}")
        st.stop()
    
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Instrucciones")
        st.markdown("""
        1. Introduzca los valores del cliente  
        2. Haga clic en **Predecir Gasto**  
        3. Verá la estimación de gasto y recomendaciones  
        """)
        st.header("📊 Estadísticas de Referencia")
        st.markdown("""
        - **Gasto promedio**: $607  
        - **Gasto mediano**: $396  
        - **Máximo gasto**: $2,525  
        """)
        # Enlace a la app de Classification
        st.sidebar.markdown("---")  # Separador
        st.sidebar.markdown("### Navegar a otras apps")
        st.sidebar.markdown(
            """
            <a href="https://marketingmlclassificationregressionclustering-2apkcvnbir7q4iuc.streamlit.app" target="_blank">
                <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                    Predicción de Aceptación de Campañas de Marketing
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
    
    # Entrada de usuario
    defaults = get_default_values()
    submitted, user_data = user_input_features(defaults)
    
    # Predicción
    if submitted:
        try:
            prediction = model.predict(user_data)
            display_results(prediction)
            
            # Sección adicional de análisis
            with st.expander("📈 Análisis Detallado"):
                st.write("**Interpretación de variables:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ingresos anuales", f"${user_data[0][0]:,.2f}")
                    st.metric("Antigüedad", f"{user_data[0][1]*100:.0f}%")
                with col2:
                    st.metric("Edad", f"{user_data[0][2]:.0f} años")
                    st.metric("Hijos en casa", "Sí" if user_data[0][5] else "No")
                
                st.write("Este modelo considera que los principales factores personales que influyen en el gasto en el negocio y son:")
                st.markdown("""
                - Ingresos anuales del hogar  
                - Antigüedad como cliente  
                - Presencia de hijos en el hogar  
                - Nivel educativo  
                """)
                
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")

if __name__ == "__main__":
    main()