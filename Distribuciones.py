# Importar Streamlit
import streamlit as st

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm

# Título de la aplicación
st.title('Distribuciones de Probabilidad')
st.markdown("#### Dra. MariCarmen González-Videgaray")
st.markdown("#### Mtro. Rubén Romero-Ruiz")
st.markdown("#### Act. Luz María Lavín-Alanís")

# Barra lateral
with st.sidebar:
    st.write("En este dashboard se muestran las distribuciones de probabilidad Binomial, Poisson y Normal.")

# Función para crear la gráfica de la binomial
def plot_binomial_distribution(n, p):
    # Valores posibles de la variable aleatoria binomial
    x = np.arange(0, n + 1, 1)

    # Función de masa de probabilidad (PMF)
    pmf = binom.pmf(x, n, p)

    # Crear la gráfica
    fig, ax = plt.subplots()
    ax.bar(x, pmf, color='blue', alpha=0.7)

    # Añadir títulos y etiquetas
    ax.set_title('Distribución Binomial')
    ax.set_xlabel('Número de éxitos')
    ax.set_ylabel('Probabilidad')
    ax.grid(True)

    return fig

# Streamlit app
st.title("Visualización de la Distribución Binomial")

# Input parameters
n = st.slider('Número de ensayos (n)', 1, 100, 10)
p = st.slider('Probabilidad de éxito (p)', 0.0, 0.5, 1.0)

# Generate plot
fig = plot_binomial_distribution(n, p)

# Display plot in Streamlit
st.matplotlib.pyplot(fig)

