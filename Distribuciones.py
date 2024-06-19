# Importar Streamlit
import streamlit as st

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Título de la aplicación
st.title('Distribuciones de Probabilidad')

# Barra lateral
with st.sidebar:
    st.markdown("# Distribuciones de Probabilidad")
    st.markdown("#### Dra. MariCarmen González-Videgaray")
    st.markdown("#### Mtro. Rubén Romero-Ruiz")
    st.markdown("#### Act. Luz María Lavín-Alanís")
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
p = st.slider('Probabilidad de éxito (p)', 0.0, 1.0, 0.5)

# Generate plot
fig = plot_binomial_distribution(n, p)

# Display plot in Streamlit
st.pyplot(fig)

# Función para crear la gráfica de la distribución de Poisson
def plot_poisson_distribution(mu):
    # Valores posibles de la variable aleatoria Poisson
    x = np.arange(0, mu * 3 + 1)

    # Función de masa de probabilidad (PMF)
    pmf = poisson.pmf(x, mu)

    # Crear la gráfica
    fig, ax = plt.subplots()
    ax.bar(x, pmf, color='blue', alpha=0.7)

    # Añadir títulos y etiquetas
    ax.set_title('Distribución de Poisson')
    ax.set_xlabel('Número de eventos')
    ax.set_ylabel('Probabilidad')
    ax.grid(True)

    return fig

# Aplicación de Streamlit
st.title("Visualización de la Distribución de Poisson")

# Parámetro de entrada
mu = st.slider('Tasa promedio de eventos (mu)', 0.1, 20.0, 3.0)

# Generar gráfica
fig = plot_poisson_distribution(mu)

# Mostrar gráfica en Streamlit
st.pyplot(fig)
