# Importar Streamlit
import streamlit as st

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm

# Título de la aplicación
st.title('Distribuciones de Probabilidad')

# Barra lateral
with st.sidebar:
    st.image("UNAM.png")
    st.markdown("# Distribuciones de Probabilidad")
    st.markdown("#### Dra. MariCarmen González-Videgaray")
    st.markdown("#### Mtro. Rubén Romero-Ruiz")
    st.markdown("#### Act. Luz María Lavín-Alanís")
    st.write("En este dashboard se muestran las distribuciones de probabilidad:")
    st.write("Binomial:")
    st.latex(r'''
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
'''
)
    st.write("Poisson:")
    st.latex(r'''
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
''')

    st.write("Normal:")
    st.latex(r'''
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
''')

# Función para crear la gráfica de la binomial
def plot_binomial_distribution(n, p):
    # Valores posibles de la variable aleatoria binomial
    x = np.arange(0, n + 1, 1)

    # Función de masa de probabilidad (PMF)
    pmf = binom.pmf(x, n, p)

    # Crear la gráfica
    fig1, ax = plt.subplots()
    ax.bar(x, pmf, color='blue', alpha=0.7)

    # Fijar los límites de los ejes
    ax.set_xlim(0, 100)  # Límite del eje X
    ax.set_ylim(0, 0.35) # Límite del eje Y

    # Añadir títulos y etiquetas
    ax.set_title('Distribución Binomial')
    ax.set_xlabel('Número de éxitos')
    ax.set_ylabel('Probabilidad')
    ax.grid(True)

    return fig1

# Streamlit app
st.markdown("## Visualización de la Distribución Binomial")

# Input parameters
n = st.slider('Número de ensayos (n)', 1, 100, 10)
p = st.slider('Probabilidad de éxito (p)', 0.0, 1.0, 0.5)

# Generate plot
fig1 = plot_binomial_distribution(n, p)

# Display plot in Streamlit
st.pyplot(fig1)

# Función para crear la gráfica de la distribución de Poisson
def plot_poisson_distribution(mu):
    # Valores posibles de la variable aleatoria Poisson
    x = np.arange(0, mu * 3 + 1)

    # Función de masa de probabilidad (PMF)
    pmf = poisson.pmf(x, mu)

    # Crear la gráfica
    fig2, ax = plt.subplots()
    ax.bar(x, pmf, color='blue', alpha=0.7)

    # Añadir títulos y etiquetas
    ax.set_title('Distribución de Poisson')
    ax.set_xlabel('Número de eventos')
    ax.set_ylabel('Probabilidad')
    ax.grid(True)

    return fig2

# Aplicación de Streamlit
st.markdown("## Visualización de la Distribución de Poisson")

# Parámetro de entrada
mu = st.slider('Tasa promedio de eventos (lambda)', 0.1, 20.0, 3.0)

# Generar gráfica
fig2 = plot_poisson_distribution(mu)

# Mostrar gráfica en Streamlit
st.pyplot(fig2)

# Función para crear la gráfica de la distribución normal
def plot_normal_distribution(mean, std_dev):
    # Rango de valores para la variable aleatoria normal
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)

    # Función de densidad de probabilidad (PDF)
    pdf = norm.pdf(x, mean, std_dev)

    # Crear la gráfica
    fig3, ax = plt.subplots()
    ax.plot(x, pdf, color='blue')

    # Fijar los límites de los ejes
    ax.set_xlim(-14, 14)  # Límite del eje X
    ax.set_ylim(0, 1) # Límite del eje Y

    # Añadir títulos y etiquetas
    ax.set_title('Distribución Normal')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad de probabilidad')
    ax.grid(True)

    return fig3

# Aplicación de Streamlit
st.markdown("## Visualización de la Distribución Normal")

# Parámetros de entrada
mean = st.slider('Media (mu)', -10.0, 10.0, 0.0)
std_dev = st.slider('Desviación estándar (sigma)', 0.1, 5.0, 1.0)

# Generar gráfica
fig3 = plot_normal_distribution(mean, std_dev)

# Mostrar gráfica en Streamlit
st.pyplot(fig3)
