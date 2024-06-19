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

# Parámetros de la distribución binomial
n = 10  # Número de ensayos
p = 0.5  # Probabilidad de éxito en cada ensayo

# Valores posibles de la variable aleatoria binomial
x = np.arange(0, n + 1)

# Función de masa de probabilidad (PMF)
pmf = binom.pmf(x, n, p)

# Crear la gráfica
plt.bar(x, pmf, color='blue', alpha=0.7)

# Añadir títulos y etiquetas
plt.title('Distribución Binomial')
plt.xlabel('Número de éxitos')
plt.ylabel('Probabilidad')

# Mostrar la gráfica
plt.grid(True)
plt.show()

