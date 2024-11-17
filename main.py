import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Carregando o dataset
df = sns.load_dataset('tips')

# Título do app
st.title("Análise de Gorjetas com Streamlit")

# Visão Geral
st.header("Visão Geral do Dataset")
st.write(df.describe())

# Seleção de variáveis
st.sidebar.header("Configurações")
x_var = st.sidebar.selectbox("Selecione a variável X:", df.columns)
y_var = st.sidebar.selectbox("Selecione a variável Y:", df.columns)

# Gráfico de dispersão
st.header("Gráfico de Dispersão")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_var, y=y_var, hue='sex', ax=ax)
st.pyplot(fig)

# Insights
st.header("Insights")
st.write("""
- Analisar os dias com maior volume de gorjetas.
- Investigar se grupos maiores tendem a deixar gorjetas proporcionalmente menores.
""")
