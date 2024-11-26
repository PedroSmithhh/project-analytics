import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Carregando o dataset
df = sns.load_dataset('tips')

# Título do app
st.title("Análise de Gorjetas com Streamlit")

# Objetivo
st.write("""
O objetivo principal desse aplicativo é explorar as relações entre gorjetas (tip), total da conta (total_bill), 
sexo, dia da semana, horário e tamanho do grupo, oferecendo insights para otimizar estratégias 
de atendimento em restaurantes. 
""")

# Dataset
st.header("Leitura inicial dos dados")
st.write(df)

# Visão Geral
st.header("Dados estatísticos gerais do Dataset")
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

# Insights do grafico de dispersão
st.header("Insights")
st.write("""
- Analisar os dias com maior volume de gorjetas (x=day, y=tip)
    - Nessa análise, nota-se que aos finais de semana a quantidade e o valor das gorjetas tendem a ser mais altos, 
    sendo majoritariamente pagas por homens (x=sex, y=tip)
- Analisar os periodos em que existe uma maior taxa de gorjetas (y=tip, x=time)
    - Nessa analise, nota-se que no periodo do jantar existe uma taxa bem maior de gorjetas, sendo normalmanete com um valor mais alto que no almoço
""")

# Conclusões do gráfico de dispersão
st.header("Conclusão")
st.write("""
Conclui-se que seria interessante para o estabelecimento investir na qualidade do atendimento aos finais de semana ou até
estender o horário de atendimento. Alèm disso, seria interessante propor algum incentivo para aumentar a presença de homens no estabelecimento
""")

