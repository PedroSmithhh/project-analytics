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

# Grafico de linhas
st.header("Gráfico de linha")
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=df, x=x_var, y=y_var, ax=ax, marker='o') # Cria bandas de confiança com 95% de conifança
st.pyplot(fig)

# Insights do grafico de dispersão
st.header("Insights")
st.write("""
- Analisar o comportmento das gorjetas com o aumento do numero de pessoas na mesa (x=size,y=tip)
    - Nessa análise, nota-se que há um aumento considerável de gorjeta conforme o número de pessoas na mesa aumenta

""")

# Conclusões do gráfico de dispersão
st.header("Conclusão")
st.write("""
Conclui-se que seria interessante para o estabelecimento investir na qualidade do atendimento aos finais de semana ou até
estender o horário de atendimento. Alèm disso, seria interessante propor algum incentivo para aumentar a presença de homens no estabelecimento
""")

# Gráfico de regressão
st.header(f"Regressão entre {x_var} e {y_var}")
fig, ax = plt.subplots(figsize=(8, 4))

# Plotando a regressão
sns.regplot(data=df, x=x_var, y=y_var, ax=ax, scatter_kws={"s": 50}, line_kws={"color": "red"})

# Título e ajustes
ax.set_title(f"Regressão Linear entre {x_var} e {y_var}", fontsize=16)
ax.set_xlabel(f"{x_var} ($)")
ax.set_ylabel(f"{y_var} ($)")

st.pyplot(fig)

