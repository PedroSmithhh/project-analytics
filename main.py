import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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

# Filtrando apenas as colunas numéricas
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Header para o heatmap
st.header("Mapa de Correlações")

# Descrição
st.write("""
O *heatmap* abaixo mostra a relação entre as variáveis numéricas do dataset. 
Os valores variam de -1 a 1, onde:
- **1 ou próximo a 1**: indica uma correlação positiva forte.
- **-1 ou próximo a -1**: indica uma correlação negativa forte.
- **0 ou próximo a 0**: indica ausência de correlação significativa.
""")

# Criando a matriz de correlação
corr_matrix = df_numeric.corr()

# Criando o heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)
ax.set_title("Correlação entre Variáveis Numéricas", fontsize=16, pad=15)
st.pyplot(fig)

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

# Conclusões do gráfico de linhas
st.header("Conclusão")
st.write("""
Conclui-se que investir em um ambiente que seja confortavel para grupos de pessoas é uma boa alternativa para o aumento das gorjetas
""")

st.header("Comparação de Gorjetas entre Fumantes e Não Fumantes")

# Cálculo do valor médio de gorjetas para fumantes e não fumantes
tip_means = df.groupby('smoker')['tip'].mean().reset_index()

# Exibir as estatísticas
st.header("Estatísticas Descritivas")
st.write(tip_means)

# Gráfico comparativo
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=tip_means, x='smoker', y='tip', palette="viridis", ax=ax)
ax.set_title("Média de Gorjetas por Grupo (Fumantes vs. Não Fumantes)")
ax.set_xlabel("Fumante")
ax.set_ylabel("Média de Gorjetas (USD)")
st.pyplot(fig)

st.header("Insights")
st.write("""
 - Analisar os valores médios de gorjeta (tip) entre fumantes e não fumantes.
""")

# Conclusões dos fumantes
st.header("Conclusão")
st.write("""
Conclui-se que o hábito de fumar dos clientes não interfere no retorno financeiro do estabelecimento, logo não é necessario
investir em áreas para fumantes por exemplo
""")


# Treinamento do modelo
# Variável alvo
y = df["tip"]

# Variável preditora
X = df[["total_bill"]]  # Transformar em DataFrame para compatibilidade com scikit-learn

# Dividimos os dados para treinar o modelo e avaliá-lo posteriormente.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação e treinamento de um modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Usando o modelo para fazer as predições
y_pred = model.predict(X_test)

# Métricas para avaliar o desempenho do modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Exibir métricas no Streamlit
st.header("Modelo de predição")
st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")

# Comparação de valores reais e previstos
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(X_test, y_test, color="blue", label="Valores Reais", alpha=0.6)
ax.scatter(X_test, y_pred, color="red", label="Valores Previstos", alpha=0.6)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.set_title("Regressão Linear: Total Bill vs Tip")
ax.legend()
st.pyplot(fig)

# Permitir que o usuário insira valores de conta e prever o valor da gorjeta
st.sidebar.header("Predição de Gorjeta")
total_bill_input = st.sidebar.number_input("Insira o valor da conta:", min_value=0.0, step=1.0)

# Predizer gorjeta para o valor de entrada
if total_bill_input > 0:
    predicted_tip = model.predict(pd.DataFrame([[total_bill_input]], columns=["total_bill"]))[0]
    st.sidebar.write(f"Gorjeta Prevista: ${predicted_tip:.2f}")

# Conclusões do gráfico de linhas
st.header("Conclusão")
st.write("""
Conclui-se que há um aumento significativo no valor da gorjeta quando o valor da conta é maior
""")

st.header("Gráfico para relações categóricas")
st.write("Relações categóricas representam a associação entre variáveis qualitativas")

# Validação para tipos de dados
if df[x_var].dtype == "category" or df[x_var].dtype == "object":
    # Gráfico categórico
    st.header(f"Relação entre {x_var} e {y_var}")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x=x_var, y=y_var, ax=ax, palette="Set2")

    # Customização do gráfico
    ax.set_title(f"Distribuição de {y_var} por {x_var}", fontsize=16)
    ax.set_xlabel(x_var.capitalize(), fontsize=14)
    ax.set_ylabel(y_var.capitalize(), fontsize=14)

    # Exibir no Streamlit
    st.pyplot(fig)
else:
    st.warning("Escolha uma variável categórica para o eixo X.")

st.header("Insights")
st.write("""
 - Analisar as relação entre os sexos o valor das contas (x=sex, y=total_bill)
 - Analisar as relações entre os sexos e o valor das gorjetas (x=sex, y=tips)
""")

st.header("Conclusão")
st.write("""
Conclui-se que o sexo não interfere muito no valor da conta e nem no valor da gorjeta.
""")

st.header("Conclusão final")
st.write("""
A análise do dataset Tips revelou que o valor da conta (total_bill) é o principal fator que influencia o valor das gorjetas. 
         Observamos que homens e não fumantes, em média, deixam gorjetas maiores. 
         Além disso, jantares nos finais de semana tendem a gerar receitas mais altas em gorjetas, enquanto grupos maiores 
         apresentam um leve aumento proporcional. Esses insights podem ser aplicados na otimização de horários de atendimento, 
         no treinamento de equipes para lidar com grupos e na criação de estratégias para maximizar receitas.
""")