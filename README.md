# Project-analytics
Este repositório contém um projeto de análise de dados interativo utilizando Streamlit e o dataset tips do Seaborn. O objetivo do aplicativo é explorar os fatores que influenciam o valor das gorjetas em restaurantes, oferecendo insights úteis para otimizar estratégias de atendimento.

## Estrututura do aplicativo

### Introdução e objetivo
O aplicativo começa com uma introdução que apresenta o objetivo principal:
- Explorar as relações entre o valor da conta (total_bill), gorjeta (tip), sexo dos clientes (sex), dias da semana (day), período (time) e tamanho do grupo (size).

### Leitura e visão geral dos dados
- Exibição inicial do dataset tips.
- Estatísticas descritivas gerais das variáveis.

### Vizualização
O aplicativo apresenta gráficos para explorar e identificar padrões nos dados:
- Heatmap de Correlações: Mostra as relações entre as variáveis numéricas.
- Gráfico de Dispersão: Explora associações entre variáveis, com insights baseados em variáveis como day, tip, sex e time.
- Gráfico de Linha: Avalia tendências do valor da conta e da gorjeta em relação ao tamanho do grupo.
- Gráfico Comparativo (Fumantes x Não Fumantes): Exibe a média de gorjetas para clientes fumantes e não fumantes.
- Gráficos de Relações Categóricas: Explora associações entre variáveis qualitativas e numéricas usando boxplots.

### Modelo de regressão linear
- Treinamento de um modelo para prever o valor da gorjeta com base no valor da conta.
- Métricas de desempenho do modelo:
- Erro Médio Absoluto (MAE)
- Coeficiente de Determinação (R²)
- Comparação gráfica entre valores reais e previstos.
- Ferramenta interativa para prever gorjetas com base no valor da conta.

## Como executar o aplicativo
1. Clone este repositório
```bash
git clone https://github.com/seu-repositorio.git
```
2. Instale as dependências
```bash
pip install -r requirements.txt
```
3. Execute o aplicativo
```bash
streamlit run app.py
```
4. O aplicativo também pode ser encontrado nesse link: https://project-analytics-trabalhofinal.streamlit.app

## Tecnologias utilizadas
- Python: Linguagem de programação principal.
- Streamlit: Framework para desenvolvimento de aplicativos interativos.
- Seaborn: Biblioteca para visualização de dados.
- Matplotlib: Suporte para visualizações gráficas.
- Scikit-learn: Modelagem e métricas de aprendizado de máquina.
