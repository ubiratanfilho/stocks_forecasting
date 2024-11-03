import streamlit as st
import yfinance as yf
import investpy
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.utils.search import search_ticker_info

### Variáveis Globais
OPEN_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

st.set_page_config(layout="wide")

### Métodos
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start='2000-01-01')
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    cols_rename = {
        'Date': 'date',
        'Ticker': 'ticker',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }

    # Renomear colunas
    data.rename(columns=cols_rename, inplace=True)

    # Convertendo a data para datetime
    data['date'] = pd.to_datetime(data['date'])

    # Set date as index
    data.set_index('date', inplace=True)

    # Colunas adicionais para análise
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    
    return data

def plot_by_year(data):
    last_monthly_data = data.groupby(['year', 'month']).last().reset_index()
    
    # Crie uma figura plotly
    fig = go.Figure()

    # Obter anos únicos para iteração
    years = data['year'].unique()

    # Iterar por cada ano e adicionar como linha no gráfico
    for year in years:
        year_data = last_monthly_data[last_monthly_data['year'] == year]
        fig.add_trace(go.Scatter(
            x=year_data['month'], 
            y=year_data['adj_close'], 
            mode='lines+text',
            name=str(year),
            text=[None]*(len(year_data)-1) + [str(year)],  # Adiciona o rótulo apenas no último ponto
            textposition='top right',
            line=dict(width=2)
        ))

    # Personalizar o layout
    fig.update_layout(
        xaxis_title='Mês',
        yaxis_title='Preço',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
        showlegend=True
    )
    
    return fig

### App
st.title('Previsão de Preços de Ações')

### Input do usuário
col1, _ = st.columns([1, 3])
with col1:
    df_ativos = investpy.stocks.get_stocks(country='brazil')
    tickers = df_ativos['symbol'].tolist()
    tickers = [ticker.strip() + '.SA' for ticker in tickers]
    tickers.insert(0, 'Selecione um ticker')
    ticker = st.selectbox('Selecione o Ticker da Ação', tickers)

col1, col2, col3 = st.columns(3)
if ticker != 'Selecione um ticker':
        ### Carregando dados
        data = load_data(ticker)
        data = preprocess_data(data)

        col1, col2, col3 = st.columns(3)
        ### Análise Fundamentalista
        # with col1:
        #     st.markdown('## Análise Fundamentalista')
        #     st.markdown(search_ticker_info(ticker))

        ### Análise Técnica
        with col2:
            st.markdown(f'## Análise Técnica')

            st.markdown(f'### Preço de Fechamento da Ação')
            fig = px.line(data, x=data.index, y='adj_close', labels={'adj_close': 'Preço', 'date': 'Data'})
            st.plotly_chart(fig)

            st.markdown('### Preço por Ano')
            st.plotly_chart(plot_by_year(data))

        ### Previsão
        with col3:
            st.markdown('## Previsão')

            periods = st.number_input('Entre o número de dias para previsão', min_value=1, max_value=1000, value=30)