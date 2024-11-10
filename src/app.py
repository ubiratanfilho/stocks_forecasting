import streamlit as st
import yfinance as yf
import investpy
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils.search import search_ticker_info
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction

### Variáveis Globais
# OPEN_API_KEY = st.secrets["OPENAI_API_KEY"]
# SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
OPEN_API_KEY = None
SERPAPI_API_KEY = None

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
    
    data = data.asfreq('B')  # 'B' para dias úteis
    data['adj_close'] = data['adj_close'].fillna(method='ffill')

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

def decomposition(data):
    result = seasonal_decompose(data['adj_close'], model='multiplicative', period=365)
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=['Preço de Fechamento', 'Tendência', 'Sazonalidade', 'Resíduo'])

    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)

    fig.update_layout(height=800, title_text='Decomposition of Time Series', showlegend=False)
    
    return fig

def auto_arima(data, periods):
    model = AutoARIMA(n_fits=10)
    fh = ForecastingHorizon(np.arange(1, periods + 1), is_relative=True)
    
    model.fit(data)
    y_pred = model.predict(fh)
    
    return y_pred

def prophet(data, periods):
    model = Prophet()
    fh = ForecastingHorizon(np.arange(1, periods + 1), is_relative=True)
    
    model.fit(data)
    y_pred = model.predict(fh)
    
    return y_pred

def forecast(data, model_name='auto_arima', evaluate=False, periods=None):
    models = {
        'auto_arima': AutoARIMA(n_fits=10),
        'prophet': Prophet(daily_seasonality=True),
        'xgboost': make_reduction(XGBRegressor(n_estimators=500), window_length=15, strategy="recursive")
    }
    
    model = models[model_name]
    
    if evaluate:
        y_train, y_test = temporal_train_test_split(data, test_size=0.2)
        fh = ForecastingHorizon(np.arange(1, len(y_test) + 1), is_relative=True)
    
        model.fit(y_train)
        y_pred = model.predict(fh)
        
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return mape
    else:
        fh = ForecastingHorizon(np.arange(1, periods + 1), is_relative=True)
        
        model.fit(data)
        y_pred = model.predict(fh)
        
        return y_pred

def plot_forecast(data, y_pred):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Observado'))
    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', name='Previsão'))
    
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço')
    
    return fig

### App
st.title('Previsão de Preços de Ações')
st.markdown('Este aplicativo tem como objetivo realizar a previsão de preços de ações utilizando diferentes modelos de Machine Learning, além de utilizar um agente de IA Generativa para realizar pesquisas sobre a ação selecionada.')

### Input do usuário
col1, _ = st.columns([1, 3])
with col1:
    st.info('Para utilizar o aplicativo, selecione um ativo e coloque sua API da OpenAI e SerpAPI.')
    st.text_input("OpenAI API", type="password", value=OPEN_API_KEY)
    st.text_input("SerpAPI API", type="password", value=SERPAPI_API_KEY)
    
    df_ativos = investpy.stocks.get_stocks(country='brazil')
    tickers = df_ativos['symbol'].tolist()
    tickers = [ticker.strip() + '.SA' for ticker in tickers]
    tickers.insert(0, 'Selecione um ticker')
    ticker = st.selectbox('Selecione o Ticker da Ação', tickers)

with st.spinner("Carregando pesquisas e previsões..."):
    ### Análise Fundamentalista
    if ticker != 'Selecione um ticker' and not OPEN_API_KEY and not SERPAPI_API_KEY:
        st.markdown('## Análise Fundamentalista')
        st.markdown(search_ticker_info(ticker))
        pass

    col1, col2, col3 = st.columns(3)
    if ticker != 'Selecione um ticker':
            ### Carregando dados
            data = load_data(ticker)
            data = preprocess_data(data)            

            col1, col2= st.columns(2)
            ### Análise Técnica
            with col1:
                st.markdown(f'## Análise Técnica')

                st.markdown('### Preço de Fechamento da Ação')
                st.plotly_chart(decomposition(data))

                st.markdown('### Preço por Ano')
                st.plotly_chart(plot_by_year(data))

            ### Previsão
            with col2:
                st.markdown('## Previsão')

                periods = st.number_input('Entre o número de dias para previsão', min_value=1, max_value=1000, value=30)
                
                st.markdown('### XGBoost')
                mape = forecast(data['adj_close'], model_name='xgboost', evaluate=True)
                st.write(f'MAPE: {mape:.2f}%')
                y_pred_xgboost = forecast(data['adj_close'][-365:], model_name='xgboost', periods=periods)
                st.plotly_chart(plot_forecast(data['adj_close'][-365:], y_pred_xgboost))
                
                st.markdown('### Auto-ARIMA')
                mape = forecast(data['adj_close'], model_name='auto_arima', evaluate=True)
                st.write(f'MAPE: {mape:.2f}%')
                y_pred_arima = forecast(data['adj_close'][-365:], model_name='auto_arima', periods=periods)
                st.plotly_chart(plot_forecast(data['adj_close'][-365:], y_pred_arima))
                
                st.markdown('### Prophet')
                mape = forecast(data['adj_close'], model_name='prophet', evaluate=True)
                st.write(f'MAPE: {mape:.2f}%')
                y_pred_prophet = forecast(data['adj_close'][-365:], model_name='prophet', periods=periods)
                st.plotly_chart(plot_forecast(data['adj_close'][-365:], y_pred_prophet))