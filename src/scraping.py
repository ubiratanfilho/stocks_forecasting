import yfinance as yf
import pandas as pd
import investpy
from datetime import datetime
import time

# Obter a lista de tickers da B3 usando investpy
df_ativos = investpy.stocks.get_stocks(country='brazil')
tickers = df_ativos['symbol'].tolist()
tickers = [ticker.strip() + '.SA' for ticker in tickers]

print(f"Número de tickers: {len(tickers)}")

# Definir as datas de início e fim
data_inicio = '2000-01-01'
data_fim = datetime.today().strftime('%Y-%m-%d')

# Inicializar um DataFrame vazio para armazenar os dados
dados_totais = pd.DataFrame()

# Baixar os dados em lotes para evitar limites de taxa
tamanho_lote = 50  # Ajuste o tamanho do lote conforme necessário
for i in range(0, len(tickers), tamanho_lote):
    lote_tickers = tickers[i:i+tamanho_lote]
    print(f"Baixando dados para tickers {i} até {i+len(lote_tickers)-1}")
    try:
        dados = yf.download(lote_tickers, start=data_inicio, end=data_fim, group_by='ticker', threads=True)
        if dados.empty:
            continue
        # Converter os dados para formato longo
        dados = dados.stack(level=0).reset_index()
        dados.rename(columns={'level_1': 'Ticker'}, inplace=True)
        dados_totais = pd.concat([dados_totais, dados], ignore_index=True)
        time.sleep(10)  # Pausa para evitar limites de taxa
    except Exception as e:
        print(f"Erro ao baixar dados para o lote {i} até {i+len(lote_tickers)-1}: {e}")
        time.sleep(60)  # Esperar mais tempo em caso de erro

# Salvar o dataset em um arquivo CSV
dados_totais.to_csv('historico_acoes_b3.csv', index=False)
print("Download concluído e dados salvos em 'historico_acoes_b3.csv'.") 