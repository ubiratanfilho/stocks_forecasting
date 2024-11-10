# Previsão de Ações com Séries Temporais e IA Generativa

## Descrição do Projeto
Aplicação Streamlit que realiza a previsão de ações de empresas listadas na B3, além de realizar uma análise fundamentalista da empresa com um agente de IA Generativa.

## Como executar o projeto
1. Clone o repositório
```
git clone https://github.com/ubiratanfilho/stocks_forecasting
```

2. Rode a imagem docker
```
docker pull ubiratanfilho/stocks-forecasting:latest
docker run -p 8502:8502 ubiratanfilho/stocks-forecasting:latest
```

3. Acesse a aplicação no navegador
```
http://localhost:8502
```

4. Adicione as credenciais das APIs da OpenAI e SerpAPI:
- Crie uma API da OpenAI: https://platform.openai.com/signup
- Crie uma API da SerpAPI: https://serpapi.com/

5. Selecione o ticker da empresa e aguarda a análise ser gerada.
