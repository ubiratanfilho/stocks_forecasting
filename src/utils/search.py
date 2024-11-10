import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper

# Inicializar o modelo de linguagem da OpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini")

# Configurar a ferramenta de busca com SerpAPI
search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Realiza buscas na internet."
)

# Procura o nome da empresa de acordo com o ticker
def buscar_nome_empresa(ticker):
    pergunta = f"Qual é o nome da empresa que corresponde ao ticker {ticker}? Retorne apenas o nome da empresa."
    nome_empresa = agent.run(pergunta)
    return nome_empresa

# Inicializar o agente com a ferramenta de busca
tools = [search_tool]
agent = initialize_agent(tools, llm, agent="openai-functions", verbose=True)

# Função para buscar informações sobre uma empresa
def buscar_informacoes_empresa(nome_empresa):
    pergunta = f"Faça um resumo da história da empresa {nome_empresa}. Em seguida, realize uma análise fundamentalista sobre a empresa para embasar a decisão de investimento. A resposta deve conter cerca de 500 palavras."
    resposta = agent.run(pergunta)
    return resposta

# Exemplo de uso
def search_ticker_info(ticker):
    nome_empresa = buscar_nome_empresa(ticker)

    informacoes = buscar_informacoes_empresa(nome_empresa)
    
    return informacoes
