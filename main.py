# SINAIS DE ENTRADA [HS, AS, HST, AST, HC, AC, HR, AR]

# FTR [H, A, D] ---> [001, 010, 011]  --> NORMALIZAR


# class HS:
#     def __init__(self):
#         self.info = 17
#         self.prox = None

# class AS:
#     def __init__(self):
#         self.info = 8
#         self.prox = None

# class HST:
#     def __init__(self):
#         self.info = 14
#         self.prox = None

# class AST:
#     def __init__(self):
#         self.info = 4
#         self.prox = None


# class HC:
#     def __init__(self):
#         self.info = 6
#         self.prox = None

# class AC:
#     def __init__(self):
#         self.info = 6
#         self.prox = None

# class HR:
#     def __init__(self):
#         self.info = 0
#         self.prox = None

# class AR:
#     def __init__(self):
#         self.info = 0
#         self.prox = None

# class FTR:
#     def __init__(self):
#         self.info = 'H001'
#         self.prox = None

# class Header:
#     def __init__(self):
#         self.info = None  # Estrutura - inicio da lista com os emissores

# neuronios = [HS, AS, HST, AST, HC, AC, HR, AR]
# neuronios = [17,8, 14, 4, 6, 6, 0, 0]
from loguru import logger
import math
import numpy as np
import pandas as pd

nome_arquivo = './dataset_futebol.csv'

def carregar_dados(nome_arquivo):
    # Leitura do arquivo CSV
    dados_csv_original = pd.read_csv(nome_arquivo, delimiter=';')
    dados_csv_copia = pd.read_csv(nome_arquivo, delimiter=';')
                    
    colunas_desejadas = ['FTR','HS','AS','HST','AST','HC','AC','HR','AR']
    dados_limpos = dados_csv_copia[colunas_desejadas]
    
    dados_limpos_entradas = dados_limpos.drop(columns=['FTR'])
    dados_limpos_saidas = dados_limpos.drop(columns=['HS','AS','HST','AST','HC','AC','HR','AR'])

    valores_entrada = dados_limpos_entradas.values
    valores_saida = dados_limpos_saidas.values

    # entrada_dimensao = valores_entrada.shape[1]

    return valores_entrada[3], valores_saida


class Neuronio:
    def __init__(self, peso):
        self.peso = peso
        self.valor = None

class Camada:
    def __init__(self, num_neuronios):
        self.neuronios = [Neuronio(0) for _ in range(num_neuronios)]

class RedeNeural:
    def __init__(self):
        NEURONIOS_CAMADA_ENTRADA = 8
        NEURONIOS_CAMADA_OCULTA = 6
        NEURONIOS_CAMADA_SAIDA = 3
        QTD_AMOSTRAS = 4
        QTD_DADOS = 8  # Número de neurônios na camada de entrada
        TAM_PESOS_OCULTA = QTD_DADOS + 1  # Mais um do bias
        TAM_PESOS_SAIDA = NEURONIOS_CAMADA_OCULTA + 1  # Mais um do bias
        logger.info(f'NEURONIOS_CAMADA_ENTRADA {NEURONIOS_CAMADA_ENTRADA}')
        logger.info(f'NEURONIOS_CAMADA_OCULTA {NEURONIOS_CAMADA_OCULTA}')
        logger.info(f'NEURONIOS_CAMADA_SAIDA {NEURONIOS_CAMADA_SAIDA}')
        logger.info(f'QTD_AMOSTRAS {QTD_AMOSTRAS}')
        logger.info(f'QTD_DADOS {QTD_DADOS}')
        logger.info(f'TAM_PESOS_OCULTA {TAM_PESOS_OCULTA}')
        logger.info(f'TAM_PESOS_SAIDA {TAM_PESOS_SAIDA}')
        
        # Camada de entrada com 8 neurônios
        self.camada_entrada = Camada(NEURONIOS_CAMADA_ENTRADA)
        logger.info(f'camada_entrada {self.camada_entrada}')
        
        # Camada oculta com 6 neurônios
        self.camada_oculta = Camada(NEURONIOS_CAMADA_OCULTA)
        logger.info(f'camada_oculta {self.camada_oculta}')

        # Camada de saída com 3 neurônios (H, D, A)
        self.camada_saida = Camada(NEURONIOS_CAMADA_SAIDA)
        logger.info(f'camada_saida {self.camada_saida}')
        self.inicializar_pesos_aleatoriamente()  # Função para inicializar os pesos aleatoriamente

    def inicializar_pesos_aleatoriamente(self):
        for camada in [self.camada_entrada, self.camada_oculta, self.camada_saida]:
            for neuronio in camada.neuronios:
                neuronio.peso = np.random.uniform(-0.1, 0.1)
                logger.info(f'Peso inicial do neurônio: {neuronio.peso}')

    def feedforward(self, entradas):
        logger.info(f'len(entradas) {len(entradas)}')
        logger.info(f'len(self.camada_entrada.neuronios) {len(self.camada_entrada.neuronios)}')
        if len(entradas) != len(self.camada_entrada.neuronios):
            raise ValueError("Número incorreto de entradas")

        # Configurar os valores nas camadas de entrada
        for i in range(len(entradas)):
            self.camada_entrada.neuronios[i].valor = entradas[i]
            logger.info(f' self.camada_entrada.neuronios[ i = {i} ].valor {self.camada_entrada.neuronios[i].valor}')

        # Calcular os valores na camada oculta
        for neuronio in self.camada_oculta.neuronios:
            soma_ponderada = sum(entrada.valor * neuronio.peso for entrada in self.camada_entrada.neuronios)
            logger.info(f' soma_ponderada {soma_ponderada}')
            logger.info(f' neuronio.peso {neuronio.peso}')
            
            # Aplicar função sigmoidal como função de ativação
            neuronio.valor = self.funcao_sigmoidal(soma_ponderada)
            logger.info(f'Valor do neurônio oculto: {neuronio.valor}')

        # Calcular os valores na camada de saída
        for neuronio in self.camada_saida.neuronios:
            soma_ponderada = sum(neuronio_oculto.valor * neuronio.peso for neuronio_oculto in self.camada_oculta.neuronios)
            logger.info(f' soma_ponderada {soma_ponderada}')
            
            # Aplicar função sigmoidal como função de ativação
            neuronio.valor = self.funcao_sigmoidal(soma_ponderada)
            logger.info(f'Valor do neurônio de saída: {neuronio.valor}')

    def calcular_erro_entropia_cruzada(self, saida_desejada):
        if len(saida_desejada) != len(self.camada_saida.neuronios):
            raise ValueError("Número incorreto de valores desejados")

        erro = 0.0
        for i, neuronio_saida in enumerate(self.camada_saida.neuronios):
            valor_desejado = saida_desejada[i]  # Valor desejado
            valor_previsto = neuronio_saida.valor  # Saída prevista
            logger.warning(f'valor_desejado: {valor_desejado}')
            logger.warning(f'valor_previsto: {valor_previsto}')

            erro_termo_1 = -valor_desejado * math.log(valor_previsto)
            erro_termo_2 = -(1 - valor_desejado) * math.log(1 - valor_previsto)

            erro += erro_termo_1 + erro_termo_2

        logger.info(f'Erro da rede neural: {erro}')
        return erro

    def funcao_sigmoidal(self, x):
        # Função sigmoidal
        result_func = 1 / (1 + math.exp(-x))
        logger.info(f'funcao_sigmoidal(soma_ponderada={x}) --> {result_func}')
        return result_func

def inicia_interface():

    entradas_teste, alvos = carregar_dados(nome_arquivo)
    logger.success(f'entradas_teste: {entradas_teste}')
    logger.success(f'alvos: {alvos}')
    # Configurar uma rede neural com 8 entradas, 1 camada oculta com 6 neurônios e 1 saída com 3 neurônios
    

    # Definir entradas (substitua pelos seus dados)
    [HS, AS, HST, AST, HC, AC, HR, AR] = entradas_teste
    logger.info(f'[HS, AS, HST, AST, HC, AC, HR, AR] --> {[HS, AS, HST, AST, HC, AC, HR, AR]}')

    rede_neural = RedeNeural()

    entradas = [HS, AS, HST, AST, HC, AC, HR, AR]


    # Executar o feedforward
    rede_neural.feedforward(entradas)

    # Definir saída desejada (codificação dummy para H, D, A)
    saida_desejada = [1, 0, 0]  # Por exemplo, para H (vitória em casa)

    # Calcular o erro usando entropia cruzada
    erro = rede_neural.calcular_erro_entropia_cruzada(saida_desejada)
    logger.success(f'Erro da rede neural: {erro}')

# Chame a função inicia_interface para iniciar a rede neural
inicia_interface()
