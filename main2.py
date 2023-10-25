import numpy as np
import pandas as pd
from loguru import logger

### python main2.py >> result.txt 2>&1

# Constantes
NEURONIOS_CAMADA_ENTRADA = 8
NEURONIOS_CAMADA_INTERMEDIARIA = 6
NEURONIOS_CAMADA_SAIDA = 3
QTD_DADOS = 8  # Número de neuronios na camada de entrada
## bias não e imprescindível mas pode auxiliar no deslocamento do eixo x p/ dados mais complexos, então foi mantido
NEURONIOS_CAMADA_INTERMEDIARIA = QTD_DADOS + 1  # Mais um do bias
TAM_PESOS_SAIDA = NEURONIOS_CAMADA_INTERMEDIARIA + 1  # Mais um do bias
NOME_ARQUIVO = "./dataset_futebol2.csv"


def carregar_dados(NOME_ARQUIVO):
    """
    Carrega dados de um arquivo CSV, processa e retorna os vetores de entrada e os valores desejados mapeados.

    Args:
        NOME_ARQUIVO (str): O caminho para o arquivo CSV.

    Returns:
        tuple: Uma tupla contendo o vetor de entrada e os valores desejados mapeados.
    """
    # Leitura do arquivo CSV
    dados_csv = pd.read_csv(NOME_ARQUIVO, delimiter=";")
    # Selecionar as colunas relevantes para as amostras de entrada (vetor_entrada) e os resultados desejados (SAIDA_ESPERADA)
    entrada_columns = ["HS", "AS", "HST", "AST", "HC", "AC", "HR", "AR"]
    SAIDA_ESPERADA_column = "FTR"
    # Extrair os dados de entrada (vetor_entrada) e resultados desejados (SAIDA_ESPERADA)
    vetor_entrada = dados_csv[entrada_columns].values
    SAIDA_ESPERADA = dados_csv[SAIDA_ESPERADA_column].values
    # Mapear os resultados (H, SAIDA_ESPERADA, A) para vetores (1, 0, 0), (0, 1, 0) e (0, 0, 1)
    # SAIDA_ESPERADA_MAP = []
    # for resultado in SAIDA_ESPERADA:
    #     if resultado == "H":
    #         SAIDA_ESPERADA_MAP.append([1, 0, 0])
    #     elif resultado == "D":
    #         SAIDA_ESPERADA_MAP.append([0, 1, 0])
    #     elif resultado == "A":
    #         SAIDA_ESPERADA_MAP.append([0, 0, 1])
    SAIDA_ESPERADA_MAP = mapear_valores_desejados(SAIDA_ESPERADA)
    return vetor_entrada, SAIDA_ESPERADA_MAP


def mapear_valores_desejados(SAIDA_ESPERADA):
    SAIDA_ESPERADA_MAP = []
    for resultado in SAIDA_ESPERADA:
        if resultado == "H":
            SAIDA_ESPERADA_MAP.append([1, 0, 0])
        elif resultado == "D":
            SAIDA_ESPERADA_MAP.append([0, 1, 0])
        elif resultado == "A":
            SAIDA_ESPERADA_MAP.append([0, 0, 1])
    return SAIDA_ESPERADA_MAP


# Função para inicializar os pesos
def inicializar_pesos(numero_de_pesos):
    """
    Inicializa os pesos com valores aleatórios.

    Args:
        numero_de_pesos (int): O número de pesos a serem inicializados.

    Returns:
        array: Um array de pesos inicializados aleatoriamente.
    """
    return np.random.rand(numero_de_pesos)


# Função para calcular u = summation_unit
def calcular_potencial_de_ativacao_neuronio(
    pesos_neuronio, vetor_entrada, bias, qtd_dados
):
    """
    Calcula o somatorio de um neuronio na camada intermediaria ou de saida.

    Args:
        pesos_neuronio (array): Os pesos do neuronio.
        vetor_entrada (array): O vetor de entrada.
        bias (float): O valor do bias.
        qtd_dados (int): O número de dados no vetor de entrada.

    Returns:
        float: O somatorio do neuronio.
    """
    summation_unit = pesos_neuronio[0] * bias
    for neuronio in range(qtd_dados):
        summation_unit += vetor_entrada[neuronio] * pesos_neuronio[neuronio + 1]
    return summation_unit


# Função de transferência/ativação --> Usado: Função sigmoide
def sigmoide(summation_unit):
    """
    Calcula a função de ativação sigmoide para um neuronio.

    Args:
        summation_unit (float): O somatorio do neuronio.

    Returns:
        float: O valor da função sigmoide.
    """
    return 1 / (1 + np.exp(-summation_unit))


# Derivada da função sigmoide --> ESTÁ CONTIDA NO CÁLCULO DO ERRO DO neuronio
def sigmoide_derivada(SAIDA_REDE):
    """
    Calcula a derivada da função sigmoide.

    Args:
        SAIDA_REDE (float): A saida do neuronio.

    Returns:
        float: O valor da derivada da sigmoide.
    """
    return SAIDA_REDE * (1 - SAIDA_REDE)


# Cálculo do erro(delta) para a camada de saida
def calcular_erro_saida(SAIDA_ESPERADA, SAIDA_REDE):
    """
    Calcula o erro para a camada de saida.

    Args:
        SAIDA_ESPERADA (float): O valor desejado.
        SAIDA_REDE (float): A saida da rede neural.

    Returns:
        float: O valor do erro.
    """
    FATOR_ERRO = SAIDA_ESPERADA - SAIDA_REDE
    derivada_sigmoide = sigmoide_derivada(SAIDA_REDE)
    erro = FATOR_ERRO * derivada_sigmoide
    return erro


# Cálculo do erro(delta) para a camada intermediaria
def erro_intermediaria(SAIDA_REDE, erro_p, indice_n, pesos_neuronio_p, qtd_neuronio_p):
    """
    Calcula o erro para a camada intermediaria.

    Args:
        SAIDA_REDE (float): A saida do neuronio na camada intermediaria.
        erro_p (array): Os erros da camada de saida.
        indice_n (int): O índice do neuronio na camada intermediaria.
        pesos_neuronio_p (array): Os pesos do neuronio na camada de saida.
        qtd_neuronio_p (int): O número de neuronios na camada de saida.

    Returns:
        float: O valor do erro para a camada intermediaria.
    """
    derivada = sigmoide_derivada(SAIDA_REDE)
    sum_erro = 0
    for neuronio in range(qtd_neuronio_p):
        sum_erro += erro_p[neuronio] * pesos_neuronio_p[neuronio][indice_n + 1]
    return derivada * sum_erro


# Função principal
def interface():
    """
    Função principal que treina a rede neural e a utiliza para classificar amostras.

    """
    # Define constantes e inicializa os pesos
    np.random.seed(0)
    vetor_entrada, SAIDA_ESPERADA = carregar_dados(NOME_ARQUIVO)
    QTD_AMOSTRAS = len(vetor_entrada)
    vetor_entrada = np.array(vetor_entrada)
    SAIDA_ESPERADA = np.array(SAIDA_ESPERADA)
    pesos_neuronio1 = np.array(
        [
            inicializar_pesos(NEURONIOS_CAMADA_INTERMEDIARIA)
            for _ in range(NEURONIOS_CAMADA_INTERMEDIARIA)
        ]
    )  # Pesos da camada intermediaria
    pesos_neuronio2 = np.array(
        [inicializar_pesos(TAM_PESOS_SAIDA) for _ in range(NEURONIOS_CAMADA_SAIDA)]
    )  # Pesos da camada de saida
    bias1 = np.ones(
        NEURONIOS_CAMADA_INTERMEDIARIA
    )  # cria um array NumPy preenchido com o valor 1.0 (float) em cada elemento no tamanho da camada intermediaria
    bias2 = np.ones(
        NEURONIOS_CAMADA_SAIDA
    )  # cria um array NumPy preenchido com o valor 1.0 (float) em cada elemento no tamanho da camada de saida
    summation_unit1 = np.zeros(
        NEURONIOS_CAMADA_INTERMEDIARIA # inicializa o array com 0.0
    )  # somatorio dos neuronios da camada intermediaria
    summation_unit2 = np.zeros(
        NEURONIOS_CAMADA_SAIDA
    )  # somatorio dos neuronios da camada de saida
    SAIDA_REDE_1 = np.zeros(
        NEURONIOS_CAMADA_INTERMEDIARIA
    )  # saidas dos neuronios da camada intermediaria
    SAIDA_REDE_INTERMEDIARIA = np.zeros(
        NEURONIOS_CAMADA_SAIDA
    )  # saidas dos neuronios da camada de saida
    erro1 = np.zeros(
        NEURONIOS_CAMADA_INTERMEDIARIA
    )  # Gradientes dos neuronios da camada intermediaria
    erro2 = np.zeros(
        NEURONIOS_CAMADA_SAIDA
    )  # Gradientes dos neuronios da camada de saida
    taxa_aprendizagem = 0.3
    momentum = 0.9
    precisao_requerida = 1e-6
    epocas = 0
    erro_quadratico_medio = 0
    erro_quadratico_medio_atual = 0
    erro_quadratico_medio_anterior = 0
    erro_global = 0
    erro_individual = 0
    # Loop de treinamento

    ###PRINT 1A CAMADA

    while True:
        erro_quadratico_medio_anterior = erro_quadratico_medio_atual
        erro_quadratico_medio = 0
        erro_global = 0
        erro_individual = 0
        # Início Fase Forward
        for amostra in range(QTD_AMOSTRAS):
            logger.info(
                "\n    --------------------------------------------------------------------------->"
            )
            erro_individual = 0
            # somatorio camada intermediaria
            for neuronio in range(NEURONIOS_CAMADA_INTERMEDIARIA):
                # 1ª FASE
                summation_unit1[neuronio] = calcular_potencial_de_ativacao_neuronio(
                    pesos_neuronio1[neuronio],
                    vetor_entrada[amostra],
                    bias1[neuronio],
                    QTD_DADOS,
                )
                # 2ª FASE
                SAIDA_REDE_1[neuronio] = sigmoide(summation_unit1[neuronio])
            # somatorio camada de saida
            for neuronio in range(NEURONIOS_CAMADA_SAIDA):
                summation_unit2[neuronio] = calcular_potencial_de_ativacao_neuronio(
                    pesos_neuronio2[neuronio],
                    SAIDA_REDE_1,
                    bias2[neuronio],
                    NEURONIOS_CAMADA_INTERMEDIARIA,
                )
                SAIDA_REDE_INTERMEDIARIA[neuronio] = sigmoide(summation_unit2[neuronio])
                # logger.error(f'SAIDA_REDE_INTERMEDIARIA: {SAIDA_REDE_INTERMEDIARIA}')
                # logger.success(f'SAIDA_REDE_INTERMEDIARIA[neuronio]: {SAIDA_REDE_INTERMEDIARIA[neuronio]}')
                # erro
                erro_individual = erro_individual + np.sum(
                    (
                        SAIDA_ESPERADA[amostra][neuronio]
                        - SAIDA_REDE_INTERMEDIARIA[neuronio]
                    )
                    ** 2
                )
            logger.error(f'--> SAIDA_REDE_INTERMEDIARIA: {SAIDA_REDE_INTERMEDIARIA}')
            logger.error(f'--> erro_individual: {erro_individual}')
            erro_global = erro_individual / 2
            erro_quadratico_medio = erro_quadratico_medio + erro_global
            # Fim Forward --------------------------------------------------------------------------->
            logger.info(
                "\n    ---------------------------------------------------------------------------|"
            )

            logger.info(
                "\n<---------------------------------------------------------------------------    "
            )
            # Início Backpropagation <----------------------------------------------------------------
            # Determinar o gradiente da camada de saida
            for neuronio in range(NEURONIOS_CAMADA_SAIDA):
                erro2[neuronio] = calcular_erro_saida(
                    SAIDA_ESPERADA[amostra][neuronio],
                    SAIDA_REDE_INTERMEDIARIA[neuronio],
                )
            # Determinar o gradiente da camada intermediaria
            for neuronio in range(NEURONIOS_CAMADA_INTERMEDIARIA):
                erro1[neuronio] = erro_intermediaria(
                    SAIDA_REDE_1[neuronio],
                    erro2,
                    neuronio,
                    pesos_neuronio2,
                    NEURONIOS_CAMADA_SAIDA,
                )

            # Ajuste de pesos da camada de saida
            for neuronio in range(NEURONIOS_CAMADA_SAIDA):
                # Atualizando o bias da camada de saida
                pesos_neuronio2[neuronio][0] += (
                    taxa_aprendizagem * erro2[neuronio] * bias2[neuronio]
                )

                # Atualizando os pesos
                for peso_index in range(1, TAM_PESOS_SAIDA):
                    pesos_neuronio2[neuronio][peso_index] += (
                        taxa_aprendizagem
                        * erro2[neuronio]
                        * SAIDA_REDE_1[peso_index - 1]
                    )
            # Ajuste de pesos da camada intermediaria
            for neuronio in range(NEURONIOS_CAMADA_INTERMEDIARIA):
                # Atualizando o bias da camada intermediaria
                pesos_neuronio1[neuronio][0] += (
                    taxa_aprendizagem * erro1[neuronio] * bias1[neuronio]
                )

                # Atualizando os pesos
                for peso_index in range(1, NEURONIOS_CAMADA_INTERMEDIARIA):
                    pesos_neuronio1[neuronio][peso_index] += (
                        taxa_aprendizagem
                        * erro1[neuronio]
                        * vetor_entrada[amostra][peso_index - 1]
                    )
        erro_quadratico_medio = erro_quadratico_medio / QTD_AMOSTRAS
        erro_quadratico_medio_atual = erro_quadratico_medio
        epocas += 1
        logger.info(
            f"epoca {epocas}, erro_quadratico_medio {erro_quadratico_medio_atual}"
        )
        if (
            abs(erro_quadratico_medio_atual - erro_quadratico_medio_anterior)
            <= precisao_requerida
        ):
            break
        logger.info(
            "\n-------------------------------------------------------------------------------"
        )
    logger.info(f"REDE TREINADA COM {epocas} epocas")
    # Fase de operação
    for amostra in range(QTD_AMOSTRAS):
        for neuronio in range(NEURONIOS_CAMADA_INTERMEDIARIA):
            # 1ª FASE
            summation_unit1[neuronio] = calcular_potencial_de_ativacao_neuronio(
                pesos_neuronio1[neuronio],
                vetor_entrada[amostra],
                bias1[neuronio],
                QTD_DADOS,
            )
            # 2ª FASE
            SAIDA_REDE_1[neuronio] = sigmoide(summation_unit1[neuronio])
            logger.info(
                f"\nsomatorio neuronio da Camada intermediaria\n --> Potencial: {summation_unit1[neuronio]}\n --> Pesos: {pesos_neuronio1[neuronio]}\n --> Amostra: {vetor_entrada[amostra]}\n --> Bias: {bias1[neuronio]}\n --> QTD_DADOS: {QTD_DADOS}\n",
                end="",
            )
            logger.info(f"\n --> saida: {SAIDA_REDE_1[neuronio]}\n")

        logger.info(f"\nAmostra {amostra} ", end="")

        for neuronio in range(NEURONIOS_CAMADA_SAIDA):
            summation_unit2[neuronio] = calcular_potencial_de_ativacao_neuronio(
                pesos_neuronio2[neuronio],
                SAIDA_REDE_1,
                bias2[neuronio],
                NEURONIOS_CAMADA_INTERMEDIARIA,
            )
            SAIDA_REDE_INTERMEDIARIA[neuronio] = sigmoide(summation_unit2[neuronio])
            logger.warning(f'______SAIDA_REDE_INTERMEDIARIA[neuronio]: {SAIDA_REDE_INTERMEDIARIA[neuronio]}')
            ### nomear essa função
            if SAIDA_REDE_INTERMEDIARIA[neuronio] > 0.5:
                logger.info(f"{1.0} ", end="")
            else:
                logger.info(f"{0.0} ", end="")
            ### receber as infos e fazer a sigmoidal da 3
            ### calcular o erro depois


if __name__ == "__main__":
    interface()
