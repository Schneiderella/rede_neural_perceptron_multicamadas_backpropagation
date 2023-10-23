import numpy as np
import pandas as pd
from loguru import logger


def carregar_dados(nome_arquivo):
    """
    Carrega dados de um arquivo CSV, processa e retorna os vetores de entrada e os valores desejados mapeados.

    Args:
        nome_arquivo (str): O caminho para o arquivo CSV.

    Returns:
        tuple: Uma tupla contendo o vetor de entrada e os valores desejados mapeados.
    """
    # Leitura do arquivo CSV
    dados_csv = pd.read_csv(nome_arquivo, delimiter=";")
    # Selecionar as colunas relevantes para as amostras de entrada (vetor_entrada) e os resultados desejados (valor_desejado)
    entrada_columns = ["HS", "AS", "HST", "AST", "HC", "AC", "HR", "AR"]
    valor_desejado_column = "FTR"
    # Extrair os dados de entrada (vetor_entrada) e resultados desejados (valor_desejado)
    vetor_entrada = dados_csv[entrada_columns].values
    valor_desejado = dados_csv[valor_desejado_column].values
    # Mapear os resultados (H, valor_desejado, A) para vetores (1, 0, 0), (0, 1, 0) e (0, 0, 1)
    valor_desejado_map = []
    for resultado in valor_desejado:
        if resultado == "H":
            valor_desejado_map.append([1, 0, 0])
        elif resultado == "D":
            valor_desejado_map.append([0, 1, 0])
        elif resultado == "A":
            valor_desejado_map.append([0, 0, 1])
    return vetor_entrada, valor_desejado_map


# Constantes
NEURONIOS_CAMADA_ENTRADA = 8
NEURONIOS_CAMADA_OCULTA = 6
NEURONIOS_CAMADA_SAIDA = 3
QTD_DADOS = 8  # Número de neurônios na camada de entrada
TAM_PESOS_OCULTA = QTD_DADOS + 1  # Mais um do bias
TAM_PESOS_SAIDA = NEURONIOS_CAMADA_OCULTA + 1  # Mais um do bias
nome_arquivo = "./dataset_futebol2.csv"


# Função para calcular u = potencial_ativacao_neuronio
def calcular_potencial_de_ativacao_neuronio(
    pesos_neuronio, vetor_entrada, bias, qtd_dados
):
    """
    Calcula o potencial de ativação de um neurônio na camada oculta ou de saída.

    Args:
        pesos_neuronio (array): Os pesos do neurônio.
        vetor_entrada (array): O vetor de entrada.
        bias (float): O valor do bias.
        qtd_dados (int): O número de dados no vetor de entrada.

    Returns:
        float: O potencial de ativação do neurônio.
    """
    potencial_ativacao_neuronio = pesos_neuronio[0] * bias
    for neuronio in range(qtd_dados):
        potencial_ativacao_neuronio += (
            vetor_entrada[neuronio] * pesos_neuronio[neuronio + 1]
        )
    return potencial_ativacao_neuronio


# Função sigmoide
def sigmoide(potencial_ativacao_neuronio):
    """
    Calcula a função de ativação sigmoide para um neurônio.

    Args:
        potencial_ativacao_neuronio (float): O potencial de ativação do neurônio.

    Returns:
        float: O valor da função sigmoide.
    """
    return 1 / (1 + np.exp(-potencial_ativacao_neuronio))


# Derivada da função sigmoide
def sigmoide_derivada(saida_rede):
    """
    Calcula a derivada da função sigmoide.

    Args:
        saida_rede (float): A saída do neurônio.

    Returns:
        float: O valor da derivada da sigmoide.
    """
    return saida_rede * (1 - saida_rede)


# Cálculo do delta para a camada de saída
def calcular_delta_saida(valor_desejado, saida_rede):
    """
    Calcula o delta para a camada de saída.

    Args:
        valor_desejado (float): O valor desejado.
        saida_rede (float): A saída da rede neural.

    Returns:
        float: O valor do delta.
    """
    erro = valor_desejado - saida_rede
    derivada_sigmoide = sigmoide_derivada(saida_rede)
    delta = erro * derivada_sigmoide
    return delta


# Cálculo do delta para a camada oculta
def deltaOculta(saida_rede, deltaP, indiceN, pesos_neuronioP, qtd_neuronio_P):
    """
    Calcula o delta para a camada oculta.

    Args:
        saida_rede (float): A saída do neurônio na camada oculta.
        deltaP (array): Os deltas da camada de saída.
        indiceN (int): O índice do neurônio na camada oculta.
        pesos_neuronioP (array): Os pesos do neurônio na camada de saída.
        qtd_neuronio_P (int): O número de neurônios na camada de saída.

    Returns:
        float: O valor do delta para a camada oculta.
    """
    derivada = sigmoide_derivada(saida_rede)
    sumDelta = 0
    for neuronio in range(qtd_neuronio_P):
        sumDelta += deltaP[neuronio] * pesos_neuronioP[neuronio][indiceN + 1]
    return derivada * sumDelta


# Função para inicializar os pesos
def inicializaPesos(numero_de_pesos):
    """
    Inicializa os pesos com valores aleatórios.

    Args:
        numero_de_pesos (int): O número de pesos a serem inicializados.

    Returns:
        array: Um array de pesos inicializados aleatoriamente.
    """
    return np.random.rand(numero_de_pesos)


# Função principal
def interface():
    """
    Função principal que treina uma rede neural e a utiliza para classificar amostras.

    """
    # Defina constantes e inicialize os pesos e bias
    np.random.seed(0)
    vetor_entrada, valor_desejado = carregar_dados(nome_arquivo)
    QTD_AMOSTRAS = len(vetor_entrada)
    vetor_entrada = np.array(vetor_entrada)
    valor_desejado = np.array(valor_desejado)
    pesos_neuronio1 = np.array(
        [inicializaPesos(TAM_PESOS_OCULTA) for _ in range(NEURONIOS_CAMADA_OCULTA)]
    )  # Pesos da camada oculta
    pesos_neuronio2 = np.array(
        [inicializaPesos(TAM_PESOS_SAIDA) for _ in range(NEURONIOS_CAMADA_SAIDA)]
    )  # Pesos da camada de saída
    bias1 = np.ones(NEURONIOS_CAMADA_OCULTA)
    bias2 = np.ones(NEURONIOS_CAMADA_SAIDA)
    potencial_ativacao_neuronio1 = np.zeros(
        NEURONIOS_CAMADA_OCULTA
    )  # Potencial de ativação dos neurônios da camada oculta
    potencial_ativacao_neuronio2 = np.zeros(
        NEURONIOS_CAMADA_SAIDA
    )  # Potencial de ativação dos neurônios da camada de saída
    saida_rede1 = np.zeros(
        NEURONIOS_CAMADA_OCULTA
    )  # Saídas dos neurônios da camada oculta
    saida_rede2 = np.zeros(
        NEURONIOS_CAMADA_SAIDA
    )  # Saídas dos neurônios da camada de saída
    delta1 = np.zeros(
        NEURONIOS_CAMADA_OCULTA
    )  # Gradientes dos neurônios da camada oculta
    delta2 = np.zeros(
        NEURONIOS_CAMADA_SAIDA
    )  # Gradientes dos neurônios da camada de saída
    taxa_aprendizagem = 0.5
    precisao_requerida = 1e-6
    epocas = 0
    erro_quadratico_medio = 0
    erro_quadratico_medio_atual = 0
    erro_quadratico_medio_anterior = 0
    erro_global = 0
    erro_individual = 0
    # Loop de treinamento
    while True:
        erro_quadratico_medio_anterior = erro_quadratico_medio_atual
        erro_quadratico_medio = 0
        erro_global = 0
        erro_individual = 0
        # Início Fase Forward
        for amostra in range(QTD_AMOSTRAS):
            erro_individual = 0
            # Potencial de ativação camada oculta
            for neuronio in range(NEURONIOS_CAMADA_OCULTA):
                potencial_ativacao_neuronio1[
                    neuronio
                ] = calcular_potencial_de_ativacao_neuronio(
                    pesos_neuronio1[neuronio],
                    vetor_entrada[amostra],
                    bias1[neuronio],
                    QTD_DADOS,
                )
                saida_rede1[neuronio] = sigmoide(potencial_ativacao_neuronio1[neuronio])
            # Potencial de ativação camada de saída
            for neuronio in range(NEURONIOS_CAMADA_SAIDA):
                potencial_ativacao_neuronio2[
                    neuronio
                ] = calcular_potencial_de_ativacao_neuronio(
                    pesos_neuronio2[neuronio],
                    saida_rede1,
                    bias2[neuronio],
                    NEURONIOS_CAMADA_OCULTA,
                )
                saida_rede2[neuronio] = sigmoide(potencial_ativacao_neuronio2[neuronio])
                erro_individual = erro_individual + np.sum(
                    (valor_desejado[amostra][neuronio] - saida_rede2[neuronio]) ** 2
                )
            erro_global = erro_individual / 2
            erro_quadratico_medio = erro_quadratico_medio + erro_global
            # Fim da fase Forward
            # Início da fase Backward
            # Determinar o gradiente da camada de saída
            for neuronio in range(NEURONIOS_CAMADA_SAIDA):
                delta2[neuronio] = calcular_delta_saida(
                    valor_desejado[amostra][neuronio], saida_rede2[neuronio]
                )
            # Determinar o gradiente da camada oculta
            for neuronio in range(NEURONIOS_CAMADA_OCULTA):
                delta1[neuronio] = deltaOculta(
                    saida_rede1[neuronio],
                    delta2,
                    neuronio,
                    pesos_neuronio2,
                    NEURONIOS_CAMADA_SAIDA,
                )
            # Ajuste de pesos da camada de saída
            for neuronio in range(NEURONIOS_CAMADA_SAIDA):
                # Atualizando o bias da camada de saída
                pesos_neuronio2[neuronio][0] += (
                    taxa_aprendizagem * delta2[neuronio] * bias2[neuronio]
                )
                # Atualizando os pesos
                for peso_index in range(1, TAM_PESOS_SAIDA):
                    pesos_neuronio2[neuronio][peso_index] += (
                        taxa_aprendizagem
                        * delta2[neuronio]
                        * saida_rede1[peso_index - 1]
                    )
            # Ajuste de pesos da camada oculta
            for neuronio in range(NEURONIOS_CAMADA_OCULTA):
                # Atualizando o bias da camada oculta
                pesos_neuronio1[neuronio][0] += (
                    taxa_aprendizagem * delta1[neuronio] * bias1[neuronio]
                )
                # Atualizando os pesos
                for peso_index in range(1, TAM_PESOS_OCULTA):
                    pesos_neuronio1[neuronio][peso_index] += (
                        taxa_aprendizagem
                        * delta1[neuronio]
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
    logger.info(f"REDE TREINADA COM {epocas} epocas")
    # Fase de operação
    for amostra in range(QTD_AMOSTRAS):
        for neuronio in range(NEURONIOS_CAMADA_OCULTA):
            potencial_ativacao_neuronio1[
                neuronio
            ] = calcular_potencial_de_ativacao_neuronio(
                pesos_neuronio1[neuronio],
                vetor_entrada[amostra],
                bias1[neuronio],
                QTD_DADOS,
            )
            saida_rede1[neuronio] = sigmoide(potencial_ativacao_neuronio1[neuronio])
            logger.info(
                f"\nPotencial de Ativação Neurônio da Camada Oculta\n --> Potencial: {potencial_ativacao_neuronio1[neuronio]}\n --> Pesos: {pesos_neuronio1[neuronio]}\n --> Amostra: {vetor_entrada[amostra]}\n --> Bias: {bias1[neuronio]}\n --> QTD_DADOS: {QTD_DADOS}\n",
                end="",
            )
            logger.info(f"\n --> Saída: {saida_rede1[neuronio]}\n")

        logger.info(f"\nAmostra {amostra} ", end="")

        for neuronio in range(NEURONIOS_CAMADA_SAIDA):
            potencial_ativacao_neuronio2[
                neuronio
            ] = calcular_potencial_de_ativacao_neuronio(
                pesos_neuronio2[neuronio],
                saida_rede1,
                bias2[neuronio],
                NEURONIOS_CAMADA_OCULTA,
            )
            saida_rede2[neuronio] = sigmoide(potencial_ativacao_neuronio2[neuronio])

            if saida_rede2[neuronio] > 0.5:
                logger.info(f"{1.0} ", end="")
            else:
                logger.info(f"{0.0} ", end="")


if __name__ == "__main__":
    interface()
