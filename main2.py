import numpy as np
import pandas as pd
import csv
from loguru import logger



def carregar_dados(nome_arquivo):
    # Leitura do arquivo CSV
    dados_csv = pd.read_csv(nome_arquivo, delimiter=';')

    # Selecionar as colunas relevantes para as amostras de entrada (x) e os resultados desejados (d)
    x_columns = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HR', 'AR']
    d_column = 'FTR'

    # Extrair os dados de entrada (x) e resultados desejados (d)
    x = dados_csv[x_columns].values
    d = dados_csv[d_column].values

    # Mapear os resultados (H, D, A) para vetores (1, 0, 0), (0, 1, 0) e (0, 0, 1)
    d_mapped = []
    for resultado in d:
        if resultado == 'H':
            d_mapped.append([1, 0, 0])
        elif resultado == 'D':
            d_mapped.append([0, 1, 0])
        elif resultado == 'A':
            d_mapped.append([0, 0, 1])

    return x, d_mapped
    

# Definindo as constantes
NEURONIOS_CAMADA_ENTRADA = 8
NEURONIOS_CAMADA_OCULTA = 6
NEURONIOS_CAMADA_SAIDA = 3

QTD_DADOS = 8  # Número de neurônios na camada de entrada
TAM_PESOS_OCULTA = QTD_DADOS + 1  # Mais um do bias
TAM_PESOS_SAIDA = NEURONIOS_CAMADA_OCULTA + 1  # Mais um do bias
nome_arquivo = './dataset_futebol2.csv'


# Função para calcular u
def u(w, x, bias, qtd_dados):
    u = w[0] * bias
    for i in range(qtd_dados):
        u += x[i] * w[i + 1]
    return u

# Função sigmoide
def sigmoide(u):
    return 1 / (1 + np.exp(-u))

# Derivada da função sigmoide
def sigmoideDerivada(y):
    return y * (1 - y)

# Cálculo do delta para a camada de saída
def deltaSaida(d, y):
    erro = d - y
    derivada = sigmoideDerivada(y)
    return erro * derivada

# Cálculo do delta para a camada oculta
def deltaOculta(y, deltaP, indiceN, wP, qtdNeuronioP):
    derivada = sigmoideDerivada(y)
    sumDelta = 0
    for j in range(qtdNeuronioP):
        sumDelta += deltaP[j] * wP[j][indiceN + 1]
    return derivada * sumDelta

# Função para inicializar os pesos
def inicializaPesos(c):
    return np.random.rand(c)

# Função principal
def main():
    np.random.seed(0)
    # carregar_dados(nome_arquivo)
    # Read data from a CSV file (replace 'data.csv' with your file name)
    x, d = carregar_dados(nome_arquivo)
    QTD_AMOSTRAS = len(x)
    # x, d = np.array([carregar_dados(nome_arquivo)])
    # Obter o conjunto de amostras de treinamento {x{k}}
    # x = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 1]
    # ])

    # Associar o vetor de saída desejada {d{k}} para cada amostra
    # d = np.array([
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1]
    # ])

    x = np.array(x)
    d = np.array(d)

    # Iniciar wji(1) e wji(2) com valores aleatórios pequenos
    w1 = np.array([inicializaPesos(TAM_PESOS_OCULTA) for _ in range(NEURONIOS_CAMADA_OCULTA)])  # Pesos da camada oculta
    w2 = np.array([inicializaPesos(TAM_PESOS_SAIDA) for _ in range(NEURONIOS_CAMADA_SAIDA)])  # Pesos da camada de saída

    bias1 = np.ones(NEURONIOS_CAMADA_OCULTA)
    bias2 = np.ones(NEURONIOS_CAMADA_SAIDA)

    u1 = np.zeros(NEURONIOS_CAMADA_OCULTA)  # Potencial de ativação dos neurônios da camada oculta
    u2 = np.zeros(NEURONIOS_CAMADA_SAIDA)  # Potencial de ativação dos neurônios da camada de saída

    y1 = np.zeros(NEURONIOS_CAMADA_OCULTA)  # Saídas dos neurônios da camada oculta
    y2 = np.zeros(NEURONIOS_CAMADA_SAIDA)  # Saídas dos neurônios da camada de saída

    delta1 = np.zeros(NEURONIOS_CAMADA_OCULTA)  # Gradientes dos neurônios da camada oculta
    delta2 = np.zeros(NEURONIOS_CAMADA_SAIDA)  # Gradientes dos neurônios da camada de saída

    # Especificar a taxa de aprendizagem {n} e precisão requerida {e}
    n = 0.5
    e = 1e-6
    epocas = 0
    EQM = 0
    EQMatual = 0
    EQManterior = 0
    Eglobal = 0
    errinho = 0

    


    while True:
        EQManterior = EQMatual
        EQM = 0
        Eglobal = 0
        errinho = 0

        # Início Fase Forward
        for k in range(QTD_AMOSTRAS):
            errinho = 0
            # Potencial de ativação camada oculta
            for j in range(NEURONIOS_CAMADA_OCULTA):
                u1[j] = u(w1[j], x[k], bias1[j], QTD_DADOS)
                y1[j] = sigmoide(u1[j])
            # Potencial de ativação camada de saída
            for j in range(NEURONIOS_CAMADA_SAIDA):
                u2[j] = u(w2[j], y1, bias2[j], NEURONIOS_CAMADA_OCULTA)
                y2[j] = sigmoide(u2[j])
                errinho = errinho + np.sum((d[k][j] - y2[j]) ** 2)
            Eglobal = errinho / 2
            EQM = EQM + Eglobal
            # Fim da fase Forward

            # Início da fase Backward
            # Determinar o gradiente da camada de saída
            for j in range(NEURONIOS_CAMADA_SAIDA):
                delta2[j] = deltaSaida(d[k][j], y2[j])

            # Determinar o gradiente da camada oculta
            for j in range(NEURONIOS_CAMADA_OCULTA):
                delta1[j] = deltaOculta(y1[j], delta2, j, w2, NEURONIOS_CAMADA_SAIDA)

            # Ajuste de pesos da camada de saída
            for j in range(NEURONIOS_CAMADA_SAIDA):
                # Atualizando o bias da camada de saída
                w2[j][0] += n * delta2[j] * bias2[j]
                # Atualizando os pesos
                for l in range(1, TAM_PESOS_SAIDA):
                    w2[j][l] += n * delta2[j] * y1[l - 1]

            # Ajuste de pesos da camada oculta
            for j in range(NEURONIOS_CAMADA_OCULTA):
                # Atualizando o bias da camada oculta
                w1[j][0] += n * delta1[j] * bias1[j]
                # Atualizando os pesos
                for l in range(1, TAM_PESOS_OCULTA):
                    w1[j][l] += n * delta1[j] * x[k][l - 1]

        EQM = EQM / QTD_AMOSTRAS
        EQMatual = EQM
        epocas += 1
        print(f"epoca {epocas}, EQM {EQMatual}")

        if abs(EQMatual - EQManterior) <= e:
            break

    print(f"REDE TREINADA COM {epocas} epocas")

    # Fase de operação
    for k in range(QTD_AMOSTRAS):
        for j in range(NEURONIOS_CAMADA_OCULTA):
            u1[j] = u(w1[j], x[k], bias1[j], QTD_DADOS)
            y1[j] = sigmoide(u1[j])
            print(f"\n____ u1[j] = u(w1[j], x[k], bias1[j], QTD_DADOS)\n --> u1[j]: {u1[j]}\n --> w1[j]: {w1[j]},\n --> x[k]: {x[k]},\n --> bias1[j]: {bias1[j]},\n --> QTD_DADOS: {QTD_DADOS}\n", end='')
            print(f"\n --> y1[j]: {y1[j]}\n")
        print(f"\nAmostra {k} ", end='')
        for j in range(NEURONIOS_CAMADA_SAIDA):
            u2[j] = u(w2[j], y1, bias2[j], NEURONIOS_CAMADA_OCULTA)
            y2[j] = sigmoide(u2[j])
            if y2[j] > 0.5:
                print(f"{1.0} ", end='')
            else:
                print(f"{0.0} ", end='')

if __name__ == "__main__":
    main()
