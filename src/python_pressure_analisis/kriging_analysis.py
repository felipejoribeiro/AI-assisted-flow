##############################################################################
# Programa para teste do kpls com método recorrente.
# Aluno: Felipe J. O. Ribeiro (11711EAR012)
# Professor: Aldemir Cavalini jr
# versão: 3.2.0
###############################################################################
#                                                                             #
# import tqdm
import time as tm
import numpy as np
import scipy as sp
from scipy import signal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from smt.surrogate_models import KRG, KPLS
from math import sqrt
import h5py
import psutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# (todas as versões mais recentes para python4 devem funcionar
# com instalação padrão via pip3)
#                                                                             #
# Funções:                                                                    #
###############################################################################
#                                                                             #
# Função que faz amostragem de vetor no console (para propósitos de debugger):
def most(vetor):
    '''
    Mostra vetor i / element
    '''
    for element_index, element in enumerate(vetor):
        print(element_index, element)


def tam(vetor):
    '''
    vector length
    '''
    print(len(vetor))


# Get current time:
def get_time(): return tm.time()


# Error check, returns l1, l2, l3
def error_check(tested, reference):
    '''
    Checking l1,l2 and l3
    '''
    # Norma L1 dos vetores de tensão:
    soma = 0
    for i in range(len(tested)):
        soma = soma + abs(tested[i] - reference[i])
    L1 = float(soma/len(tested))

    # Norma L2 dos vetores de tensão:
    soma = 0
    for i in range(len(tested)):
        soma = soma + (tested[i] - reference[i])**2
    L2 = sqrt(soma/len(tested))

    # Norma Li dos vetores de tensão:
    soma = 0
    for i in range(len(tested)):
        if soma < abs(tested[i] - reference[i]):
            soma = abs(tested[i] - reference[i])
    Li = float(soma)

    return [L1, L2, Li]


# Função que abre arquivo e devolve os dados:
def open_archive(folder_path, initial_time, final_time, domain):
    '''
    Take data from simulation
    '''
    value_to_grep = 2 # Preassure

    # General data
    data = []

    # Lendo as linhas da série temporal.
    for step in range(initial_time, final_time):
        # Read from time step file:
        step_file = open(folder_path + "/step" + str(step) + ".dat")

        column_counter = 0
        time_step_data = []
        row_data = []
        for line in step_file:
            if column_counter > 2:
                column_counter_fixed = column_counter - 2
                line_data = line.split()
                cell_value = line_data[value_to_grep]
                row_data.append(float(cell_value))

                # Checks if row has ended
                if column_counter_fixed == domain[0]:
                    time_step_data.append(row_data)
                    row_data = []
                    column_counter = 2
            column_counter +=1
        data.append(time_step_data)
    return data


# Função para processar o dado na forma de forecasting escolhida
def parse_data(input_matrix, target_cells):
    '''
    Create the forecasting lists
    '''
    count = 0
    input_data = []
    print( '    - cells deleted on input:')
    for time_step in input_matrix:
        if count <= len(input_matrix) - 2:
            cell_data = []
            for cell in target_cells:
                cell_data.append(time_step[cell[0]][cell[1]])
            input_data.append(cell_data)
        else:
            print('    ', count)
        count +=1

    count = 0
    output_data = []
    print( '    - cells deleted on output:')
    for time_step in input_matrix:
        if count >= 1:
            output_data.append(time_step[target_cells[0][0]][target_cells[0][1]])
        else:
            print('    ', count)
        count +=1
    return [input_data, output_data]


# Função para predição do kpls
def kpls_predict(training_in, training_out, forecast_in, forecast_out):
    '''
    Predição no método do KPLS
    '''
    sm = KPLS(theta0=[1e-2], poly="constant", corr="abs_exp", print_global=False)
    sm.set_training_values(np.array(training_in), np.array(training_out))
    sm.train()
    prediction = []
    for time_step in range(len(forecast_in)):
        if time_step > -1:
            prev_step = sm.predict_values(np.array([forecast_in[time_step]])).tolist()[0][0]
            prediction.append(prev_step)
        else:
            # WARNING: Hard Coded 1:1 relationship. For creating the 4 x 1 relation, more things would have to change
            new_step = sm.predict_values(np.array([[prev_step]])).tolist()[0][0]
            prediction.append(new_step)
            prev_step = new_step

    error = error_check(prediction, forecast_out)
    return [error, prediction]


# Função para predição do keras
def keras_predict(training_in, training_out, forecast_in, forecast_out):
    '''
    Predição no método do Keras
    '''
    model = Sequential() 
    model.add(Dense(len(training_in[0]), input_dim=len(training_in[0]), activation='linear'))
    model.add(Dense(80, activation='linear')) 
    model.add(Dense(80, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(np.array(training_in), np.array(training_out), batch_size = 64,epochs=50,use_multiprocessing=True, verbose=0)
    prediction = model.predict(np.array(forecast_in))
    error = error_check(prediction, forecast_out)
    return [error, prediction]


if __name__ == "__main__":
    # Controles
    training_time = [0, 150]           # Tempo para treino do modelo
    forecasting_time = [150, 300]      # Tempo para predição do modelo
    spacial_domain = [30, 30]          # domínio espacial

    # Element (or elements) To use in training [[1,3], [2,3], [0,3], ...]
    element = [[15,15]]

    # Dados de leitura:
    nome_pasta_entrada =\
    "../fluid_solver/results/cavity_30_30_5.0_50.0_0.2E+06"

    # Dados de escrita:
    identificador = "result.h5"
    nome_arquivo_saida = "output/krigging_"\
        + identificador

    # Initial time measure:
    start_time = get_time()

    # Reading data
    print("=> Reading input data...")
    all_data_from_time_training = open_archive(nome_pasta_entrada,
                                               training_time[0],
                                               training_time[1], spacial_domain)
    all_data_from_time_forecasting = open_archive(nome_pasta_entrada,
                                                  forecasting_time[0],
                                                  forecasting_time[1], spacial_domain)
    print("     - done.")

    # Parcing data
    print("=>  Parsing data to chosen formating...")
    [training_in, training_out] = parse_data(all_data_from_time_training, element)
    [forecast_in, forecast_out] = parse_data(all_data_from_time_forecasting, element)
    print("     - done.")

    # Forecasting processes 
    print("=>  Forecasting based on training data...")
    [error_kpls, prediction_kpls] = kpls_predict(training_in, training_out,
                                                 forecast_in, forecast_out)
    # [error_keras, prediction_keras] = keras_predict(training_in, training_out,
    #                                              forecast_in, forecast_out)
    print("     - done.")

    # Results
    print("=>  Showing results...")
    print("        KPLS error:", error_kpls)
    # print("    KERAS error:", error_keras)
    plt.figure()
    plt.plot([x for x in range(len(training_in))], training_in, label="Training data")
    plt.plot([training_time[1] + x for x in range(len(forecast_out))], forecast_out, label="Reference data")
    plt.plot([training_time[1] + x for x in range(len(prediction_kpls))], prediction_kpls, label="KPLS Forecast")
    # plt.plot([100 + x for x in range(len(prediction_keras))], prediction_keras, label="Keras Forecast")
    plt.legend()
    plt.show()
    print("     - done.")

    print("fim")
    pass
