# Author: Junran Chen
# Date: 2023-June-23
# Function: SOP algorithm
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datetime import datetime, timedelta
import math
from JPS_VEstimLSTM import JPS_VEstimLSTM


def functionAlgorithmSOP(
        SOC, Temperature, Power, FrunSOPOrNot, net, Fh_s, Fh_c, Voltage):
    """SOP estimation algorithm, used for Samsung 30T cell.

        Args:
            SOC: State-of-charge
            Temperature: Temperature
            Power: Power
            FrunSOPOrNot: 1-run discharging SOP; 0- run charging SOP, other - run voltage estimation only.
            net: LSTM model
            Fh_s: LSTM h_s
            Fh_c: LSTM h_c
            Voltage: Was used to update RLS, but it can also be used to update LSTM.
        Returns:
            FchargingSOP:charging SOP
            FdischargingSOP: discharging SOP
            FchargingFlag: 1-SOC limited; 2-current limited; 3-voltage limited
            FdischargingFlag: 1-SOC limited; 2-current limited; 3-voltage limited
            Fh_s: LSTM hidden states
            Fh_c: LSTM hidden states
            y_pred: Estimated voltage
        """
    # ---------------------Parameter initialization----------------------------------
    minLimitVoltage = 3.2
    maxLimitVoltage = 4.1
    minLimitCurrent = -30
    maxLimitCurrent = 5
    voltageTolerance = 0.005  # 5 mV
    currentTolerance = 0.01  # 10 mA
    pulseTime = 10  # Power pulse time
    nomCapacity = 3
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # select CPU or GPU
    device = 'cpu'
    net.eval()
    # ---------------------Keep running LSTm to update network------------------------
    tempX_LSTM = torch.tensor([[SOC, Temperature, Power]])
    # testLoader = data.DataLoader(data.TensorDataset(tempX_LSTM, Voltage), shuffle=False, batch_size=1)
    with torch.no_grad():
        # for X_batch in testLoader:
        tempX_LSTM = tempX_LSTM.to(device)
        y_pred, (Fh_s, Fh_c) = net(tempX_LSTM, Fh_s, Fh_c)
        FchargingSOP = 0  # remember to delete.
        FchargingFlag = 0
        FdischargingSOP = 0
        FdischargingFlag = 0
        if FrunSOPOrNot == 1:
            # ----------------------discharging SOP estimation------------------------------
            iteration_steps = 0
            maxPowerBinary = -200
            minPowerBinary = 0
            updatedPower = (maxPowerBinary + minPowerBinary) / 2
            TemperatureTemp = Temperature
            while abs(minPowerBinary) < abs(maxPowerBinary):
                updatedPowerTemp = updatedPower
                voltagePulse = []
                currentPulse = []
                SOCTemp = SOC
                if SOC < 0:  # if SOC is not correct, do not run the SOP
                    FdischargingSOP = 0
                    FdischargingFlag = 1
                    break
                Fh_s_SOP, Fh_c_SOP = Fh_s, Fh_c
                Fh_s_SOP.to(device)  # Maybe I cannot put it in the GPU here.
                Fh_c_SOP.to(device)
                for powerPulse in range(pulseTime):
                    tempX_LSTM = torch.tensor([[SOCTemp, TemperatureTemp, updatedPowerTemp]])
                    tempX_LSTM = tempX_LSTM.to(device)
                    y_pred, (Fh_s_SOP, Fh_c_SOP) = net(tempX_LSTM, Fh_s_SOP, Fh_c_SOP)
                    current = updatedPower / y_pred
                    lossSOC = current / 3600
                    SOCTemp = SOCTemp + lossSOC
                    voltagePulse = voltagePulse + y_pred[:, :].cpu().numpy().tolist()
                    currentPulse = currentPulse + current[:, :].cpu().numpy().tolist()
                    if SOCTemp < 0:
                        voltagePulse = np.zeros((pulseTime, 1))
                        currentPulse = np.zeros((pulseTime, 1))
                        break
                # Binary search
                iteration_steps = iteration_steps + 1
                voltagePulse = np.array(voltagePulse)
                currentPulse = np.array(currentPulse)
                if iteration_steps > 50:
                    FdischargingSOP = updatedPower
                    FdischargingFlag = 1
                    break
                if voltagePulse[-1, 0] <= (minLimitVoltage - voltageTolerance) \
                        or voltagePulse[-1, 0] >= (maxLimitVoltage + voltageTolerance) \
                        or currentPulse[-1, 0] <= (minLimitCurrent - currentTolerance) \
                        or currentPulse[-1, 0] >= (maxLimitCurrent + currentTolerance):
                    maxPowerBinary = updatedPower
                elif (voltagePulse[-1, 0] > minLimitVoltage or voltagePulse[-1, 0] < maxLimitVoltage
                      or currentPulse[-1, 0] > minLimitCurrent or currentPulse[-1, 0] < maxLimitCurrent) \
                        and (voltagePulse[-1, 0] < (minLimitVoltage + voltageTolerance)
                             or voltagePulse[-1, 0] > (maxLimitVoltage - voltageTolerance)
                             or currentPulse[-1, 0] < (minLimitCurrent + currentTolerance)
                             or currentPulse[-1, 0] > (maxLimitCurrent - currentTolerance)):
                    if currentPulse[-1, 0] > (maxLimitCurrent - currentTolerance) \
                            or currentPulse[-1, 0] < (minLimitCurrent + currentTolerance):
                        FdischargingFlag = 2  # It is current limited
                    elif voltagePulse[-1, 0] < (minLimitVoltage + voltageTolerance) \
                            or voltagePulse[-1, 0] > (maxLimitVoltage - voltageTolerance):
                        FdischargingFlag = 3  # It is voltage limited
                    FdischargingSOP = updatedPower
                    break
                elif minLimitVoltage < voltagePulse[-1, 0] < maxLimitVoltage \
                        and maxLimitCurrent > currentPulse[-1, 0] > minLimitCurrent:
                    minPowerBinary = updatedPower
                updatedPower = (maxPowerBinary + minPowerBinary) / 2
        elif FrunSOPOrNot == 0:
            # ----------------------charging SOP estimation------------------------------
            iteration_steps = 0
            maxPowerBinary = 200
            minPowerBinary = 0
            updatedPower = (maxPowerBinary + minPowerBinary) / 2
            TemperatureTemp = Temperature
            while abs(minPowerBinary) < abs(maxPowerBinary):
                updatedPowerTemp = updatedPower
                voltagePulse = []
                currentPulse = []
                SOCTemp = SOC
                if SOC > 3:  # if SOC is not correct, do not run the SOP
                    FchargingSOP = 0
                    FchargingFlag = 1
                    break
                Fh_s_SOP, Fh_c_SOP = Fh_s, Fh_c
                Fh_s_SOP.to(device)  # Maybe I cannot put it in the GPU here.
                Fh_c_SOP.to(device)
                for powerPulse in range(pulseTime):
                    tempX_LSTM = torch.tensor([[SOCTemp, TemperatureTemp, updatedPowerTemp]])
                    tempX_LSTM = tempX_LSTM.to(device)
                    y_pred, (Fh_s_SOP, Fh_c_SOP) = net(tempX_LSTM, Fh_s_SOP, Fh_c_SOP)
                    current = updatedPower / y_pred
                    lossSOC = current / 3600 / nomCapacity
                    SOCTemp = SOCTemp + lossSOC
                    voltagePulse = voltagePulse + y_pred[:, :].cpu().numpy().tolist()
                    currentPulse = currentPulse + current[:, :].cpu().numpy().tolist()
                    if SOCTemp > 3:
                        voltagePulse = np.zeros((pulseTime, 1))
                        currentPulse = np.zeros((pulseTime, 1))
                        break
                # Binary search
                iteration_steps = iteration_steps + 1
                voltagePulse = np.array(voltagePulse)
                currentPulse = np.array(currentPulse)
                if iteration_steps > 50:
                    FchargingSOP = updatedPower
                    FchargingFlag = 1
                    break
                if voltagePulse[-1, 0] <= (minLimitVoltage - voltageTolerance) \
                        or voltagePulse[-1, 0] >= (maxLimitVoltage + voltageTolerance) \
                        or currentPulse[-1, 0] <= (minLimitCurrent - currentTolerance) \
                        or currentPulse[-1, 0] >= (maxLimitCurrent + currentTolerance):
                    maxPowerBinary = updatedPower
                elif (voltagePulse[-1, 0] > minLimitVoltage or voltagePulse[-1, 0] < maxLimitVoltage
                      or currentPulse[-1, 0] > minLimitCurrent or currentPulse[-1, 0] < maxLimitCurrent) \
                        and (voltagePulse[-1, 0] < (minLimitVoltage + voltageTolerance)
                             or voltagePulse[-1, 0] > (maxLimitVoltage - voltageTolerance)
                             or currentPulse[-1, 0] < (minLimitCurrent + currentTolerance)
                             or currentPulse[-1, 0] > (maxLimitCurrent - currentTolerance)):
                    if currentPulse[-1, 0] > (maxLimitCurrent - currentTolerance) \
                            or currentPulse[-1, 0] < (minLimitCurrent + currentTolerance):
                        FchargingFlag = 2  # It is current limited
                    elif voltagePulse[-1, 0] < (minLimitVoltage + voltageTolerance) \
                            or voltagePulse[-1, 0] > (maxLimitVoltage - voltageTolerance):
                        FchargingFlag = 3  # It is volt5age limited
                    FchargingSOP = updatedPower
                    break
                elif minLimitVoltage < voltagePulse[-1, 0] < maxLimitVoltage \
                        and maxLimitCurrent > currentPulse[-1, 0] > minLimitCurrent:
                    minPowerBinary = updatedPower
                updatedPower = (maxPowerBinary + minPowerBinary) / 2
    return FchargingSOP, FdischargingSOP, FchargingFlag, FdischargingFlag, Fh_s, Fh_c, y_pred


# ----------------------Run SOP algorithm--------------------------------------------------
# Preparation
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # select CPU or GPU
# device = 'mps'
device = 'cpu'
Data = pd.read_csv('TestFile/10s/SOP_30T_July_13_40degC_Channel_5_Wb_1.CSV')
# Data = pd.read_csv('data/HIL/JC_Jetson_25degC_10s_March27_2_Channel_1_Wb_1.CSV')
# Data = pd.read_csv('data/HIL/JC_Jetson_25degC_10s_March28_cellSOPnew_Channel_1_Wb_1.CSV')
# Data = pd.read_csv('data/HIL/SOP_new_cell/JC_Jetson_trainingSet_US06_and_CCCV_April_1_Channel_1_Wb_1.csv')
# load model and initialize LSTM hidden states.
#-------old model
# # model_lowT = torch.load('result/LSTM_July3.model', map_location=device).get('wmodel').to(device)
# LSTM_LAYERS = 2a
# sizeIndex = 4a
# model = torch.load('result/LSTM_Aug17.model', map_location=device).get('model').to(device)
#-------JPS model
LSTM_LAYERS = 2
sizeIndex = 8
model = JPS_VEstimLSTM(input_size=3,
                       MemoryState_size=2 ** sizeIndex,
                       LSTM_layers=LSTM_LAYERS,
                       FNNhidden_size=2 ** sizeIndex).to(device)
# model.load_state_dict(torch.load('result_JPS/LSTM_Size8_Fold1_Layer2_26_Feb_2025_400.pth', map_location=device))
model.load_state_dict(torch.load('result_JPS/LSTM_Size8_Fold1_Layer2Training_12_March_2025_200.pth', map_location=device))
# model.load_state_dict(torch.load('./result_JPS/LSTM_Size8_Fold1_Layer2Training_2_April_2025_200.pth', map_location=device))
model_hightT = model
dataSOC = Data["SOC"].values.astype('float32')
dataSOCTemp = dataSOC * 3
dataTemperatureTemp = Data["Aux_Temperature_5(C)"].values.astype('float32')
# dataTemperatureTemp = Data["Aux_Temperature_1(thermocouple1(C))"].values.astype('float32')
# dataTemperature = Data["Aux_Temperature_7(thermocouple7(C))"].values.astype('float32')
dataVTemp = Data["Voltage(V)"].values.astype('float32')
dataRunSOPTemp = Data["runSOPflag"].values.astype('float32')
# Interpolate time step
dataPowerTemp = Data["Power(W)"].values.astype('float32')
dataTime = Data["Test_Time(s)"].values.astype('float32')
# Instead of interpolation, we can choose values from increased second.
dataV = []
dataPower = []
dataTemperature = []
dataSOC = []
dataRunSOP = []
for i in range(dataTime.size):
    if dataTime[i] - dataTime[i-1] >= 0.95:
        dataV = np.append(dataV, dataVTemp[i])
        dataPower = np.append(dataPower, dataPowerTemp[i])
        dataTemperature = np.append(dataTemperature, dataTemperatureTemp[i])
        dataSOC = np.append(dataSOC, dataSOCTemp[i])
        dataRunSOP = np.append(dataRunSOP, dataRunSOPTemp[0])  # a way to insert nan
    if math.isnan(dataRunSOPTemp[i]) == False:
        dataRunSOP[-1] = dataRunSOPTemp[i]
dataV = dataV.astype('float32')
dataPower = dataPower.astype('float32')
dataTemperature = dataTemperature.astype('float32')
dataSOC = dataSOC.astype('float32')
dataRunSOP = dataRunSOP.astype('float32')
# Run SOP algorithm
# h_s_lowT = torch.zeros(2, 1, 16).to(device)  # 2-layers, 1-batch, 16-hidden layers.
# h_c_lowT = torch.zeros(2, 1, 16).to(device)
# h_s_highT = torch.zeros(2, 1, 16).to(device)  # 2-layers, 1-batch, 16-hidden layers.
# h_c_highT = torch.zeros(2, 1, 16).to(device)
h_s_highT = torch.zeros(LSTM_LAYERS, 2 ** sizeIndex)
h_c_highT = torch.zeros(LSTM_LAYERS, 2 ** sizeIndex)
h_s_highT = h_s_highT.to(device)
h_c_highT = h_c_highT.to(device)
predV_lowT = []
predV_highT = []
SOP_dis = []
SOP_char = []
SOC_dis = []
SOC_char = []
SOP_dis_flag = []
SOP_char_flag = []
# ---------------------Run a round first to have memory----------------------------
for i in range(dataSOC.size):
    # chargingSOP, dischargingSOP, chargingFlag, dischargingFlag, h_s_lowT, h_c_lowT, tempPredV_lowT = \
    #     functionAlgorithmSOP(dataSOC[i], dataTemperature[i], dataPower[i],
    #                          3, model_lowT, h_s_lowT, h_c_lowT, dataV[i])
    chargingSOP, dischargingSOP, chargingFlag, dischargingFlag, h_s_highT, h_c_highT, tempPredV_highT = \
        functionAlgorithmSOP(dataSOC[i], dataTemperature[i], dataPower[i],
                             3, model_hightT, h_s_highT, h_c_highT, dataV[i])
    # predV_lowT = predV_lowT + tempPredV_lowT[:, -1, :].cpu().numpy().tolist()
    predV_highT = predV_highT + tempPredV_highT[:, :].cpu().numpy().tolist()
torch.save([h_s_highT, h_c_highT], './HiddenStates.pt') # save the hidden states for Jetson validation
predV_lowT = np.asarray(predV_lowT)
predV_highT = np.asarray(predV_highT)
dataV = np.asarray([dataV]).T
compareV = np.concatenate((dataV, predV_highT, dataPower.reshape(-1, 1), dataTemperature.reshape(-1, 1)), axis=1)
error_P = (predV_highT - dataV)
RMSEP = np.sqrt(np.mean(np.square(error_P)))*1000
print(RMSEP)
np.savetxt("./result/Voltage_estimation.csv", compareV)
# ----------------------SOP algorithm running-------------------------------------
for i in range(dataSOC.size):
    # time.sleep(0.2)
    runSOPOrNot = 3  # 3 means do not run SOP algorithm
    if dataRunSOP[i] == 1:
        runSOPOrNot = 1
    elif dataRunSOP[i] == 0:
        runSOPOrNot = 0
    chargingSOP, dischargingSOP, chargingFlag, dischargingFlag, h_s_highT, h_c_highT, tempPredV = \
        functionAlgorithmSOP(dataSOC[i], dataTemperature[i], dataPower[i], runSOPOrNot, model_hightT, h_s_highT,
                             h_c_highT, dataV[i])
    # predV = predV + tempPredV[:, -1, :].cpu().numpy().tolist()
    if runSOPOrNot == 1:
        SOP_dis = np.append(SOP_dis, dischargingSOP)
        SOC_dis = np.append(SOC_dis, dataSOC[i])
        SOP_dis_flag = np.append(SOP_dis_flag, dischargingFlag)
    elif runSOPOrNot == 0:
        SOP_char = np.append(SOP_char, chargingSOP)
        SOC_char = np.append(SOC_char, dataSOC[i])
        SOP_char_flag = np.append(SOP_char_flag, chargingFlag)
SOP_dis = np.asarray([SOP_dis]).T
SOC_dis = np.asarray([SOC_dis]).T
SOP_dis_flag = np.asarray([SOP_dis_flag]).T
SOP_char = np.asarray([SOP_char]).T
SOC_char = np.asarray([SOC_char]).T
SOP_char_flag = np.asarray([SOP_char_flag]).T
SOPresults = np.concatenate((SOC_dis, SOP_dis, SOP_dis_flag, SOC_char, SOP_char, SOP_char_flag), axis=1)
# SOPresults = np.concatenate((SOC_dis, SOP_dis, SOP_dis_flag), axis=1)
np.savetxt("./result/SOPresults.csv", SOPresults)