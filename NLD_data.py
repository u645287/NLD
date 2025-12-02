# -*- coding: utf-8 -*-
import numpy as np
from Database import Database
from datetime import datetime, timedelta

def get_input(target_year, target_month, early_month):
    db = Database()
    df = db.get_input()
    
    df['ym'] = [datetime(df['year'][i],df['month'][i],1) for i in range(len(df))]
    
    start_ym = (datetime(target_year, target_month, 1) - timedelta(days=(early_month+12)*30)).replace(day=1)
    end_ym = (start_ym + timedelta(days=11*31+6)).replace(day=1)
    
    ref_index = (df['ym']>=start_ym) & (df['ym']<=end_ym)
    fcst_index = (df['year']==target_year) & (df['month']==target_month)
    
    nl_diff_r = df[ref_index]['nl_diff_avg'].values/10000
    pv_cap_r = df[ref_index]['cap_pv'].values/20000
    wind_cap_r = df[ref_index]['cap_wind'].values/7000
    tx_n_r = (df[ref_index]['temp_avg_north'].values-15)/25
    tx_c_r = (df[ref_index]['temp_avg_central'].values-15)/25
    tx_s_r = (df[ref_index]['temp_avg_south'].values-15)/25
    load_r = (df[ref_index]['load_avg'].values-20000)/20000
    year_f = np.array([target_year])/2030
    month_f = np.array([target_month])-1
    pv_cap_f = df[fcst_index]['cap_pv'].values/20000
    wind_cap_f = df[fcst_index]['cap_wind'].values/7000
    tx_n_f = (df[fcst_index]['temp_avg_north'].values-15)/25
    tx_c_f = (df[fcst_index]['temp_avg_central'].values-15)/25
    tx_s_f = (df[fcst_index]['temp_avg_south'].values-15)/25
    load_f = (df[fcst_index]['load_avg'].values-20000)/20000
    return [np.reshape(nl_diff_r, (1,12)),
            np.reshape(pv_cap_r, (1,12)),
            np.reshape(wind_cap_r, (1,12)),
            np.reshape(tx_n_r, (1,12)),
            np.reshape(tx_c_r, (1,12)),
            np.reshape(tx_s_r, (1,12)),
            np.reshape(load_r, (1,12)),
            np.reshape(year_f, (1,1)),
            np.reshape(month_f, (1,1)),
            np.reshape(pv_cap_f, (1,1)),
            np.reshape(wind_cap_f, (1,1)),
            np.reshape(tx_n_f, (1,1)),
            np.reshape(tx_c_f, (1,1)),
            np.reshape(tx_s_f, (1,1)),
            np.reshape(load_f, (1,1))]

def get_data(future_point, refer_length=12, val_num=6):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = data_split(future_point, refer_length, val_num)
    X_train_fit, Y_train_fit = get_XY(X_train, Y_train)
    X_val_fit, Y_val_fit = get_XY(X_val, Y_val)
    X_test_fit, Y_test_fit = get_XY(X_test, Y_test)
    return {'X_train' : X_train_fit,
            'Y_train' : Y_train_fit,
            'X_val'   : X_val_fit,
            'Y_val'   : Y_val_fit,
            'X_test'  : X_test_fit,
            'Y_test'  : Y_test_fit}

#%% ------資料切片(data discretization)-----------
def data_split(future_point, refer_length=12, val_num=6):
    db = Database()
    df = db.get_input()
    df = df[df['temp_avg_north'].notna()]
    
    test_num = sum(df['is_real']==False)         
    train_num = len(df) - test_num - val_num - future_point - (1 + refer_length)


    Year = df['year'].values/2030
    Month = df['month'].values-1
    NetPeakDiff = df['nl_diff_avg'].values/10000
    PV_Cap = df['cap_pv'].values/20000
    Wind_Cap = df['cap_wind'].values/7000
    Temp_avg_N = (df['temp_avg_north'].values-15)/(25)
    Temp_avg_C = (df['temp_avg_central'].values-15)/(25)
    Temp_avg_S = (df['temp_avg_south'].values-15)/(25)
    Load = (df['load_avg'].values-20000)/20000
    
    x_NetPeakDiff_r = []
    x_PV_Cap_r = []
    x_Wind_Cap_r = []
    x_Temp_avg_N_r = []
    x_Temp_avg_C_r = []
    x_Temp_avg_S_r = []
    x_Load_r = []
    
    x_Year_f = []
    x_Month_f = []
    x_PV_Cap_f = []
    x_Wind_Cap_f = []
    x_Temp_avg_N_f = []
    x_Temp_avg_C_f = []
    x_Temp_avg_S_f = []
    x_Load_f = []
    y = []
    #概念是從最後的資料拿回來，
    #             總長度 - 測試所需長度 - 驗證所需長度 - 訓練所需長度 - 訓練資料的第一筆數據所需的前期資料長度(12個月)
    subset_index = len(Year) - future_point - train_num - val_num - test_num - (1 + refer_length)
    Year_subset = Year[subset_index:]
    Month_subset = Month[subset_index:]
    NetPeakDiff_subset = NetPeakDiff[subset_index:]
    PV_Cap_subset = PV_Cap[subset_index:]
    Wind_Cap_subset = Wind_Cap[subset_index:]
    Temp_avg_N_subset = Temp_avg_N[subset_index:]
    Temp_avg_C_subset = Temp_avg_C[subset_index:]
    Temp_avg_S_subset = Temp_avg_S[subset_index:]
    Load_subset =  Load[subset_index:]
    i = 13
    while i + future_point < len(Year_subset):
        x_NetPeakDiff_r.append(NetPeakDiff_subset[i-1-refer_length:i-1])
        x_PV_Cap_r.append(PV_Cap_subset[i-1-refer_length:i-1])
        x_Wind_Cap_r.append(Wind_Cap_subset[i-1-refer_length:i-1])
        x_Temp_avg_N_r.append(Temp_avg_N_subset[i-1-refer_length:i-1])
        x_Temp_avg_C_r.append(Temp_avg_C_subset[i-1-refer_length:i-1])
        x_Temp_avg_S_r.append(Temp_avg_S_subset[i-1-refer_length:i-1])
        x_Load_r.append(Load_subset[i-1-refer_length:i-1])
        x_Year_f.append(Year_subset[i + future_point-1])
        x_Month_f.append(Month_subset[i + future_point-1])
        x_PV_Cap_f.append(PV_Cap_subset[i + future_point-1])
        x_Wind_Cap_f.append(Wind_Cap_subset[i + future_point-1])
        x_Temp_avg_N_f.append(Temp_avg_N_subset[i + future_point-1])
        x_Temp_avg_C_f.append(Temp_avg_C_subset[i + future_point-1])
        x_Temp_avg_S_f.append(Temp_avg_S_subset[i + future_point-1])
        x_Load_f.append(Load_subset[i + future_point-1])
        y.append(NetPeakDiff_subset[i + future_point-1])
        i += 1
        
    X_NetPeakDiff_r = np.array(x_NetPeakDiff_r)
    X_PV_Cap_r = np.array(x_PV_Cap_r)
    X_Wind_Cap_r = np.array(x_Wind_Cap_r)
    X_Temp_avg_N_r = np.array(x_Temp_avg_N_r)
    X_Temp_avg_C_r = np.array(x_Temp_avg_C_r)
    X_Temp_avg_S_r = np.array(x_Temp_avg_S_r)
    X_Load_r = np.array(x_Load_r)
    X_Year_f = np.array(x_Year_f)
    X_Month_f = np.array(x_Month_f)
    X_PV_Cap_f = np.array(x_PV_Cap_f)
    X_Wind_Cap_f = np.array(x_Wind_Cap_f)
    X_Temp_avg_N_f = np.array(x_Temp_avg_N_f)
    X_Temp_avg_C_f = np.array(x_Temp_avg_C_f)
    X_Temp_avg_S_f = np.array(x_Temp_avg_S_f)
    X_Load_f = np.array(x_Load_f)
    Y = np.array(y)
    
    train_index = train_num
    val_index = (train_num + val_num)
    test_index = (train_num + val_num + test_num)
    X_train = []
    X_val = []
    X_test = []
    Y_train = []
    Y_val = []
    Y_test = []
    X_train.append([X_NetPeakDiff_r[:train_index,:],
                    X_PV_Cap_r[:train_index,:], X_Wind_Cap_r[:train_index,:],
                    X_Temp_avg_N_r[:train_index,:], X_Temp_avg_C_r[:train_index,:], X_Temp_avg_S_r[:train_index,:],
                    X_Load_r[:train_index,:],
                    X_Year_f[:train_index], X_Month_f[:train_index],
                    X_PV_Cap_f[:train_index], X_Wind_Cap_f[:train_index], 
                    X_Temp_avg_N_f[:train_index], X_Temp_avg_C_f[:train_index], X_Temp_avg_S_f[:train_index],
                    X_Load_f[:train_index]])
    X_val.append([X_NetPeakDiff_r[train_index:val_index,:],
                    X_PV_Cap_r[train_index:val_index,:], X_Wind_Cap_r[train_index:val_index,:],
                    X_Temp_avg_N_r[train_index:val_index,:], X_Temp_avg_C_r[train_index:val_index,:], X_Temp_avg_S_r[train_index:val_index,:],
                    X_Load_r[train_index:val_index,:],
                    X_Year_f[train_index:val_index], X_Month_f[train_index:val_index],
                    X_PV_Cap_f[train_index:val_index], X_Wind_Cap_f[train_index:val_index], 
                    X_Temp_avg_N_f[train_index:val_index], X_Temp_avg_C_f[train_index:val_index], X_Temp_avg_S_f[train_index:val_index],
                    X_Load_f[train_index:val_index]])
    X_test.append([X_NetPeakDiff_r[val_index:test_index,:],
                    X_PV_Cap_r[val_index:test_index,:], X_Wind_Cap_r[val_index:test_index,:],
                    X_Temp_avg_N_r[val_index:test_index,:], X_Temp_avg_C_r[val_index:test_index,:], X_Temp_avg_S_r[val_index:test_index,:],
                    X_Load_r[val_index:test_index,:],
                    X_Year_f[val_index:test_index], X_Month_f[val_index:test_index],
                    X_PV_Cap_f[val_index:test_index], X_Wind_Cap_f[val_index:test_index], 
                    X_Temp_avg_N_f[val_index:test_index], X_Temp_avg_C_f[val_index:test_index], X_Temp_avg_S_f[val_index:test_index],
                    X_Load_f[val_index:test_index]])
    Y_train.append(Y[:train_index])
    Y_val.append(Y[train_index:val_index])
    Y_test.append(Y[val_index:test_index])
    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

#%% ------模型-資料最終整理------------------------
def get_XY(X, Y):
    X_new = []
    Y_new = []
    X_new.append(X[0][0].reshape((X[0][0].shape[0],X[0][0].shape[1],))) # NetPeakDiff_r
    X_new.append(X[0][1].reshape((X[0][1].shape[0],X[0][1].shape[1],))) # PV_Cap_r
    X_new.append(X[0][2].reshape((X[0][2].shape[0],X[0][2].shape[1],))) # Wind_Cap_r
    X_new.append(X[0][3].reshape((X[0][3].shape[0],X[0][3].shape[1],))) # Temp_avg_N_r
    X_new.append(X[0][4].reshape((X[0][4].shape[0],X[0][4].shape[1],))) # Temp_avg_C_r
    X_new.append(X[0][5].reshape((X[0][5].shape[0],X[0][5].shape[1],))) # Temp_avg_S_r
    X_new.append(X[0][6].reshape((X[0][6].shape[0],X[0][6].shape[1],))) # Load_r
    X_new.append(X[0][7]) # Year_f
    X_new.append(X[0][8]) # Month_f
    X_new.append(X[0][9]) # PV_Cap_f
    X_new.append(X[0][10]) # Wind_Cap_f
    X_new.append(X[0][11]) # Temp_avg_N_f
    X_new.append(X[0][12]) # Temp_avg_C_f
    X_new.append(X[0][13]) # Temp_avg_S_f
    X_new.append(X[0][14]) # Load_f
    Y_new.append(Y[0])
    return (X_new, Y_new)