# -*- coding: utf-8 -*-
from keras.layers import Input, BatchNormalization, Dense, concatenate, Embedding, Flatten, add, average, LayerNormalization, Dropout, Permute, Reshape, RepeatVector
from tensorflow.keras.regularizers import l1_l2
from keras.models import Model

#%% ------模型-宣告----------------
def get_model():
    # 宣告輸入層
    input_NPD_r = BatchNormalization()(Input(shape=(12,)))
    input_PV_Cap_r = BatchNormalization()(Input(shape=(12,)))
    input_Wind_Cap_r = BatchNormalization()(Input(shape=(12,)))
    input_Temp_N_r = BatchNormalization()(Input(shape=(12,)))
    input_Temp_C_r = BatchNormalization()(Input(shape=(12,)))
    input_Temp_S_r = BatchNormalization()(Input(shape=(12,)))
    input_Load_r = BatchNormalization()(Input(shape=(12,)))
    input_Year_f = BatchNormalization()(Input(shape=(1,)))
    input_Month_f = BatchNormalization()(Input(shape=(1,)))
    input_PV_Cap_f = BatchNormalization()(Input(shape=(1,)))
    input_Wind_Cap_f = BatchNormalization()(Input(shape=(1,)))
    input_Temp_N_f = BatchNormalization()(Input(shape=(1,)))
    input_Temp_C_f = BatchNormalization()(Input(shape=(1,)))
    input_Temp_S_f = BatchNormalization()(Input(shape=(1,)))
    input_Load_f = BatchNormalization()(Input(shape=(1,)))
    # 基礎結構
    output_pre = get_basic_structure(input_NPD_r,
                                     input_PV_Cap_r, input_Wind_Cap_r,
                                     input_Temp_N_r, input_Temp_C_r, input_Temp_S_r,
                                     input_Load_r,
                                     input_Year_f, input_Month_f,
                                     input_PV_Cap_f, input_Wind_Cap_f,
                                     input_Temp_N_f, input_Temp_C_f, input_Temp_S_f,
                                     input_Load_f)
    # 殘差層
    input_1 = output_pre
    input_2 = output_pre
    output_list = [output_pre]

    num_resnetplus_layer = 10

    for i in range(num_resnetplus_layer):
        output_res_ave, output_list = resnetplus_layer(input_1, input_2, output_list)
        input_1 = output_res_ave
        if i == 0:
            input_2 = output_res_ave

    output = output_res_ave
    # 組合
    model = Model(inputs=[input_NPD_r,
                          input_PV_Cap_r, input_Wind_Cap_r,
                          input_Temp_N_r, input_Temp_C_r, input_Temp_S_r,
                          input_Load_r,
                          input_Year_f, input_Month_f,
                          input_PV_Cap_f, input_Wind_Cap_f,
                          input_Temp_N_f, input_Temp_C_f, input_Temp_S_f,
                          input_Load_f],
                  outputs=[output])
    return model

#%% ------模型-基礎結構(Basic Structure)----------------
def get_basic_structure(input_NPD_r,
                        input_PV_Cap_r, input_Wind_Cap_r,
                        input_Temp_N_r, input_Temp_C_r, input_Temp_S_r,
                        input_Load_r,
                        input_Year_f, input_Month_f,
                        input_PV_Cap_f, input_Wind_Cap_f,
                        input_Temp_N_f, input_Temp_C_f, input_Temp_S_f,
                        input_Load_f):
    num_dense = 10
    l1 = 0
    l2 = 0
    # 過去資訊
    dense_NPD_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_NPD_r)
    dense_PV_Cap_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_PV_Cap_r)
    dense_Wind_Cap_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Wind_Cap_r)
    
    dense_Temp_N_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Temp_N_r)
    dense_Temp_C_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Temp_C_r)
    dense_Temp_S_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Temp_S_r)
    concat_Temp_r = concatenate([dense_Temp_N_r, dense_Temp_C_r, dense_Temp_S_r])
    dense_Temp_r = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(concat_Temp_r)
    
    dense_Load_r = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Load_r)
    
    concat_r = concatenate([dense_NPD_r, dense_PV_Cap_r, dense_Wind_Cap_r, dense_Temp_r, dense_Load_r])
    dense_r = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(concat_r)
    # 未來資訊
    embed_Month_f = Embedding(input_dim=12, output_dim=6, input_length=1)(input_Month_f)
    dense_Month_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(Flatten()(embed_Month_f))
    dense_PV_Cap_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_PV_Cap_f)
    dense_Wind_Cap_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Wind_Cap_f)
    
    dense_Temp_N_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Temp_N_f)
    dense_Temp_C_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Temp_C_f)
    dense_Temp_S_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Temp_S_f)
    concat_Temp_f = concatenate([dense_Temp_N_f, dense_Temp_C_f, dense_Temp_S_f])
    dense_Temp_f = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(concat_Temp_f)
    
    dense_Load_f = Dense(num_dense, activation='silu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(input_Load_f)
    # 合併
    concat_f = concatenate([dense_PV_Cap_f, dense_Wind_Cap_f, dense_Temp_f, dense_Load_f])
    dense_f = Dense(num_dense, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(concat_f)
    # FC
    FC = concatenate([Reshape((num_dense,1))(dense_r),
                      Reshape((num_dense,1))(dense_f),
                      Reshape((num_dense,1))(dense_Month_f),  RepeatVector(num_dense)(input_Year_f)])

    pre_output = Dense(num_dense, activation='silu', kernel_regularizer=l1_l2(l1=l1, l2=l2))(Flatten()(FC))
    output = Dense(1)(pre_output)
    return output

#%% ------模型-殘差層(ResNet Layer)----------------
def get_res_layer(output, last=False):
    dense_res11 = Dense(2, activation='silu', kernel_initializer='lecun_normal')(output)
    dense_res12 = Dense(1, activation='linear', kernel_initializer='lecun_normal')(dense_res11)
    
    dense_res21 = Dense(2, activation='silu', kernel_initializer='lecun_normal')(output)
    dense_res22 = Dense(1, activation='linear', kernel_initializer='lecun_normal')(dense_res21)
    
    dense_res31 = Dense(2, activation='silu', kernel_initializer='lecun_normal')(output)
    dense_res32 = Dense(1, activation='linear', kernel_initializer='lecun_normal')(dense_res31)
    
    dense_res41 = Dense(2, activation='silu', kernel_initializer='lecun_normal')(output)
    dense_res42 = Dense(1, activation='linear', kernel_initializer='lecun_normal')(dense_res41)
    
    dense_add = add([dense_res12, dense_res22, dense_res32, dense_res42])
    
    if last:
        output_new = add([dense_add, output], name='output')
    else:
        output_new = add([dense_add, output])
    return output_new

#%% ------模型-均化殘差層(ResNetPlus Layer)--------
def resnetplus_layer(input_1, input_2, output_list):
    output_res = get_res_layer(input_1)
    output_res_ = get_res_layer(input_2)
    output_res_ave_mid = average([output_res, output_res_])
    output_list.append(output_res_ave_mid)
    output_res_ave = average(output_list)
    return output_res_ave, output_list
