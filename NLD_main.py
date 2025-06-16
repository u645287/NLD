# -*- coding: utf-8 -*-
import NLD_data
import NLD_model
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from Database import Database, postgres_upsert
# 屏蔽所有警告
warnings.filterwarnings('ignore')

def pred(pred_ym=None, is_upsert=False):
    '''
    從資料庫判斷需預測之期間，預測三個月淨尖離峰落差值
    參數說明：
        pred_ym -- 預設為空，依據資料庫自動預測未來兩個月，也可用 ['2025-05', '2025-06'] 指定預測月份(最多兩個月)
        is_upsert -- 是否將結果更新至資料庫
    '''
    db = Database()
    df = db.get_input()
    # df = df[df['temp_avg_north'].notna()]
    df['YM'] = [datetime(df['year'][i],df['month'][i],1).strftime('%Y-%m') for i in range(len(df))]
    if pred_ym is None:
        pred_ym = df['YM'][df['is_real']==False].reset_index(drop=True)[1:].reset_index(drop=True)[:2]
    samples = 30
    Result = pd.DataFrame(np.full((samples, len(pred_ym)),np.nan), columns=pred_ym)
    Model_tag = [i+1 for i in range(len(pred_ym))]
    Result_db = pd.DataFrame({'year'        :np.repeat([int(pred_ym[i][:4]) for i in range(len(pred_ym))],samples),
                             'month'        :np.repeat([int(pred_ym[i][5:7]) for i in range(len(pred_ym))],samples),
                             'model_name'   :[str(Model_tag[p])+'_'+str(s) for p in range(len(pred_ym)) for s in range(1,samples+1)],
                             'nl_diff_avg'  :np.nan,
                             'exe_time'     :datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    for p in range(len(pred_ym)):
        data = NLD_data.get_input(target_year=int(pred_ym[p][:4]),
                                  target_month=int(pred_ym[p][5:]),
                                  early_month=p+1)
        for r in range(samples):
            model = tf.keras.models.load_model(f'model/{Model_tag[p]}_{r+1}.h5', compile = False)
            value = model.predict(data)[0][0]*10000
            Result_db.loc[p * samples + r, 'nl_diff_avg'] = value
            Result.iloc[r,p] = value
            print(f'{round(100*(p*30+r+1)/(samples*len(pred_ym)))}%')
    stat_Result = pd.DataFrame({'YM'    :pred_ym,
                                'avg'   :Result.mean().values,
                                'sd'    :Result.std().values})
    if is_upsert:
        Result_db.to_sql('ordc_edreg_forecast',db.engine,if_exists='append',index=False,method=postgres_upsert)
    return (stat_Result, Result)
    
def train(pre_period=1, tolerance=3):
    data = NLD_data.get_data(pre_period)
    start_from_epoch = 100
    patience = 100
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=start_from_epoch, patience=patience, restore_best_weights=True)
    model = NLD_model.get_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mape'])
    model.fit(data['X_train'], data['Y_train'], epochs=12000, batch_size=32, validation_data=(data['X_val'], data['Y_val']),callbacks=[callback])
    train_loss = round(np.mean(abs(model.predict(data['X_train']).flatten() - data['Y_train'][0])/data['Y_train'][0])*100,3)
    val_loss = round(np.mean(abs(model.predict(data['X_val']).flatten() - data['Y_val'][0])/data['Y_val'][0])*100,3)
    test_loss = round(np.nanmean(abs(model.predict(data['X_test']).flatten() - data['Y_test'][0])/data['Y_test'][0])*100,3)
    print(f'train: {train_loss}')
    print(f'val: {val_loss}')
    print(f'test: {test_loss}')
    if (train_loss <= tolerance) & (val_loss <= tolerance) & (test_loss <= tolerance):
        print(f'train_loss : {train_loss}, val_loss : {val_loss}, test_loss : {test_loss}')
        print(f'達到訓練終止條件tolerance={tolerance}!')
        model_files = [file for file in os.listdir('model') if f'{pre_period}_' in file]
        i = 1
        while True:
            if str(pre_period)+'_'+str(i)+'.h5' in model_files:
                i += 1
            else:
                filename = str(pre_period)+'_'+str(i)+'.h5'
                print(f'儲存模型為檔案名稱:{filename}')
                print(f"{model.predict(data['X_test'])}")
                break
        model.save(f'model/{filename}')

if __name__ == "__main__":
    result = pred()
    with pd.ExcelWriter("./RESULT.xlsx") as writer:
        result[0].to_excel(writer, sheet_name="執行結果", index=False)
        result[1].to_excel(writer, sheet_name="執行明細", index=False)
