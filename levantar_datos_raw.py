# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import scipy.signal as sc
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)


#%% 
subjects = ["02","03","04","05","06"]

for subject in subjects:
    fname = f'sub-{subject}/ses-A/eeg/sub-{subject}_ses-a_task-experiment_vr_non_immersive_eeg.fif'

    print(f'Leyendo archivo raw de {subject}')
    raw = mne.io.read_raw(fname, verbose=True, preload = True)
    
    print('Cargando datos raw')
    data = raw.load_data()    
    markers = raw.annotations
    
    conteo_anotaciones = markers.count() # --> acá cuento directo del archivo los markers, así se de primera mano cuantos hay
    
    df_data = data.to_data_frame(time_format=None)
    df_markers = markers.to_data_frame(time_format="ms")
    df_markers["onset"] = df_markers["onset"]/1000 # Para que quede en segundos
    
    print(df_data.columns)
    print(df_markers.columns)
    
    print('Guardando datos raw')
    df_markers.rename(columns={"onset": "time"}, inplace=True)
    merged_df = pd.merge(df_data, df_markers, on='time', how='outer')
    merged_df.to_csv(f'sub-{subject}/ses-A/data_raw_sujeto_{subject}.csv')

    # Segmentar por bloques
    tiempos_block_start = merged_df[merged_df["description"]=="BLOCK_START"].time
    tiempos_block_end = merged_df[merged_df["description"]=="BLOCK_END"].time
    tiempos_merged = pd.concat([tiempos_block_start.reset_index(), tiempos_block_end.reset_index()],axis=1)
    
    
    for i in range(len(tiempos_block_start)):
        bloque = merged_df.iloc[tiempos_block_start.index[i]-(25*512):tiempos_block_end.index[i]+(15*512)]
        
        bloque.to_csv(f'sub-{subject}/ses-A/df_bloque_{i+1}_sujeto_{subject}.csv')
        
    
    
    # Esto es para saber la diferencia en secs entre inicio del bloque y primer video
#    print(bloque[bloque["description"]=="video_start"].time.iloc[0]-bloque.time.iloc[0])
    # Esto es para saber la diferencia en secs entre ultimo video y final de bloque
#    print(bloque.time.iloc[-1]-bloque[bloque["description"]=="video_end"].time.iloc[-1])
#    print("\n")
    
    #Debería agregar algo para agarrar hasta 15 segundos antes del primer
    # video y hasta 15 segundos despues del último

