# -*- coding: utf-8 -*-
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.signal as sc

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

#%%

#%% Pruebas con un solo bloque para ver en un principio
subject = "02"
bloque = pd.read_csv("sub-02/ses-A/df_bloque_1_sujeto_02.csv").drop(["Unnamed: 0","Unnamed: 0.1"],axis=1)

primer_estimulo = bloque[bloque["description"] == "video_start"].index[0]
ultimo_estimulo = bloque[bloque["description"] == "video_end"].index[-1]

bloque_final = bloque[primer_estimulo-(15*512):ultimo_estimulo+(15*512)]

print('Extrayendo features con neurokit')
df_bio, info_bio = bio_process_nuevo(ecg=bloque_final['ECG'], rsp=bloque_final['RESP'], eda=bloque_final['GSR'], keep=bloque_final['time'], rsa=False, sampling_rate=512)

# Guardo los df de interés para despues
print(f'Guardando archivos en "sub-{subject}/ses-A"')
df_bio.to_csv(f'sub-{subject}/ses-A/fisio_sujeto_{subject}.csv')



#%%
subjects = ["02","03","04","05","06"]

seleccion = ["EDA_Clean","EDA_Phasic","EDA_Tonic","SCR_Peaks","SCR_Amplitude",
             "ECG_Clean","ECG_Rate", "ECG_R_Peaks",
             "RSP_Clean","RSP_Amplitude","RSP_Rate","RSP_RVT",
             "time"]

for subject in subjects:
    df_fisio = pd.read_csv(f'sub-{subject}/ses-A/fisio_sujeto_{subject}.csv',usecols=seleccion)
    df_markers = pd.read_csv(f'sub-{subject}/ses-A/marcadores_sujeto_{subject}.csv').drop("Unnamed: 0", axis=1)
    
    # Acá intenta mergear los datos y los markers según la columna time y onset (se te van a sumar filas)
    df_markers.rename(columns={"onset": "time"}, inplace=True)
    merged_df = pd.merge(df_fisio, df_markers, on='time', how='outer')
    
    video_dict = {}
    
    # Agarrar linea base
    lb_start = merged_df.loc[merged_df["description"]=="baseline_start","time"]
    lb_end = merged_df.loc[merged_df["description"]=="baseline_end","time"]
    
    video_dict[0] = merged_df[lb_start.index[0]+1:lb_end.index[0]].reset_index()
    
    # Agarrar los tiempos de los videos
    video_start_times = merged_df.loc[merged_df["description"]=="video_start","time"]
    video_end_times = merged_df.loc[merged_df["description"]=="video_end","time"]
    
    # Para cada indice y time en video_start_times
    for i, t in enumerate(video_start_times):
        print(i)
        
        # El slice se guarda con una llave en el diccionario
#        try:
        video_dict[i+1] = merged_df[(merged_df["time"] > video_start_times.iloc[i]) & (merged_df["time"] < video_end_times.iloc[i])].reset_index()
        
        # Por si hay más inicios de video que finales (quizas pasa con el último)
   #     except IndexError:
   #         video_dict[i+1] = merged_df[merged_df["time"] > video_start_times.iloc[i]].reset_index()
    
    # Largo aproximado de cada video
    for n in range(len(video_dict)):
        print(f'Largo video {n}: {len(video_dict[n])}')

    
    # Leo el df_beh y extraigo las columnas que quiero  
    df_beh = pd.read_csv(f'sub-{subject}/ses-A/beh/sub-{subject}_ses-A_task-Experiment_VR_non_immersive_beh.csv')
    
    id_videos = df_beh["id"]
    valence_videos = df_beh["stimulus_type"]
    annotations = [ast.literal_eval(continuous_annotation) for continuous_annotation in df_beh["continuous_annotation"]]
    dimension_annotated = df_beh["dimension"]

    for i, video in enumerate(id_videos):
        df = video_dict[i+1]
        df["time"] = df["time"]-df.loc[0,"time"]
        df["video_id"] = video
        df["stimulus_type"] = list(valence_videos)[i]
        df["dimension_annotated"] = list(dimension_annotated)[i]
        df_annotations = pd.DataFrame(annotations[i],columns=["annotation","time"])
        annotations = pd.merge(df,df_annotations,on="time",how="outer")["annotation"]
        df["annotation"] = annotations.interpolate(method='linear')
        df["subject_id"] = subject
    
    df_final = pd.concat([df for df in video_dict.values()],ignore_index=True)
    df_final.to_csv(f'sub-{subject}/ses-A/df_sub-{subject}_final.csv')

#%%




