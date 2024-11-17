# -*- coding: utf-8 -*-
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

#%% Junto los datos en 1 df para cada sujeto

subjects = ["02","03","04", "05","06"] #

dict_videos_sujetos = {}
for subject in subjects:
    df_videos_sujeto = pd.DataFrame()
    for bloque_n in range(1,9): # Por ahora no hubo sujetos que no tuvieran 8 bloques
        if subject == "05" and bloque_n == 3: # no tengo el fisio de este bloque porque crashea
            continue
        
        bloque = pd.read_csv(f'sub-{subject}/ses-A/df_sub-{subject}_block_{bloque_n}_final.csv').drop("Unnamed: 0",axis=1)
        
        if bloque_n > 4:
            bloque["video_rep"] = 2
        
        else:
            bloque["video_rep"] = 1
        
        
        df_videos_sujeto = pd.concat([df_videos_sujeto, bloque], axis=0)
    
    dict_videos_sujetos[subject] = df_videos_sujeto
    
#%% Ver señal x de cada pasada de estímulo por sujeto

señal_objetivo = "EDA_Clean"

for subject in subjects:
    df_videos_sujeto = dict_videos_sujetos[subject]
    
    for video in df_videos_sujeto.video_id.unique():
        data_video = df_videos_sujeto[df_videos_sujeto.video_id == video]
        primera_pasada = data_video[data_video["video_rep"]==1]
        segunda_pasada = data_video[data_video["video_rep"]==2]
    
        plt.figure()
        plt.plot(primera_pasada.time,stats.zscore(primera_pasada[f'{señal_objetivo}']),color="red",label="Primer pasada")
        plt.plot(segunda_pasada.time,stats.zscore(segunda_pasada[f'{señal_objetivo}']),color="blue",label="Segunda pasada")
        plt.title(f"{señal_objetivo} Subject {subject} - Video {video}")
        plt.legend()
        plt.show()
        
# %% Comparar anotaciones entre primer y segunda pasada

for subject in subjects:
    df_videos_sujeto = dict_videos_sujetos[subject]
    
    for video in df_videos_sujeto.video_id.unique():
        data_video = df_videos_sujeto[df_videos_sujeto.video_id == video]
        arousal_annotated = data_video[data_video["dimension_annotated"]=="arousal"]
        valence_annotated = data_video[data_video["dimension_annotated"]=="valence"]

    
        plt.figure()
        plt.plot(arousal_annotated.time,arousal_annotated.annotation,color="red",label="Arousal")
        plt.plot(valence_annotated.time,valence_annotated.annotation,color="blue",label="Valencia")
        plt.title(f"Anotaciones Subject {subject} - Video {video}")
        plt.legend()
        plt.show()
        
# %% Ver señal X de cada sujeto por estimulo
señal_objetivo = "EDA_Tonic"

sujetos_por_video = {}

for video_id in range(1,15):   
    primeras_pasadas = pd.DataFrame()
    segundas_pasadas = pd.DataFrame()
    
    for subject in subjects:
        df_videos_sujeto = dict_videos_sujetos[subject]
        
        data_video = df_videos_sujeto[df_videos_sujeto.video_id == video_id]
        primera_pasada = data_video[data_video["video_rep"]==1]
        segunda_pasada = data_video[data_video["video_rep"]==2]
        
        primeras_pasadas = pd.concat([primeras_pasadas,primera_pasada],axis=0)
        segundas_pasadas = pd.concat([segundas_pasadas,segunda_pasada],axis=0)        
        
    sujetos_por_video[f'video_{video_id}_primer_rep'] = primeras_pasadas
    sujetos_por_video[f'video_{video_id}_segundo_rep'] = segundas_pasadas
    

for video in sujetos_por_video.keys():
    for subject in range(2,7):
        df_rep = sujetos_por_video[f'{video}']
        señal = df_rep[df_rep["subject_id"]==subject]
        
        plt.figure(1)
        plt.plot(señal.time,stats.zscore(señal[f'{señal_objetivo}']),label=f"Subject {subject}")
        plt.title(f'{señal_objetivo} - {video}')
        plt.legend()
    plt.show()

# %% Ver anotacion X de cada sujeto por estimulo
sujetos_por_video = {}

for video_id in range(1,15):   
    pasadas_arousal = pd.DataFrame()
    pasadas_valencia = pd.DataFrame()
    
    for subject in subjects:
        df_videos_sujeto = dict_videos_sujetos[subject]
        
        data_video = df_videos_sujeto[df_videos_sujeto.video_id == video_id]
        pasada_arousal = data_video[data_video["dimension_annotated"]=="arousal"]
        pasada_valencia = data_video[data_video["dimension_annotated"]=="valence"]
        
        pasadas_arousal = pd.concat([pasadas_arousal,pasada_arousal],axis=0)
        pasadas_valencia = pd.concat([pasadas_valencia,pasada_valencia],axis=0)
        
    sujetos_por_video[f'video_{video_id}_arousal'] = pasadas_arousal
    sujetos_por_video[f'video_{video_id}_valence'] = pasadas_valencia
    

for video in sujetos_por_video.keys():
    for subject in range(2,7):
        df_anotacion = sujetos_por_video[f'{video}']
        señal = df_anotacion[df_anotacion["subject_id"]==subject]
        
        plt.figure(1)
        plt.plot(señal.time,señal.annotation,label=f"Subject {subject}")
        plt.title(f'{video}')
        plt.legend()
    plt.show()
    
# %%
