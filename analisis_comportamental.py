# -*- coding: utf-8 -*-
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

#%%
subjects = ["02", "03", "04", "05", "06"]#

seleccion = ["EDA_Clean", "EDA_Phasic", "EDA_Tonic", "SCR_Peaks", "SCR_Amplitude", "SMNA",
             "ECG_Clean", "ECG_Rate", "ECG_R_Peaks", "HRV",
             "RSP_Clean", "RSP_Amplitude", "RSP_Rate", "RSP_RVT",
             "description", "time"]

for subject in subjects:
    # Ya lo leo, porque corresponde al sujeto y asi me queda para despues
    df_beh = pd.read_csv(f'sub-{subject}/ses-A/beh/sub-{subject}_ses-A_task-Experiment_VR_non_immersive_beh.csv')

    for bloque_n in range(1,9): # Por ahora no hubo sujetos que no tuvieran 8 bloques
        if subject == "05" and bloque_n == 3: # no se por qué pero este crashea
            continue

        print(f'Procesando sujeto {subject} - bloque {bloque_n}')
        bloque_fisio = pd.read_csv(f'sub-{subject}/ses-A/fisio_bloque_{bloque_n}_sujeto_{subject}.csv').drop("Unnamed: 0",axis=1)
        
        indices_video_start = bloque_fisio[bloque_fisio["description"] == "video_start"].index
        indices_video_end = bloque_fisio[bloque_fisio["description"] == "video_end"].index

        primer_estimulo = indices_video_start[0]
        ultimo_estimulo = indices_video_end[-1]

        # Agarrar linea base del sujeto / Esto va a haber que cambiarlo porque no lo tuve en cuenta anteeeeeeeeeeeeeees
        """
        lb_start = merged_df.loc[merged_df["description"]=="baseline_start","time"]
        lb_end = merged_df.loc[merged_df["description"]=="baseline_end","time"]

        video_dict[0] = merged_df[lb_start.index[0]+1:lb_end.index[0]].reset_index()
        """

        # Capaz no haga falta sacarlo así, porque se podría sacar directamente del df fisio haciendo el mismo corte
        """
        # Lineas base de los videos (15 segundos antes del inicio del video)
        i_start_lb_videos = [(i-(15*512)) for i in indices_video_start]
        i_end_lb_videos = indices_video_start

        lb_dict = {}
        # Guardar las lineas base
        for i in indices_video_start:
            lb_dict[i+1] = bloque[i_start_lb_videos[i]:i_end_lb_videos[i]]
        """

        video_dict = {}

        # Para cada indice y time en video_start_times
        for i, indice_video in enumerate(indices_video_start):
            print(i)
            
            # El slice se guarda con una llave en el diccionario
    #        try:
            video_dict[i+1] = bloque_fisio[indices_video_start[i]:indices_video_end[i]].reset_index()
            
            # Por si hay más inicios de video que finales (quizas pasa con el último)
    #     except IndexError:
    #         video_dict[i+1] = merged_df[merged_df["time"] > video_start_times.iloc[i]].reset_index()
        
        # Largo aproximado de cada video
        for n in range(len(video_dict)):
            print(f'Largo video {n+1}: {video_dict[n+1].time.max()}')

      
        # Agarro los videos correspondientes a este bloque
        bloque_beh = df_beh[df_beh["subbloque_number"]==float(bloque_n)]

        # Features que quiero agarrar del bloque comportamental
        id_videos = bloque_beh["id"]
        valence_type = bloque_beh["stimulus_type"]
        annotations = [ast.literal_eval(continuous_annotation) for continuous_annotation in bloque_beh["continuous_annotation"]]
        dimension_annotated = bloque_beh["dimension"]

        for i, video in enumerate(id_videos):
            df = video_dict[i+1]
            df["time"] = df["time"]-df.loc[0,"time"]
            df["video_id"] = video
            df["stimulus_type"] = list(valence_type)[i]
            df["dimension_annotated"] = list(dimension_annotated)[i]
            df_annotations = pd.DataFrame(annotations[i],columns=["annotation","time"])
            merged_annotations = pd.merge(df,df_annotations,on="time",how="outer")["annotation"]
            df["annotation"] = merged_annotations.interpolate(method='linear')
            df["subject_id"] = subject
        
        df_final = pd.concat([df for df in video_dict.values()],ignore_index=True)
        print(f'Guardando datos en df_sub-{subject}_block_{bloque_n}_final.csv')
        df_final.to_csv(f'sub-{subject}/ses-A/df_sub-{subject}_block_{bloque_n}_final.csv')

#%%




