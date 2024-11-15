# -*- coding: utf-8 -*-
#%%
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
import cvxopt as cv

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

#%%

def ecg_process_nuevo(ecg_signal, sampling_rate=1000, method="neurokit"):
    """
    Modificación para que el método de ecg_clean sea biosppy
    Cambio el default de ecg_clean a biosppy (acá abajo)
    """

    # Sanitize and clean input
    ecg_signal = nk.signal_sanitize(ecg_signal)
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="biosppy")

    # Detect R-peaks
    instant_peaks, info = nk.ecg_peaks(
        ecg_cleaned=ecg_cleaned,
        sampling_rate=sampling_rate,
        method=method,
        correct_artifacts=True,
    )

    # Calculate heart rate
    rate = nk.signal_rate(
        info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )

    # Assess signal quality
    quality = nk.ecg_quality(
        ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    # Merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Raw": ecg_signal,
            "ECG_Clean": ecg_cleaned,
            "ECG_Rate": rate,
            "ECG_Quality": quality,
        }
    )

    # Delineate QRS complex
    delineate_signal, delineate_info = nk.ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  # Merge waves indices dict with info dict

    # Determine cardiac phases
    cardiac_phase = nk.ecg_phase(
        ecg_cleaned=ecg_cleaned,
        rpeaks=info["ECG_R_Peaks"],
        delineate_info=delineate_info,
    )

    # Add additional information to signals DataFrame
    signals = pd.concat(
        [signals, instant_peaks, delineate_signal, cardiac_phase], axis=1
    )

    # return signals DataFrame and R-peak locations
    return signals, info

def eda_methods_nuevo(
    sampling_rate=1000,
    method="default",
    method_cleaning="default",
    method_peaks="default",
    method_phasic="cvxeda",
    **kwargs,
):
    """
    Modificación para que el método de EDA_Tonic sea cvxOPT
    Cambio el default de method_phasic a cvxeda (ahí arriba)
    """
    
    # Sanitize inputs
    method_cleaning = str(method).lower() if method_cleaning == "default" else str(method_cleaning).lower()
    method_phasic = str(method).lower() if method_phasic == "default" else str(method_phasic).lower()
    method_peaks = str(method).lower() if method_peaks == "default" else str(method_peaks).lower()

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method_cleaning": method_cleaning,
        "method_phasic": method_phasic,
        "method_peaks": method_peaks,
        "kwargs": kwargs,
    }

    # Get arguments to be passed to underlying functions
    kwargs_cleaning, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_clean)
    kwargs_phasic, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_phasic)
    kwargs_peaks, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_peaks)

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_phasic"] = kwargs_phasic
    report_info["kwargs_peaks"] = kwargs_peaks

    # Initialize refs list
    refs = []

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning == "biosppy":
        report_info["text_cleaning"] += " was cleaned using the biosppy package."
    elif method_cleaning in ["default", "neurokit", "nk"]:
        report_info["text_cleaning"] += " was cleaned using the default method of the neurokit2 package."
    elif method_cleaning in ["none"]:
        report_info["text_cleaning"] += "was directly used without cleaning."
    else:
        report_info["text_cleaning"] += " was cleaned using the method described in " + method_cleaning + "."

    # 2. Phasic decomposition
    # -----------------------
    report_info["text_phasic"] = "The signal was decomposed into phasic and tonic components using"
    if method_phasic is None or method_phasic in ["none"]:
        report_info["text_phasic"] = "There was no phasic decomposition carried out."
    else:
        report_info["text_phasic"] += " the method described in " + method_phasic + "."

    # 3. Peak detection
    # -----------------
    report_info["text_peaks"] = "The cleaned signal was used to detect peaks using"
    if method_peaks in ["gamboa2008", "gamboa"]:
        report_info["text_peaks"] += " the method described in Gamboa et al. (2008)."
        refs.append("""Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci
        and electrophysiology. PhD ThesisUniversidade.""")
    elif method_peaks in ["kim", "kbk", "kim2004", "biosppy"]:
        report_info["text_peaks"] += " the method described in Kim et al. (2004)."
        refs.append("""Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term
      monitoring of physiological signals. Medical and biological engineering and computing, 42(3),
      419-427.""")
    elif method_peaks in ["nk", "nk2", "neurokit", "neurokit2"]:
        report_info["text_peaks"] += " the default method of the `neurokit2` package."
        refs.append("https://doi.org/10.21105/joss.01667")
    elif method_peaks in ["vanhalem2020", "vanhalem", "halem2020"]:
        report_info["text_peaks"] += " the method described in Vanhalem et al. (2020)."
        refs.append("""van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.""")
    elif method_peaks in ["nabian2018", "nabian"]:
        report_info["text_peaks"] += " the method described in Nabian et al. (2018)."
        refs.append("""Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.""")
    else:
        report_info[
            "text_peaks"
        ] = f"The peak detection was carried out using the method {method_peaks}."

    # References
    report_info["references"] = list(np.unique(refs))
    return report_info

def eda_process_nuevo(
    eda_signal, sampling_rate=1000, method="neurokit", report=None, **kwargs
):
    """
    Modificación para que el método de EDA_Tonic sea cvxOPT
    Cambio eda_methods() por eda_methods_nuevo()

    """
    # Sanitize input
    eda_signal = nk.signal_sanitize(eda_signal)
    methods = eda_methods_nuevo(sampling_rate=sampling_rate, method=method, **kwargs)

    # Preprocess
    # Clean signal
    eda_cleaned = nk.eda_clean(
        eda_signal,
        sampling_rate=sampling_rate,
        method=methods["method_cleaning"],
        **methods["kwargs_cleaning"],
    )
    if methods["method_phasic"] is None or methods["method_phasic"].lower() == "none":
        eda_decomposed = pd.DataFrame({"EDA_Phasic": eda_cleaned})
    else:
        eda_decomposed = nk.eda_phasic(
            eda_cleaned,
            sampling_rate=sampling_rate,
            method=methods["method_phasic"],
            **methods["kwargs_phasic"],
        )

    # Find peaks
    peak_signal, info = nk.eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=sampling_rate,
        method=methods["method_peaks"],
        amplitude_min=0.1,
        **methods["kwargs_peaks"],
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    if report is not None:
        # Generate report containing description and figures of processing
        if ".html" in str(report):
            fig = nk.eda_plot(signals, info, static=False)
        else:
            fig = None
        nk.create_report(file=report, signals=signals, info=methods, fig=fig)

    return signals, info


def bio_process_nuevo(
    ecg=None,
    rsp=None,
    eda=None,
    emg=None,
    ppg=None,
    eog=None,
    keep=None,
    rsa=False,
    sampling_rate=1000,
):
    """
    Modificación para que el método de EDA_Tonic sea cvxOPT
    Cambio eda_process() por eda_process_nuevo()
    Agrego parámetro rsa=False, si se cambia a True y hay señal ecg y rsp se analiza, sino no.

    """
    bio_info = {}
    bio_df = pd.DataFrame({})

    # Error check if first argument is a Dataframe.
    if ecg is not None:
        if isinstance(ecg, pd.DataFrame):
            data = ecg.copy()
            if "RSP" in data.keys():
                rsp = data["RSP"]
            else:
                rsp = None
            if "EDA" in data.keys():
                eda = data["EDA"]
            else:
                eda = None
            if "EMG" in data.keys():
                emg = data["EMG"]
            else:
                emg = None
            if "ECG" in data.keys():
                ecg = data["ECG"]
            elif "EKG" in data.keys():
                ecg = data["EKG"]
            else:
                ecg = None
            if "PPG" in data.keys():
                ppg = data["PPG"]
            else:
                ppg = None
            if "EOG" in data.keys():
                eog = data["EOG"]
            else:
                eog = None
            cols = ["ECG", "EKG", "RSP", "EDA", "EMG", "PPG", "EOG"]
            keep_keys = [key for key in data.keys() if key not in cols]
            if len(keep_keys) != 0:
                keep = data[keep_keys]
            else:
                keep = None

    # ECG
    if ecg is not None:
        print("Analizando ECG")
        ecg = nk.as_vector(ecg)
        ecg_signals, ecg_info = ecg_process_nuevo(ecg, sampling_rate=sampling_rate)
        bio_info.update(ecg_info)
        bio_df = pd.concat([bio_df, ecg_signals], axis=1)
        
    # RSP
    if rsp is not None:
        print("Analizando resp")
        rsp = nk.as_vector(rsp)
        rsp_signals, rsp_info = nk.rsp_process(rsp, sampling_rate=sampling_rate)
        bio_info.update(rsp_info)
        bio_df = pd.concat([bio_df, rsp_signals], axis=1)

    # EDA
    if eda is not None:
        print("Analizando EDA")
        eda = nk.as_vector(eda)
        eda_signals, eda_info = eda_process_nuevo(eda, sampling_rate=sampling_rate)
        bio_info.update(eda_info)
        bio_df = pd.concat([bio_df, eda_signals], axis=1)
        
    # EMG
    if emg is not None:
        print("Analizando EMG")
        emg = nk.as_vector(emg)
        emg_signals, emg_info = nk.emg_process(emg, sampling_rate=sampling_rate)
        bio_info.update(emg_info)
        bio_df = pd.concat([bio_df, emg_signals], axis=1)

    # PPG
    if ppg is not None:
        print("Analizando PPG")
        ppg = nk.as_vector(ppg)
        ppg_signals, ppg_info = nk.ppg_process(ppg, sampling_rate=sampling_rate)
        bio_info.update(ppg_info)
        bio_df = pd.concat([bio_df, ppg_signals], axis=1)

    # EOG
    if eog is not None:
        print("Analizando EOG")
        eog = nk.as_vector(eog)
        eog_signals, eog_info = nk.eog_process(eog, sampling_rate=sampling_rate)
        bio_info.update(eog_info)
        bio_df = pd.concat([bio_df, eog_signals], axis=1)

    # Additional channels to keep
    if keep is not None:
        if isinstance(keep, pd.DataFrame) or isinstance(keep, pd.Series):
            keep = keep.reset_index(drop=True)
        else:
            raise ValueError("The 'keep' argument must be a DataFrame or Series.")

        bio_df = pd.concat([bio_df, keep], axis=1)

    # RSA
    if (ecg is not None and rsp is not None) and rsa:
        print("Analizando HRV_RSA")
        rsa = nk.hrv_rsa(
            ecg_signals,
            rsp_signals,
            rpeaks=None,
            sampling_rate=sampling_rate,
            continuous=True,
        )
        bio_df = pd.concat([bio_df, rsa], axis=1)

    # Add sampling rate in dict info
    bio_info["sampling_rate"] = sampling_rate

    return bio_df, bio_info

def hrv_sliding_window(data,window_secs=30,sampling_rate=512,step_secs=15):
    window_length = window_secs*sampling_rate
    step = step_secs*sampling_rate

    hrv = pd.Series()
    hrv_per_windows = []
    
    windows_index = [i for i in range(0,len(data),step)]
    
    for i in windows_index:
        # Si el indice es menor a 15 segundos de señal, o mayor a 15 segundos antes
        # de que termine la señal, no se saca el hrv de esa ventana
        if i < window_length/2 or i > len(data)-(window_length/2):
            continue # Esto es provisional, debería sacarlo? Pienso que si usas la señal completa
                     # no vas a tener samples antes del momento 0 ni despues del último momento
            
        previous_samples = data[int(i-(window_length/2)):i]
        next_samples = data[i:int(i+(window_length/2))]
        window = pd.concat([previous_samples,next_samples],axis=0)
        
        window_hrv = nk.hrv_time(window, sampling_rate=sampling_rate)
        hrv_rmssd = pd.Series(window_hrv["HRV_RMSSD"])
        hrv_per_windows.append(hrv_rmssd)
        nan_fill = pd.Series([np.nan] * step)
        
#        if i == windows_index[1]:    
#            window_hrv_continuo = pd.concat([nan_fill,hrv_rmssd,nan_fill],ignore_index=True)
        
        window_hrv_continuo = pd.concat([hrv_rmssd,nan_fill],ignore_index=True)
            
        hrv = pd.concat([hrv,window_hrv_continuo],ignore_index=True)
            
    hrv = hrv.interpolate(method='linear')
        
    return hrv, hrv_per_windows, windows_index

def cvxEDA_pyEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol':1e-9}):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """
    
    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,y,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C], 
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*y,  -(Ct*y), -(Bt*y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n+nC]
    t = B*l + C*d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))

#%% Acá estoy armando para hacerlos todos juntos de una. Falta agregar lo de
# agarrar la lb de cada video

subjects = ["02","03","04", "05","06"] #

for subject in subjects:
    for bloque_n in range(1,9): # Por ahora no hubo sujetos que no tuvieran 8 bloques
        if subject == "05" and bloque_n == 3: # no se por qué pero este crashea
            continue
        
        print(f'Leyendo bloque {bloque_n} del sujeto {subject}')
        bloque = pd.read_csv(f'sub-{subject}/ses-A/df_bloque_{bloque_n}_sujeto_{subject}.csv').drop("Unnamed: 0",axis=1)
        
        # Agarro el principio del primer estimulo y el final del ultimo en el bloque
        primer_estimulo_bloque = bloque[bloque["description"] == "video_start"].index[0]
        ultimo_estimulo_bloque = bloque[bloque["description"] == "video_end"].index[-1]
        
        # Me quedo con 30 segundos previos al primer estimulo (15 de lb y 15 extra para sacar hrv)
        # y 15 posteriores al final del ultimo (para sacar hrv)
        bloque_final = bloque[primer_estimulo_bloque-(30*512):ultimo_estimulo_bloque+(15*512)]
        
        print('Extrayendo features con neurokit')
        df_bio, info_bio = bio_process_nuevo(ecg=bloque_final['ECG'], rsp=bloque_final['RESP'], eda=bloque_final['GSR'], keep=pd.DataFrame([bloque_final['time'],bloque_final['description']]).T, rsa=False, sampling_rate=512)
        
        # Los vuelvo a agarrar, porque el indice ya no corresponde al del df
        primer_estimulo = df_bio[df_bio["description"] == "video_start"].index[0]
        ultimo_estimulo = df_bio[df_bio["description"] == "video_end"].index[-1]
        
        print("Analizando HRV")
        hrv, windows, windows_index = hrv_sliding_window(df_bio["ECG_R_Peaks"])
        
        print("Obteniendo SMNA")
        [r, p, t, _ , _ , _ , _] = cvxEDA_pyEDA(df_bio["EDA_Clean"],1./512)
        df_bio["SMNA"] = p
        
        # Corto el df entre el principio del primer estimulo y el final del ultimo
        df_bio_final = df_bio[primer_estimulo-(15*512):ultimo_estimulo+1]
        df_bio_final.reset_index(inplace=True)
        df_bio_final.drop("index",axis=1,inplace=True)
        
        # Agrego HRV al df final (lo hice acá porque me desaparecian datos si lo hacia
        # antes de cortarlo)
        df_bio_final["HRV"] = hrv
        
        print(f'Guardando datos fisio del bloque {bloque_n} del sujeto {subject}')
        df_bio_final.to_csv(f'sub-{subject}/ses-A/fisio_bloque_{bloque_n}_sujeto_{subject}.csv')
