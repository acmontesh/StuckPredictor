# -*- coding: utf-8 -*-

def graph_around(window_begin, window_end, df, time='time_date', spp='spp', tq='tq', hkl='hkl', hkh='hkh', wellname='Well X'):
    #Plotting around sticking event to obtain a visual of exact time stamp
    #Remember: Time stamp of stuck event from NLP module gives context, but not exact stamp
    #VARIABLES ----------------------
        #window_begin: Stamp of initial datetime to display
        #window_end  : Stamp of final datetime to display
        #df          :Dataframe with time series of each sensor. 
            #NOTE: timedate column MUST be already in dtype=datetime.datime
        #time/spp/tq/hkl/hkh/Well Name: (*Optional), are the labels of each described variable.
            #NOTE: The labels MUST be the same than the dataframe columns, except for the well name.
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import matplotlib.dates as mdates
    buffer = df[(df[time]>window_begin) & (df[time]<window_end)].copy()
    x = buffer[time]
    y1 = buffer[spp].values
    y2 = buffer[tq].values
    y3 = buffer[hkl].values
    y4 = buffer[hkh].values
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(15,10))
    fig.suptitle('Channels over stuck: '+wellname)
    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    ax3.tick_params(axis='x', rotation=90)
    ax4.tick_params(axis='x', rotation=90)
    ax1.plot(x, y1, c='blue')
    ax2.plot(x, y2, c='red')
    ax3.plot(x, y3, c='orange')
    ax4.plot(x, y4, c='cyan')
    ax1.set_ylabel('SPP[psi]')
    ax2.set_ylabel('TQ[klb-ft]')
    ax3.set_ylabel('HKL[klb]')
    ax4.set_ylabel('HKH[ft]')
    buffer2 = []
    datebuffer = window_begin
    while (datebuffer<window_end):
        buffer2.append(datebuffer)
        datebuffer = datebuffer + dt.timedelta(minutes=1)
    ax1.set_xticks(buffer2)
    ax1.vlines(buffer2, ymin=0, ymax=3000, colors='gainsboro')
    ax2.vlines(buffer2, ymin=0, ymax=40, colors='gainsboro')
    ax3.vlines(buffer2, ymin=0, ymax=400, colors='gainsboro')
    ax4.vlines(buffer2, ymin=0, ymax=130, colors='gainsboro')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
    
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def mvms_forecast_multioutput(model, X, window_length, target_col=-1):
    #Forecasts new data from base time series of shape [Window length by Number of channels]
    #from a trained recurrent model. 
    #VARIABLES ----------------------
    #model  :               Trained recurrent model
    #X      :               Series from which the model will predict.
        #NOTE: X is reshaped reshaped to (1,T,D). i.e. T time stamps (Window) by D (channels)
    #window_length  :       Window length (input neurons to recurrent model)
    #target_col     :       Position of the target column in the X array (spp, tq or hkl)
        #NOTE: *Optional, by default it assumes the target is the last column
    
    import numpy as np
    N      =  X.shape[0]
    T      =  window_length
    D      =  X.shape[1]
    Row_T  =  np.arange(T)
    y_hat  =   model.predict(X[Row_T].reshape(1,T,D))
    return y_hat      


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def forecast_metrics_sumofdiff(yhat_spp, yhat_hkl, yhat_tq, yfw_spp, yfw_hkl, yfw_tq):
    #Returns the sum of the squared differences between forecast and real data 
    #VARIABLES ----------------------
    #yhat_spp  :   predicted time series (multistep) for pressure.
    #yhat_tq  :   predicted time series (multistep) for torque.
    #yhat_hkl  :   predicted time series (multistep) for hook load.
    #yfw_spp   :   Real pressure for the same predicted interval.
    #yfw_tq   :   Real torque for the same predicted interval.
    #yfw_hkl   :   Real hook load for the same predicted interval.
    
    import numpy as np
    delta1 = sum((yhat_spp-yfw_spp)**2)
    delta2 = sum((yhat_hkl-yfw_hkl)**2)
    delta3 = sum((yhat_tq-yfw_tq)**2)
    return delta1, delta2, delta3

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def subtract_window_multioutput(window_length, forecast_length, channels_array, target_array):
    #Extracts the X array of shape [N-T-F by T by D] by sliding a window over the 
    #entire time series, where:
        #N is the number of time steps in the original dataframe (rows)
        #T is the sliding window length
        #D is the number of channels (features), i.e. the number of sensors taken into account
        #for a particular variable.
    #VARIABLES ----------------------
    #window_length  :       Window length (input neurons to recurrent model)
    #forecast_length  :     Forecast window length (number of output neurons of the recurrent model)
    #channels_array     :   Original time series of all sensors readings converted to numpy array and standardized.
    #target_array:          Original time series of target sensor readings converted to numpy array.
    
    import numpy as np
    N,D   =   channels_array.shape
    T     =   window_length
    F     =   forecast_length
    N     =   N-T-F
    X     =   np.zeros((N,T,D))
    y     =   np.zeros((N,F))
    for i in range(N):
        X[i]        =    channels_array[i:i+T]
        y[i]        =    target_array[i+T:i+T+F]
        
    return X,y

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def mvms_forecast_classical(model, X, window_length, target_col=-1):
    import numpy as np
    N      =  X.shape[0]
    T      =  window_length
    D      =  X.shape[1]
    Row_T  =  np.arange(T)
    y_hat  =  np.zeros(N-T)
    for j in range(N-T):
        print(X[Row_T].shape)
        y_hat[j]            =   model.predict(X[Row_T].reshape(T,D))
        X[T+j,target_col]   =   y_hat[j]
        Row_T               =   Row_T+1
    return y_hat      
  

