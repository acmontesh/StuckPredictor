# -*- coding: utf-8 -*-
# =============================================================================
# 
# 
# 
# 
# 
# =============================================================================
class DFProcessor:
    
    def __init__(self, window_default_size = 20):
        self.window_default_size = window_default_size
        
    def _read_csv(self,path, stamps_column='time_date', filename='df.csv'):
        import pandas as pd
        df = pd.read_csv(path+'\\'+filename, parse_dates=[stamps_column])
        return df
    
    def process_preprocessed_DF(self, path, shoe, section_start_date, section_end_date, 
                                other_events_list, stucks_list,
                                T=10, F=10, dateformat='%Y-%m-%d %H:%M', 
                                stamps_column='time_date', 
                                bit_depth_column='bit_depth', time_gap_filter=0,
                                n_non_stuck_windows=5, 
                                general_sensors_list = ['time_date','hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 
                                                'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt', 'dhkl_dt', 
                                                'dhkh_dt', 'bit_depth', 'drpm_dt'],
                                filename_tosave = 'filtered_df_with_stucks.csv'):
    
        df = self._read_csv(path)
        filters_dict = {
            'inner_dates':[(section_start_date, section_end_date)],
            'outer_dates':other_events_list,
            'below_depth':shoe,
            'col_lessthan_col': ('bit_depth', 'hole_depth')
            }
        df = self._filter_dataframe(df, filters_dict, dateformat, stamps_column, bit_depth_column, time_gap_filter)
        df = self.filter_columns(general_sensors_list, df)
        df.to_csv(path+'\\'+filename_tosave, index=False)
        X, stucks_array, non_stucks_array, df, num_pipeline = self._process_filtered_DF(
                                                                    df, stucks_list, T, F,  
                                                                    dateformat, 
                                                                    stamps_column, bit_depth_column,
                                                                    n_non_stuck_windows, time_gap_filter
                                                                    )         
        return X, stucks_array, non_stucks_array, df, num_pipeline
    
    def _process_filtered_DF(self, df, stucks_list, T, F,
                             dateformat, stamps_column, bit_depth_column, n_non_stuck_windows, time_gap_filter):
        stucks_array, non_stucks_array = self._extract_windows(df, stucks_list, T, F, dateformat, 
                                                               stamps_column, n_non_stuck_windows)
        filters_dict = {
            'outer_dates':stucks_list
            } 
        df = self._filter_dataframe(df, filters_dict, dateformat, stamps_column, bit_depth_column, time_gap_filter=5)
        df = df.drop([stamps_column],axis=1)
        X, num_pipeline  = self._pipeline(df)
        
        return X, stucks_array, non_stucks_array, df, num_pipeline
    
    def prepare_for_specific_channel(self, channel_name, X, df, specific_sensors_list, 
                                     stucks_array, non_stucks_array, trained_pipeline):
        indices_X, index_y = self._extract_indices(specific_sensors_list, df, channel_name)   
        X,y = self._extract_Xy_with_indices(indices_X, index_y, X)
        y = df[channel_name]
        y_stucks = []
        y_non_stucks = []
        for stuck_array in stucks_array:
            y_stucks.append(stuck_array[channel_name].copy())
        for non_stuck_array in non_stucks_array:
            y_non_stucks.append(non_stuck_array[channel_name].copy())
        stucks_array = self._transform_from_trained_pipeline(trained_pipeline, stucks_array)
        non_stucks_array = self._transform_from_trained_pipeline(trained_pipeline, non_stucks_array)        
        stucks_array = self._filter_columns_from_list(stucks_array, specific_sensors_list, df)
        non_stucks_array = self._filter_columns_from_list(non_stucks_array, specific_sensors_list, df)
        
        return X, y, stucks_array, non_stucks_array, y_stucks, y_non_stucks
    
    def _transform_from_trained_pipeline(self, trained_pipeline, arrays_list):
        arrays_transformed_list=[]
        for array in arrays_list:
            arrays_transformed_list.append(trained_pipeline.transform(array))
        return arrays_transformed_list
    
    def _filter_columns_from_list(self, events_list, columns, df):
        events_arrays_list = []
        for event in events_list:
            _,X = self.filter_columns(columns, df, event)
            events_arrays_list.append(X)
        return  events_arrays_list
            
    def process_from_CSV_filtered_DF(self, path, stucks_list, 
                                T=10, F=10, dateformat='%Y-%m-%d %H:%M', 
                                stamps_column='time_date', bit_depth_column='bit_depth',
                                time_gap_filter=0,
                                n_non_stuck_windows=5,
                                filename_toopen = 'filtered_df_with_stucks.csv'):
        df = self._read_csv(path, filename=filename_toopen)   
        X, stucks_array, non_stucks_array, df, num_pipeline = self._process_filtered_DF(
                                                                    df, stucks_list, T, F, dateformat, 
                                                                    stamps_column, bit_depth_column, 
                                                                    n_non_stuck_windows, time_gap_filter
                                                                    )  
        return X, stucks_array, non_stucks_array, df, num_pipeline
    
    def _pipeline(self,df):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import KNNImputer        
        num_pipeline = Pipeline([
            ('imputer', KNNImputer()),
            ('std_scaler', StandardScaler())
        ])        
        X   =  df.values        
        X   = num_pipeline.fit_transform(X)
        return X, num_pipeline
    
    def _filter_dataframe(self, df, filters_dict, dateformat, stamps_column, bit_depth_column, time_gap_filter):
        for filter_type in filters_dict:
            if filter_type=='inner_dates':
                for begin_end_tuple in filters_dict.get('inner_dates'):
                    df = self._filter_inner_dates(df, begin_end_tuple, dateformat, stamps_column)
            elif filter_type=='outer_dates':
                for begin_end_tuple in filters_dict.get('outer_dates'):
                    df = self._filter_outer_dates(df, begin_end_tuple, dateformat, stamps_column, time_gap_filter)
            elif filter_type=='above_depth':
                df = self._filter_depth_above(df, filters_dict.get('above_depth'), bit_depth_column)
            elif filter_type=='below_depth':
                df = self._filter_depth_below(df, filters_dict.get('below_depth'), bit_depth_column)
            elif filter_type=='col_lessthan_col':
                df = self._filter_col_lessthan_col(df, filters_dict.get('col_lessthan_col'))
        return df
    
    def _filter_inner_dates(self, df, begin_end_tuple, dateformat, stamps_column):
        import datetime as dt
        DATE_START = dt.datetime.strptime(begin_end_tuple[0], dateformat)
        DATE_END   = dt.datetime.strptime(begin_end_tuple[1], dateformat)
        return df[(df[stamps_column]>DATE_START) & (df[stamps_column]<DATE_END)]
    
    def _filter_outer_dates(self, df, begin_end_tuple, dateformat, stamps_column, time_gap_filter):
        import datetime as dt
        DATE_START = dt.datetime.strptime(begin_end_tuple[0], dateformat)
        DATE_END   = dt.datetime.strptime(begin_end_tuple[1], dateformat)
        if not time_gap_filter==0:
            DATE_START = DATE_START - dt.timedelta(minutes=time_gap_filter)
            DATE_END = DATE_END + dt.timedelta(minutes=time_gap_filter)
        return df[(df[stamps_column]<DATE_START) | (df[stamps_column]>DATE_END)]
    
    def _filter_depth_above(self, df, cutoff_depth, bit_depth_column):
        return df[(df[bit_depth_column]<cutoff_depth)]
    
    def _filter_depth_below(self, df, cutoff_depth, bit_depth_column):
        return df[(df[bit_depth_column]>cutoff_depth)]
    
    def _filter_col_lessthan_col(self, df, columns_tuple):
        return df[(df[columns_tuple[0]]<df[columns_tuple[1]])]
    
    def _extract_windows(self, df, stucks_list, T, F, dateformat, stamps_column, n_non_stuck_windows):
        import datetime as dt
        import numpy as np
        stucks_array = []
        non_stucks_array = []
        
        for t in stucks_list:
            STUCK_START = dt.datetime.strptime(t[0], dateformat)
            condition_evaluation = lambda date: np.where(df[stamps_column]==date)[0]
            while condition_evaluation(STUCK_START).size==0:
                STUCK_START = STUCK_START + dt.timedelta(seconds=10)
            i_stuck = condition_evaluation(STUCK_START)[0]
            df_stucks = df[i_stuck-T-F:i_stuck]
            stucks_array.append(df_stucks.drop([stamps_column], axis=1))
        
        N = df.shape[0]
        for i in range(n_non_stuck_windows):
            index_START     = np.random.randint(N-T-F-1)
            index_END       = index_START + T + F
            df_non_stuck    = df[index_START:index_END]
            non_stucks_array.append(df_non_stuck.drop([stamps_column], axis=1))
            
        return stucks_array, non_stucks_array
    
    def filter_columns(self, columns_list, df, X=None):
        df = df[columns_list]
        if X is None:
            return df
        else:
            indices_X = self._extract_indices(columns_list, df)
            X = X[:,indices_X]
            return df, X
    
    def _extract_indices(self, columns_list, df, y_column=None):
        indices_X = []
        for col in columns_list:
            indices_X.append(df.columns.get_loc(col))
        if y_column is None:
            return indices_X
        else:
            index_y = df.columns.get_loc(y_column)
            return indices_X, index_y
    
    def _extract_Xy_with_indices(self,idxs_X, idx_y, X):
        y = X[:,idx_y]
        X = X[:,idxs_X]
        return X,y
    
# =============================================================================
#     
#     
# 
# 
# =============================================================================


class TimeSlider:
    
    def __init__(self, T, F):
        self.T = T
        self.F = F
        
    def process_sets(self, X, y, stucks_array, non_stucks_array, y_stucks, y_non_stucks):
        X_train, y_train = self._slide(X, y)
        stucks_array = self._adjust_sample_windows(stucks_array)
        non_stucks_array = self._adjust_sample_windows(non_stucks_array)
        y_stucks= self._adjust_sample_windows(y_stucks)
        y_non_stucks=self._adjust_sample_windows(y_non_stucks)
        return X_train, y_train, stucks_array, non_stucks_array, y_stucks, y_non_stucks

    def _slide(self, channels_array, target_array):
        import numpy as np
        N,D   =   channels_array.shape
        N     =   N-self.T-self.F
        X     =   np.zeros((N,self.T,D))
        y     =   np.zeros((N,self.F))
        for i in range(N):
            X[i]        =    channels_array[i:i+self.T]
            y[i]        =    target_array[i+self.T:i+self.T+self.F]       
        return X,y
    
    def _adjust_sample_windows(self, samples_array):
        samples_trim_array=[]
        for sample in samples_array:
            if len(sample.shape)==1:
                samples_trim_array.append(sample[-(self.T+self.F):])
            else:
                samples_trim_array.append(sample[-(self.T+self.F):,:])
        return samples_trim_array

# =============================================================================
# 
# 
# 
# 
# 
# 
# =============================================================================

class Modeler:
    def __init__(self, id_model):
        self.id_model = id_model
        self.score = (0,0)
        
    def get_id(self):
        return self.id_model
    
    def get_score(self):
        return self.score
    
    def set_id(self,identifier):
        self.id_model = identifier
        
    def set_score(self,scores_tuple):
        self.score = scores_tuple

# =============================================================================
# 
# 
# 
# 
# 
# 
# =============================================================================



class RecurrentPredictor(Modeler):
        
    def build_from_Xy(self, X, y, model_dictionary, path,
                      T, F,
                      learning_rate = 0.01, epochs=40,
                      identifier = ''
                      ):
        from tensorflow.keras.layers import Input, Dense, LSTM, GRU, SimpleRNN
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        
        N   =   X.shape[0]
        D   =   X.shape[2]
        i = Input(shape=(T, D))
        
        if model_dictionary['model_type']=='lstm' and model_dictionary['hidden_layers']==1:        
            x = LSTM(model_dictionary['hidden_neurons'])(i)
            x = Dense(F)(x)
            model = Model(i, x)
        elif model_dictionary['model_type']=='lstm' and model_dictionary['hidden_layers']==2:        
            x = LSTM(model_dictionary['hidden_neurons'], return_sequences=True)(i)
            x = LSTM(model_dictionary['hidden_neurons'])(x)
            x = Dense(F)(x)
            model = Model(i, x)
        elif model_dictionary['model_type']=='srnn' and model_dictionary['hidden_layers']==1:        
            x = SimpleRNN(model_dictionary['hidden_neurons'])(i)
            x = Dense(F)(x)
            model = Model(i, x)
        elif model_dictionary['model_type']=='srnn' and model_dictionary['hidden_layers']==2:        
            x = SimpleRNN(model_dictionary['hidden_neurons'], return_sequences=True)(i)
            x = SimpleRNN(model_dictionary['hidden_neurons'])(x)
            x = Dense(F)(x)
            model = Model(i, x)
        elif model_dictionary['model_type']=='gru' and model_dictionary['hidden_layers']==1:        
            x = GRU(model_dictionary['hidden_neurons'])(i)
            x = Dense(F)(x)
            model = Model(i, x)
        else:        
            x = GRU(model_dictionary['hidden_neurons'], return_sequences=True)(i)
            x = GRU(model_dictionary['hidden_neurons'])(x)
            x = Dense(F)(x)
            model = Model(i, x)
        
        model.compile(
          loss='mse',
          optimizer=Adam(lr=learning_rate),
        )       
        r, model = self._train_model(model, X, y, N, epochs)
        self._save_history(r.history['loss'], path)
        self._save_history(r.history['val_loss'], path)
        self._save_model(model, path, identifier)
        return r, model

    def build_from_h5py(self, path, spec_id_model=''):
        from keras.models import load_model
        model = load_model(path+'\\'+spec_id_model+'model.h5')
        return model
    
    def _train_model(self, model, X, y, N, epochs):
        import tensorflow as tf
        r = model.fit(
              X[:2*N//3], y[:2*N//3],
              epochs=epochs,
              validation_data=(X[-N//3:], y[-N//3:]))
        return r, model
    
    def _save_history(self, r, path):
        import numpy
        numpy.savetxt(path+'\\'+str(self.id_model)+"_losshistory.csv", r, delimiter=",")
    
    def _save_model(self, model, path, identifier):
        model.save(path+'\\'+identifier+"model.h5")
    
    def _forecast(self, X, model, T):
        import numpy as np        
        D      =  X.shape[1]
        Row_T  =  np.arange(T)
        y_hat           =   model.predict(X[Row_T].reshape(1,T,D))[0]
        return y_hat.reshape((-1,))
    
    def forecast_measure_single_sample(self, X, model, y_true, T):
        y_hat = self._forecast(X, model, T)
        window_mse, window_std, window_kur = self._measure(y_hat, y_true)
        return window_mse, window_std, window_kur
    
    def mape_measure_single_sample(self, X, model, y_true, T):
        y_hat = self._forecast(X[:T], model, T)
        mape = self._mape(y_hat, y_true.values)
        return mape
    
    def _mape(self, ypred, ytrue):
        mape = abs(ypred-ytrue)/abs(ytrue)
        n = ytrue.size
        mape = sum(mape)/n
        return mape
    
    def _measure(self, y_hat, y_true):
        import numpy as np
        from scipy.stats import kurtosis as kur
        diff_array = (y_hat - y_true)**2
        window_mse = np.mean(diff_array)
        window_std = np.std(diff_array)
        window_kur = kur(diff_array)
        return window_mse, window_std, window_kur
    
    def calculate_score_over_samples(self,samples_list, labels_list,model, T, F):
        score_mse = 0
        score_std = 0
        for j, sample in enumerate(samples_list):
            y_true = labels_list[j]
            mse,std,_ = self.forecast_measure_single_sample(sample[:T],model,y_true[T:], T)
            score_mse = score_mse + mse
            score_std = score_std + std
        score_mse = score_mse/len(samples_list)
        score_std = score_std/len(samples_list)
        self.set_score((score_mse, score_std))
        return score_mse, score_std

# =============================================================================
# 
# 
# 
# 
# 
# 
# 
# 
# 
# =============================================================================

class StuckPredictor(Modeler):
    
    def build_from_json(self, path, model_dictionary, identifier, filename='results_table.json', epochs=40, 
                       label_name='labels', test_fraction=0.3):
        df = self._read_json(path, filename)
        y = df[label_name].values
        df = df.drop([label_name], axis=1)
        X,_ = self._pipeline(df)
        model, cm, accu, sens, spec, fpr, fnr = self._train_validate_model(X, y, model_dictionary, 
                                              test_fraction, identifier)
        return model, cm, accu, sens, spec, fpr, fnr
    
    def build_from_table(self, X,y, model_dictionary, identifier, epochs=40, 
                       label_name='labels', test_fraction=0.3):
        model, cm, accu, sens, spec, fpr, fnr = self._train_validate_model(X, y, model_dictionary, 
                                              test_fraction, identifier)
        return model, cm, accu, sens, spec, fpr, fnr
    
    def _train_validate_model(self, X, y, model_dictionary, test_fraction, identifier):
        from sklearn.model_selection import train_test_split
        import numpy as np
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, 
                                                            random_state=np.random.randint(0,1200))
        model_type = model_dictionary['model_type']
        model_params_dict = model_dictionary['params']
        if model_type =='svm':
            from sklearn.svm import SVC
            C = model_params_dict['C_value']
            kernel = model_params_dict['kernel']
            gamma = model_params_dict['gamma_value']
            model = SVC(C=C, kernel=kernel, gamma=gamma)
        elif model_type =='rf':
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = model_params_dict['n_trees']
            max_depth = model_params_dict['max_depth']
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif model_type =='dt':
            from sklearn.tree import DecisionTreeClassifier
            max_depth = model_params_dict['max_depth_tree']
            model = DecisionTreeClassifier(random_state=0,max_depth=max_depth)
        
        model.fit(X_train, y_train)  
        y_hat = model.predict(X_test)
        cm, accu, sens, spec, fpr, fnr = self._display_classification_metrics(y_test, y_hat) 
        self.set_score((accu, spec))
        self._save_model(model, identifier)
        return model, cm, accu, sens, spec, fpr, fnr
    
    def _plot_history(self):
        pass
    
    def _display_confusion_matrix(self):
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_classification
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        X, y = make_classification(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=0)
        clf = SVC(random_state=0)
        clf.fit(X_train, y_train)        
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test, y_test)        
        plt.show()
    
    def _display_classification_metrics(self, y_true, y_hat):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_hat, labels=[0,1])
        accu, sens, spec, fpr, fnr = self._calculate_metrics(cm)
        # =============================================================================
        #         print('Accuracy: ', accu)
        #         print('Sensitivity (Recall): ', sens)
        #         print('Specificity (Selectivity): ', spec)
        #         print('False Alarms (Fallout): ', fpr)
        #         print('Miss Rate: (FNR)', fnr)
        # =============================================================================
        return cm, accu, sens, spec, fpr, fnr
        
    def _calculate_metrics(self, conf_matrix):
        tp = conf_matrix[1,1]
        tn = conf_matrix[0,0]
        fp = conf_matrix[0,1]
        fn = conf_matrix[1,0]
        n = tn + fp
        p = tp + fn
        accu = (tp+tn) / (p+n)
        sens = tp/p
        spec = tn/n
        fpr = fp/n
        fnr = fn/p
        return accu, sens, spec, fpr, fnr
    
    def _read_json(path, filename='results_table.json'):
        import pandas as pd
        df = pd.read_json(path+'\\'+filename)
        return df
    
    def _pipeline(self,df):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import KNNImputer        
        num_pipeline = Pipeline([
            ('imputer', KNNImputer()),
            ('std_scaler', StandardScaler())
        ])        
        X   =  df.values        
        X   = num_pipeline.fit_transform(X)
        return X, num_pipeline
    
    def _save_model(self, model, identifier):
        from joblib import dump
        dump(model, identifier+'model.joblib')
    
    


# =============================================================================
# 
# 
# 
# 
# 
# 
# 
# =============================================================================

class Erlenmeyer:
    
    def __init__(self, bars=10, steps=20):
        from TDG_classes import ProgressBar
        self.counter_experiments = 0
        self.performance_record = {}
        self.progress_bar = ProgressBar(bars,steps)
        
        
    def update_counter_experiments(self, steps=1):
        self.counter_experiments = self.counter_experiments + steps
    
    def get_counter_experiments(self):
        return self.counter_experiments

    def get_performance_record(self):
        return self.performance_record
    
    def set_performance_record(self, perf_dictionary):
        self.performance_record = perf_dictionary
        
    def save_performance_dictionary(self, filename='experiment_best_results.json'):
        import json
        with open(filename, 'w') as file:
            json.dump(self.performance_record, file)
    
    def run_experiments(self):
        pass
    

# =============================================================================
#     
#     
#     
# 
# 
# =============================================================================

class RecurrentErlenmeyer(Erlenmeyer):
    
    def run_experiments(self, experiments_dictionary, wells_dictionary, 
                        channels_dictionary, df_status='pristine', 
                        best_score_initializer=1000000, epochs=40):
        import matplotlib.pyplot as plt
        best_model_per_well = {}
        for well in wells_dictionary:
            print('-------------------------- WORKING ON WELL: '+well)
            processor = DFProcessor()
            if df_status=='pristine':           
                X, stucks_array, non_stucks_array, df, num_pipeline = processor.process_preprocessed_DF(wells_dictionary[well].get('path'),
                                                                                      wells_dictionary[well].get('shoe'),
                                                                                      wells_dictionary[well].get('beginend_dates')[0],
                                                                                      wells_dictionary[well].get('beginend_dates')[1],
                                                                                      wells_dictionary[well].get('others_list'),
                                                                                      wells_dictionary[well].get('stucks_list'))
            else:
                X, stucks_array, non_stucks_array, df, num_pipeline = processor.process_from_CSV_filtered_DF(wells_dictionary[well].get('path'),
                                                                                           wells_dictionary[well].get('stucks_list'))
            best_model_per_channel = {}
            for channel in channels_dictionary:
                print('-------------------------- WORKING ON CHANNEL: '+channel)
                X_0, y_0, stucks_array_0, non_stucks_array_0, y_stucks_0, y_non_stucks_0 = processor.prepare_for_specific_channel(
                    channel, X, df, channels_dictionary[channel].get('specific_sensors_list'), 
                    stucks_array, non_stucks_array, num_pipeline)
                models = experiments_dictionary['models']
                windows = experiments_dictionary['windows']
                best_score = best_score_initializer
                best_dict = {}
                model_counter = 0
                for T,F in windows:
                    slider = TimeSlider(T,F)
                    X_train, y_train, stucks_array_1, non_stucks_array_1, y_stucks_1, y_non_stucks_1 = slider.process_sets(
                        X_0, y_0, stucks_array_0, non_stucks_array_0, y_stucks_0, y_non_stucks_0)
                    for model in models.get('model'):
                        for L,H in models.get('params'):
                            self.progress_bar.update_bar()
                            self.update_counter_experiments()
                            counterlabel = str('%05d' %self.counter_experiments)
                            print('WORKING ON MODEL ID: '+counterlabel)
                            rp = RecurrentPredictor(counterlabel)
                            model_dictionary = {'model_type':model,
                                                'hidden_layers':L,
                                                'hidden_neurons':H}
                            r, m = rp.build_from_Xy(X_train, y_train, model_dictionary, 
                                                        wells_dictionary[well].get('path'), T, F, identifier=counterlabel,
                                                        epochs=epochs)
                            fig,ax=plt.subplots()
                            ax.plot(r.history['loss'])
                            score_mse, score_std = rp.calculate_score_over_samples(
                                non_stucks_array_1, y_non_stucks_1, m, T, F)
                            
                            if model_counter==0:
                                best_score = score_mse
                            model_counter = model_counter+1
                            if score_mse <= best_score:
                                best_dict = {'model_id':counterlabel,
                                             'score_mse': score_mse,
                                             'score_std': score_std,
                                             'T':T,
                                             'F':F,
                                             'model_type':model,
                                             'L':L,
                                             'H':H,
                                             'path':wells_dictionary[well].get('path')
                                                    }
                best_model_per_channel[channel]=best_dict
            best_model_per_well[well] = best_model_per_channel
        self.set_performance_record(best_model_per_well)
        self.save_performance_dictionary(filename='recurrent_erlenmeyer_results.json')
    
# =============================================================================
#     
#     
#     
# 
# 
# =============================================================================

class SupervisedErlenmeyer(Erlenmeyer):
    
    
    
    
    def run_experiments(self, experiments_dictionary, wells_dictionary, channels_dictionary):
        import numpy as np

        table_builder = FeedTableBuilder()
        models = experiments_dictionary['models']
        params = models['params']
        perf_dict = {'models':[],
                     'aug_type':[],
                     'aug_factor':[],
                     'aug_noise_mean':[],
                     'aug_noise_stdv':[],
                     'accu':[],
                     'sens':[],
                     'spec':[],
                     'fpr':[],
                     'fnr':[]}
        best_models_dictionary = table_builder.dictionary_from_json()
        augmentation_types = experiments_dictionary['augmentation_type']
        
        for aug_type in augmentation_types:
            noise_mean, noise_std = ('','')
            aug_factor = ''
            if aug_type == 'replication':                
                for aug_factor in experiments_dictionary['augmentation_factor']:
                    X,y,_,_ = table_builder.build_dataset(best_models_dictionary, wells_dictionary, 
                                                          channels_dictionary, {'type' : aug_type,
                                                                                'factor': aug_factor})
                    perf_dict = self._iterate_over_models(models, params, perf_dict, X, y,
                                                          aug_type, aug_factor, noise_mean, noise_std)
            elif aug_type == 'noise_injection':
                for aug_tuple in experiments_dictionary['aug_gaussian_noise']:
                    noise_mean, noise_std = aug_tuple
                    X,y,_,_ = table_builder.build_dataset(best_models_dictionary, wells_dictionary, 
                                                          channels_dictionary, 
                                                          {'type' : aug_type,
                                                           'mean': noise_mean, 'std': noise_std})
                    perf_dict = self._iterate_over_models(models, params, perf_dict, X, y,
                                                          aug_type, aug_factor, noise_mean, noise_std)
            
            elif aug_type == 'combination':
                for aug_factor in experiments_dictionary['augmentation_factor']:
                    for aug_tuple in experiments_dictionary['aug_gaussian_noise']:
                        noise_mean, noise_std = aug_tuple
                        X,y,_,_ = table_builder.build_dataset(best_models_dictionary, wells_dictionary, 
                                                              channels_dictionary, 
                                                              {'type' : aug_type, 'factor':aug_factor,
                                                               'mean': noise_mean, 'std': noise_std})
                        perf_dict = self._iterate_over_models(models, params, perf_dict, X, y,
                                                              aug_type, aug_factor, noise_mean, noise_std)
            
        self.set_performance_record(perf_dict)
        self.save_performance_dictionary('supervised_erlenmeyer_results.json')
        

                          
        
    def _iterate_over_models(self,models,params,perf_dict, X, y, 
                             aug_type, aug_factor, noise_mean, noise_std):
        for m in models['model']:
            self.progress_bar.update_bar()
            if m=='svm':
                for C in params['C']:
                    for gamma in params['gamma']:
                        for kernel in params['kernel']:                                
                                  
                            parameters = {'C_value': C,
                                          'gamma_value': gamma,
                                          'kernel':kernel}
                            _, cm, accu, sens, spec, fpr, fnr = self._grid(
                                                                parameters, m, X,y)
                            perf_dict,label = self._include_performance(perf_dict, m, parameters, 
                                                                  accu, sens, spec, fpr, fnr,
                                                                  aug_type, aug_factor, noise_mean, noise_std)
                            print('WORKING ON MODEL ID: '+label)
            elif m=='rf':
                for depth in params['max_depth']:
                    for trees in params['n_trees']:
                              
                        parameters = {'max_depth': depth,
                                      'n_trees': trees}
                        _, cm, accu, sens, spec, fpr, fnr = self._grid(
                                                                parameters, m, X,y)
                        perf_dict,label = self._include_performance(perf_dict, m, parameters, 
                                                                  accu, sens, spec, fpr, fnr,
                                                                  aug_type, aug_factor, noise_mean, noise_std)
                        print('WORKING ON MODEL ID: '+label)
                        
            elif m=='dt':
                for depth in params['max_depth_tree']:                                  
                    parameters = {'max_depth_tree': depth}
                    _, cm, accu, sens, spec, fpr, fnr = self._grid(
                                                            parameters, m, X,y)
                    perf_dict,label = self._include_performance(perf_dict, m, parameters, 
                                                              accu, sens, spec, fpr, fnr,
                                                              aug_type, aug_factor, noise_mean, noise_std)
                    print('WORKING ON MODEL ID: '+label)

        return perf_dict
    
    
    def _include_performance(self,perf_dict, m, parameters, 
                             accu, sens, spec, fpr, fnr,
                             aug_type, aug_factor, noise_mean, noise_std):
        label = m
        for p in parameters:
            label = label+p+str(parameters[p])+'_'
        perf_dict['models'].append(label)
        perf_dict['accu'].append(accu)
        perf_dict['sens'].append(sens)
        perf_dict['spec'].append(spec)
        perf_dict['fpr'].append(fpr)
        perf_dict['fnr'].append(fnr)
        perf_dict['aug_type'].append(aug_type)
        perf_dict['aug_factor'].append(aug_factor)
        perf_dict['aug_noise_mean'].append(noise_mean)
        perf_dict['aug_noise_stdv'].append(noise_std)
        return perf_dict, label    
    
    
    def _grid(self,parameters, m, X,y):
        self.update_counter_experiments()
        counterlabel = str('%05d' %self.counter_experiments)
        predictor = StuckPredictor(self.counter_experiments)  
        model_dictionary = {'model_type':m,
                            'params': parameters}
        model, cm, accu, sens, spec, fpr, fnr = predictor.build_from_table(
                                    X,y, model_dictionary, counterlabel)
        return model, cm, accu, sens, spec, fpr, fnr


# =============================================================================
# 
# 
# 
# 
# 
# 
# 
# =============================================================================


class FeedTableBuilder:
    
    def __init__(self):
        self.table = {}
    
    def dictionary_from_json(self, filename='recurrent_erlenmeyer_results.json'):
        import json
        with open(filename, 'r') as file:
            dictionary = json.load(file)
        return dictionary
    
    def build_dataset(self, best_models_dictionary, wells_dictionary, 
                      channels_dictionary, augmentation_dictionary):
        import pandas as pd
        import winsound
        beep_alert_frequency = 500
        beep_alert_duration = 500 
        
        channels = list(channels_dictionary.keys())
        for channel in channels:
            msekey = channel+'_mse'
            self.table[msekey] = []
            stdkey = channel+'_std'
            self.table[stdkey] = []
            kurkey = channel+'_kur'
            self.table[kurkey] = []
        self.table['labels'] = []    


        for well in best_models_dictionary:
            processor = DFProcessor()
            X, stucks_array, non_stucks_array, df, num_pipeline = processor.process_from_CSV_filtered_DF(
                                            wells_dictionary[well].get('path'),
                                            wells_dictionary[well].get('stucks_list'))       
            stucks_array, non_stucks_array = self._apply_augmentation(augmentation_dictionary, 
                                                                      stucks_array, non_stucks_array)
            for i,channel in enumerate(best_models_dictionary[well]):
                print('******* FILLING TABLE FOR CHANNEL '+channel+' IN WELL: '+well)
                winsound.Beep(beep_alert_frequency, beep_alert_duration)
                stucks_array_0, non_stucks_array_0, y_stucks_0, y_non_stucks_0, T = self._extract_samples(
                        processor, best_models_dictionary, channels_dictionary,
                         well, channel, X, stucks_array, non_stucks_array, df, num_pipeline)
                rp = RecurrentPredictor('default')
                model = rp.build_from_h5py(wells_dictionary[well].get('path'),
                                           best_models_dictionary[well][channel]['model_id'])
                self._feed_table_with_cases(True, channel, model, stucks_array_0, y_stucks_0, rp, T, i)
                self._feed_table_with_cases(False, channel, model, non_stucks_array_0, y_non_stucks_0, rp, T, i)
        self._save_table()
        df = pd.DataFrame(self.table)
        y = df['labels'].values
        X = df.drop(['labels'], axis=1).values
        X = self._pipeline(X)
        return X,y,df,self.table

    
    def _apply_augmentation(self, augmentation_dictionary, stucks_array, non_stucks_array): 
        aug_type = augmentation_dictionary['type']
        if aug_type == 'replication':
            aug_factor = augmentation_dictionary['factor']
            stucks_array = stucks_array*aug_factor
            non_stucks_array = non_stucks_array*aug_factor
        elif aug_type == 'noise_injection':
            mean = augmentation_dictionary['mean']
            std = augmentation_dictionary['std']
            stucks_array = self._inject_noise(stucks_array, mean, std)
            non_stucks_array = self._inject_noise(non_stucks_array, mean, std)
        elif aug_type == 'combination':
            aug_factor = augmentation_dictionary['factor']
            mean = augmentation_dictionary['mean']
            std = augmentation_dictionary['std']
            stucks_array, non_stucks_array = self._apply_augmentation({'type': 'replication',
                                                     'factor':aug_factor}, stucks_array, non_stucks_array)
            stucks_array, non_stucks_array = self._apply_augmentation({'type': 'noise_injection',
                                                     'mean':mean, 'std':std}, stucks_array, non_stucks_array)
        return stucks_array, non_stucks_array


    def _inject_noise(self, array, mean, std):
        import numpy as np
        new_array=[]
        for window in array:
            new_array.append(window)
            rows = window.shape[0]
            for column in window.columns:
                noise_factors = np.random.normal(mean,std,rows)
                window[column] = window[column]*noise_factors
            new_array.append(window)
        return new_array

    
    def _pipeline(self,X):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        return X
            
    def _save_table(self, filename='results_table.json'):
        import json
        with open(filename, 'w') as file:
            json.dump(self.table, file)
    
    def _feed_table_with_cases(self, stuck_boolean, channel, model, arrays, y_arrays, recurrent_predictor, T, i_channel):        
        for i,array in enumerate(arrays):
            window_mse, window_std, window_kur = recurrent_predictor.forecast_measure_single_sample(
                                                                array, model, y_arrays[i][T:],T)
            msekey = channel+'_mse'
            stdkey = channel+'_std'
            kurkey = channel+'_kur'
            self.table[msekey].append(window_mse)
            self.table[stdkey].append(window_std)
            self.table[kurkey].append(window_kur)
            if i_channel==0:
                if stuck_boolean is True:
                    self.table['labels'].append(1)
                else:
                    self.table['labels'].append(0)
     
                    
    def _extract_samples(self, processor, best_models_dictionary, channels_dictionary,
                         well, channel, X, stucks_array, non_stucks_array, df, num_pipeline):
        _, _, stucks_array, non_stucks_array, y_stucks, y_non_stucks = processor.prepare_for_specific_channel(
                                        channel, X, df, channels_dictionary[channel], 
                                        stucks_array, non_stucks_array, num_pipeline)
        T = best_models_dictionary[well][channel]['T']
        F = best_models_dictionary[well][channel]['F']
        slider = TimeSlider(T,F)
        stucks_array = slider._adjust_sample_windows(stucks_array)
        non_stucks_array = slider._adjust_sample_windows(non_stucks_array)
        y_stucks= slider._adjust_sample_windows(y_stucks)
        y_non_stucks=slider._adjust_sample_windows(y_non_stucks)
        return stucks_array, non_stucks_array, y_stucks, y_non_stucks, T
    
# =============================================================================
# 
# 
# 
# 
#           MAPE CALCULATOR
# 
# 
# 
# =============================================================================

class MapeCalculator:
    def __init__(self):
        self.performance_dict = {}
    
    def calculateMAPE(self,X,y,N,loaded_model,T,F,n_windows,n_iters,rp):
        import numpy as np
        mapes_global=[]
        for i in range(n_iters):
            mapes_local=[]
            if i%10==0:
                print('==>',end='')
            for j in range(n_windows):
                finished = False
                while not finished:
                    index_START     = np.random.randint(N)
                    index_END       = index_START + T
                    X_fw            = X[index_START:index_END].copy()
                    y_fw            = y[index_END:index_END+F].copy()
                    if sum(y_fw<=0)==0:
                        finished=True
                y_fc  =  rp._forecast(X_fw, loaded_model, T)
                mapes_local.append(sum(abs(y_fw-y_fc)/abs(y_fw))/F)
           #print(mape)
            mapes_global.append(np.mean(mapes_local))
        return np.mean(mapes_global), np.std(mapes_global), mapes_global
    
    
    def calculate_MAPE_wells(self, wells_dictionary, channels_dictionary, n_windows=10, n_iters=1000):
        import winsound
        import numpy as np
        beep_alert_frequency, beep_alert_duration = (1500,500)
        
        ft = FeedTableBuilder()
        best_models_dictionary=ft.dictionary_from_json()
        for well in best_models_dictionary:
            self.performance_dict[well] = {
                'spp':{},
                'tq':{},
                'hkl':{}
                }
            processor = DFProcessor()
            X, stucks_array, non_stucks_array, df, num_pipeline = processor.process_from_CSV_filtered_DF(
                                            wells_dictionary[well].get('path'),
                                            wells_dictionary[well].get('stucks_list'))       
            
            for i,channel in enumerate(best_models_dictionary[well]):
                print('******* CACULATING MAPE FOR CHANNEL '+channel+' IN WELL: '+well)
                winsound.Beep(beep_alert_frequency, beep_alert_duration)
                X_0, y_0, _,_,_,_ = processor.prepare_for_specific_channel(channel,
                                                             X,
                                                             df,
                                                             specific_sensors_list=channels_dictionary[channel],
                                                             stucks_array=stucks_array,
                                                             non_stucks_array=non_stucks_array,
                                                             trained_pipeline=num_pipeline)
                rp = RecurrentPredictor('default')
                loaded_model = rp.build_from_h5py(wells_dictionary[well].get('path'),
                                           best_models_dictionary[well][channel]['model_id'])
                T = best_models_dictionary[well][channel]['T']
                F = best_models_dictionary[well][channel]['F']
                N = X_0.shape[0]-T-F-1
                
                mu, sigma, mapes_global = self.calculateMAPE(X_0,y_0,N,loaded_model,
                                                             T,F,n_windows,n_iters,rp)
                self.performance_dict[well][channel]['mean_mapes'] = mu
                self.performance_dict[well][channel]['std_mapes'] = sigma
                self._plot_accum(mapes_global)
        self._save_JSON()
        return self.performance_dict
                
    
    def _extract_samples(self, processor, best_models_dictionary, channels_dictionary,
                         well, channel, X, stucks_array, non_stucks_array, df, num_pipeline):
        _, _, stucks_array, non_stucks_array, y_stucks, y_non_stucks = processor.prepare_for_specific_channel(
                                        channel, X, df, channels_dictionary[channel], 
                                        stucks_array, non_stucks_array, num_pipeline)
        T = best_models_dictionary[well][channel]['T']
        F = best_models_dictionary[well][channel]['F']
        slider = TimeSlider(T,F)
        stucks_array = slider._adjust_sample_windows(stucks_array)
        non_stucks_array = slider._adjust_sample_windows(non_stucks_array)
        y_stucks= slider._adjust_sample_windows(y_stucks)
        y_non_stucks=slider._adjust_sample_windows(y_non_stucks)
        return stucks_array, non_stucks_array, y_stucks, y_non_stucks, T, F
    
    def _plot_accum(self,array):
      import matplotlib.pyplot as plt
      import seaborn as sb
      import numpy as np
      fig,ax = plt.subplots(figsize=(6,5))
      sb.set_style("ticks")
      sb.histplot(x=array,stat='percent', cumulative=True, color='darkblue', ax=ax)
      font = {'family': 'monospace',
        'color':  'darkred',
        'weight': 'bold',
        'size': 16,
        }
      ax.set_xlabel('MAPE',fontdict=font)
      ax.set_ylabel('Percent of samples',fontdict=font)
      ax.tick_params(axis='both', labelsize=15)
      plt.xlim(0,20)
      plt.xticks(np.arange(0,20,2.5))
      plt.yticks(np.arange(0,110,10));
      
    def _save_JSON(self, filename='results_table_mapes.json'):
        import json
        with open(filename, 'w') as file:
            json.dump(self.performance_dict, file)










