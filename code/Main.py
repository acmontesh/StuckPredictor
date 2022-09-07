# -*- coding: utf-8 -*-


path_list_timesummary = [r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\FLORENA AP13',
                            r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\FLORENA I9',
                            r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\FLORENA IP10',
                            r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\Floreña TP12',
                            r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\Pauto Sur CP8',
                            r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\Pauto Sur Cp10',
                            r'C:\Users\abrah\OneDrive\Maestría en Sistemas\TRABAJO DE GRADO 1\DATA\PautoJ7']

wells_dictionary = {

     'flo-ap13':{
             'path':path_list_timesummary[0],
             'mudlogging_provider':'baker',
             'stucks_list':[('2014-08-15 00:07', '2014-08-16 08:45'), 
                            ('2014-08-18 22:52', '2014-08-20 04:00')],
             'others_list':[('2014-08-28 03:00', '2014-08-28 06:00')],
             'shoe':6200,
             'beginend_dates':('2014-08-06 13:45','2014-09-10 13:45')
         },
     
     'flo-tp12':{
             'path':path_list_timesummary[3],
             'mudlogging_provider':'baker',
             'stucks_list':[('2015-01-12 18:53', '2015-01-14 18:00')],
             'others_list':[('2015-01-16 06:50', '2015-01-16 10:20')],
             'shoe':15650,
             'beginend_dates':('2014-11-22 16:00','2015-02-20 15:30')
         },
     
     'psc-8':{
             'path':path_list_timesummary[4],
             'mudlogging_provider':'baker',
             'stucks_list':[('2013-08-15 00:40', '2013-08-18 22:00')],
             'others_list':[],
             'shoe':14950,
             'beginend_dates':('2013-07-28 10:45','2013-10-13 08:00')
         },
     
     'pscp-10':{
             'path':path_list_timesummary[5],
            'mudlogging_provider':'baker',
             'stucks_list':[('2015-11-23 14:25', '2015-11-24 04:35'), 
                           ('2015-11-27 14:55', '2015-11-28 22:00'),
                            ('2015-11-29 12:29','2015-11-29 13:20'),
                            ('2015-11-29 17:27','2015-11-30 13:50'),
                            ('2015-11-14 10:57','2015-11-14 16:35'),
                            ('2015-11-17 20:29','2015-11-18 16:35'),
                            ('2015-12-05 20:31','2015-12-07 22:50'),
                            ('2015-12-14 13:31','2015-12-14 14:20')],
             'others_list':[],
            'shoe':6500,
             'beginend_dates':('2015-10-19 00:00','2015-12-17 01:30')
         },
    

    'p-j7':{
            'path':path_list_timesummary[6],
            'mudlogging_provider':'baker',
            'stucks_list':[('2012-12-27 19:23', '2012-12-27 23:00'), 
                           ('2012-12-31 15:37', '2013-01-22 10:00'),
                           ('2013-02-25 04:56','2013-02-25 09:50')],
            'others_list':[('2013-01-24 12:00', '2013-02-13 20:15')],
            'shoe':12700,
            'beginend_dates':('2012-12-16 12:00','2013-03-05 20:45')
        }
    
    }

channels_dictionary= {
            "spp":{'specific_sensors_list':['hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt']},
            "tq":{'specific_sensors_list':['hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt', 'drpm_dt']},
            "hkl":{'specific_sensors_list':['hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt', 'dhkl_dt', 'bit_depth']}
                    }

from TDG_classes_main import SupervisedErlenmeyer, RecurrentErlenmeyer, MapeCalculator

# =============================================================================
# 
#   EXPERIMENTS ON BEST RECURRENT PREDICTORS OF "NORMAL" CONDITIONS (NON-STUCK)
# 
# 
# =============================================================================

re = RecurrentErlenmeyer()
experiments_dictionary = {
                            'models': {
                                        'model':['srnn','lstm','gru'],
                                        'params':[(1,10),(2,5)] #(1,5)
                                        },
                            'windows':[(6,6),(10,6)]  #(10,10) ,(6,10),
                            }
re.run_experiments(experiments_dictionary, wells_dictionary, 
                        channels_dictionary, df_status='pristine', epochs=25)
pr1 = re.get_performance_record()






channels_dictionary= {
            "spp":['hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt'],
            "tq":['hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt', 'drpm_dt'],
            "hkl":['hkl', 'rpm', 'gpm', 'cp1_gr', 'cp2_gr', 'tq', 'dtq_dt', 'dspp_dt', 'spp', 'dgpm_dt', 'dhkl_dt', 'bit_depth']
                    }

# =============================================================================
# 
#    SUPERVISED CLASSIFICATION (ANOMALY DETECTION)
# 
# =============================================================================

se = SupervisedErlenmeyer()
experiments_dictionary = {
                            'models': {
                                        'model':['svm','rf','dt'],
                                        'params':{
                                                    'C':[1,10,100,1000],
                                                    'gamma':[1,2,3,4],
                                                    'kernel':['rbf','linear','poly'],
                                                    'max_depth':[5,10,20],
                                                    'n_trees':[50,100,500],
                                                    'max_depth_tree':[5,10,20]
                                                    }
                                        },
                            'augmentation_factor':[1,2,3,5,10],
                            'augmentation_type':['replication','noise_injection', 'combination'],
                            'aug_gaussian_noise':[(1.05, 0.03)]
                            }

se.run_experiments(experiments_dictionary, wells_dictionary, channels_dictionary)
pr2 = se.get_performance_record()


# =============================================================================
# 
#    MAPE CALCULATION FOR RECURRENT PREDICTORS
# 
# =============================================================================

from TDG_classes_main import MapeCalculator
mc= MapeCalculator()
performance_dict=mc.calculate_MAPE_wells(wells_dictionary, channels_dictionary, n_windows=5, n_iters=100)