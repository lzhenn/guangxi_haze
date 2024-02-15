#/usr/bin/env python
"""Data ETL from raw reanalysis/dyn forecast/station data"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import datetime
import xarray as xr
from lib import utils

print_prefix='lib.ETL>>'

class ETLManager():
    '''
    Construct ETL manager 
    
    Attributes
    -----------

    Methods
    '''
 
    def __init__(self, cfg):
        ''' init etl manager '''
        self.label_fn=cfg['INPUT_OPT']['label_file']
        self.feature_lib_file=cfg['INPUT_OPT']['feature_lib_file']
        self.infer_file=cfg['INPUT_OPT']['infer_file']

        self.model_name=cfg['CORE']['model_name']
       
        self.std_label=cfg['CORE'].getboolean('label_standardize')
        self.test_size=float(cfg['CORE']['test_size'])
        self.iteration_times=int(cfg['CORE']['iteration_times'])
        
        self.model_start_time=datetime.datetime.strptime(cfg['INPUT_OPT']['model_strt_time'], '%Y%m%d')
        self.model_end_time=datetime.datetime.strptime(cfg['INPUT_OPT']['model_end_time'], '%Y%m%d')
   
        self.infer_start_time=datetime.datetime.strptime(cfg['INPUT_OPT']['infer_strt_time'], '%Y%m%d')
        self.infer_end_time=datetime.datetime.strptime(cfg['INPUT_OPT']['infer_end_time'], '%Y%m%d')
        # get label list
        self.Ynames=utils.get_lblist(cfg)
        
    def load_xy(self):
        self.load_x()
        self.load_y()
        self.check_xy()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.X, self.Y, test_size=self.test_size, shuffle=False)

    def load_x(self,call_from='train'):
        ''' load feature lib '''
        parser = lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')
        if call_from == 'train':
            flib_all=pd.read_csv(
                    'feature_lib/'+self.feature_lib_file, 
                    parse_dates=True, date_parser=parser, index_col='time')
        elif call_from=='cast':
            flib_all=pd.read_csv(
                    'inferX/'+self.infer_file, 
                    parse_dates=True, date_parser=parser, index_col='time')


        self.select_x(flib_all, call_from)
        
    def select_x(self, flib, call_from):
        ''' select x according to prescribed method '''
        if call_from == 'train':
            strt_time=self.model_start_time
            end_time=self.model_end_time
        elif call_from== 'cast':
            strt_time=self.infer_start_time
            end_time=self.infer_end_time
        X=flib.loc[strt_time:end_time]
        self.Xdate=X.index
        self.X=X.values
        self.Xnames=flib.columns.values.tolist()

    def load_y(self):
        ''' select y according to prescribed method '''
        strt_time=self.model_start_time
        end_time=self.model_end_time
        
        parser = lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')
        lb_all=pd.read_csv(
                'label/'+self.label_fn, 
                parse_dates=True, date_parser=parser, index_col='time')
        
        if self.std_label:
            self.amean, self.astd, lb_all=utils.get_ltm_std(lb_all)
        self.Y=lb_all[strt_time:end_time].values
        self.Ynames=lb_all.columns.values.tolist()

    def check_xy(self):
        if self.X.shape[0] != self.Y.shape[0]:
            utils.throw_error(print_prefix+'Size of dim0 in X and Y does not match!')
        else:
            utils.write_log(print_prefix+'Sample Size:'+str(self.Y.shape[0]))
