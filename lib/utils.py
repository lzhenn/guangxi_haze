import numpy as np
import pandas as pd
import xarray as xr
import datetime
import logging

def throw_error(msg):
    '''
    throw error and exit
    '''
    logging.error(msg)
    exit()

def write_log(msg, lvl=20):
    '''
    write logging log to log file
    level code:
        CRITICAL    50
        ERROR   40
        WARNING 30
        INFO    20
        DEBUG   10
        NOTSET  0
    '''

    logging.log(lvl, msg)


def get_ltm_std(df):
    return df.mean(), df.std(), (df-df.mean())/df.std()

def get_lblist(cfg):
    ''' seperate vars in cfg varlist csv format '''
    varlist=cfg['INPUT_OPT']['infer_labels'].split(',')
    varlist=[ele.strip() for ele in varlist]
    return varlist

def get_para_int_list(parastr):
    ''' seperate hyperparameters in str into int list '''
    varlist=parastr.split(',')
    varlist=[int(var) for var in varlist]
    return varlist
