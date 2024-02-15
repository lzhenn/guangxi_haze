#!/home/metctm1/array/soft/anaconda3/bin/python3
'''
Date: Jun 5, 2021

This is the main script to load model and perform inference.

Zhenning LI
'''
import os, logging, logging.config
import pandas as pd
import datetime
import lib, core
import lib.utils as utils


def main_run():
    '''main script'''

    time_mgr=lib.time_manager.time_manager()
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
    
    utils.write_log('>>Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')
 
    utils.write_log('>>Prepare raw data...')
    etl_mgr=lib.etl_manager.ETLManager(cfg_hdl)
    etl_mgr.load_x(call_from='cast')
    
    for idx, lb in enumerate(etl_mgr.Ynames):
        foreseer=core.oculus.Oculus(etl_mgr) 
        foreseer.load(lb)
        foreseer.cast(etl_mgr, lb)
     
    utils.write_log('>>Inference has been completed successfully!')
    time_mgr.dump()
 

if __name__=='__main__':
    main_run()

