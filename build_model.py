#!/home/metctm1/array/soft/anaconda3/bin/python3
'''
Date: Jun 5, 2021

This is the main script to build the model.

Zhenning LI
'''

import os, logging, logging.config
import pandas as pd
import datetime
import lib, core
import lib.utils as utils

def main_run():
    """ main script """
    
    time_mgr=lib.time_manager.time_manager()
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
    
    utils.write_log('>>Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')
    
    etl_mgr=lib.etl_manager.ETLManager(cfg_hdl)

    utils.write_log('>>Prepare raw data...')
   
    etl_mgr.load_xy()
    
    # construct foreseer  
    foreseer=core.oculus.Oculus(etl_mgr) 
    for idx, lb in enumerate(etl_mgr.Ynames):
        utils.write_log('>>Label name: %s for training...' % lb)
        
        X_train, y_train=etl_mgr.X_train, etl_mgr.Y_train[:,idx]
        X_test, y_test=etl_mgr.X_test, etl_mgr.Y_test[:,idx]
        
        foreseer.train(cfg_hdl, X_train, y_train)
        foreseer.evaluate(
                etl_mgr, X_train, X_test, y_train, y_test, lb)

        if cfg_hdl['CORE'].getboolean('archive_model'):
            utils.write_log('>>Archiving Model and evaluation dict...')
            foreseer.archive(lb)
    
    utils.write_log('>>Building model has been completed successfully!')
    time_mgr.dump()
    


if __name__=='__main__':
    main_run()
