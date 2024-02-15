#!/home/metctm1/array/soft/anaconda3/bin/python3
'''
Date: Nov 9, 2021

This script will rewrite config file and call
build_model to build models for multiple standalone
cases.

Zhenning LI
'''

import os, logging, logging.config
import lib
import lib.utils as utils

def main_run():
    
    """ 
        main pipeline
        1. fetch feature_lib files
        2. loop, modify config, and run build_model
    """
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
 
    Xfiles=os.listdir('./feature_lib')
    Yfiles=os.listdir('./label')
    total_files=len(Xfiles)
    for idx, (xfile, yfile) in enumerate(zip(Xfiles, Yfiles)):
        utils.write_log(
                '********BATCH CONTROL: RUN ON %s vs %s, %d/%d********' % (
                    xfile, yfile, idx+1, total_files))
        cfg_hdl=lib.cfgparser.read_cfg('./conf/config.sample.ini')
        cfg_hdl['INPUT_OPT']['feature_lib_file']=xfile
        cfg_hdl['INPUT_OPT']['label_file']=yfile
        lib.cfgparser.write_cfg(cfg_hdl, './conf/config.ini')
        os.system('python build_model.py')

if __name__=='__main__':
    main_run()
