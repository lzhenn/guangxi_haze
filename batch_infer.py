#!/home/metctm1/array/soft/anaconda3/bin/python3
'''
Date: Nov 9, 2021

This script will rewrite config file and call
infer_model for multiple standalone cases.

Zhenning LI
'''

import os, logging, logging.config
import lib
import lib.utils as utils

def main_run():
    
    """ 
        main pipeline
        1. fetch inferX files
        2. loop, modify config, and run infer_model
    """
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
 
    Xfiles=os.listdir('./inferX')
    total_files=len(Xfiles)
    for idx, xfile in enumerate(Xfiles):
        utils.write_log(
                '********BATCH CONTROL: RUN ON %s, %d/%d********' % (
                    xfile, idx+1, total_files))
        cfg_hdl=lib.cfgparser.read_cfg('./conf/config.sample.ini')
        cfg_hdl['INPUT_OPT']['infer_file']=xfile
        lib.cfgparser.write_cfg(cfg_hdl, './conf/config.ini')
        os.system('python infer_model.py')

if __name__=='__main__':
    main_run()
