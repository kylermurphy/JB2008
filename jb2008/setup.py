# -*- coding: utf-8 -*-

import requests


from os import path, makedirs
from pathlib import Path
from datetime import datetime
from utils.utils import dl_file, wf_mtime

def setup():
    
    url1 = 'https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT'
    url2 = 'https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT' 
    
    file1 = 'SOLFSMY.TXT' 
    file2 = 'DTCFILE.TXT'
    
    # find directory of module
    # module directory/swdata/ is where the data is stored
    file_path = Path(__file__).resolve()
    data_path = file_path / '..' / 'swdata'
    data_path = data_path.resolve()
    
    # create file names
    file1 = path.join(data_path,file1)
    file2 = path.join(data_path,file2)
    
    # create it if it doesn't exist
    if not data_path.exists():
       makedirs(data_path)
       
    if not path.exists(file1):
        dl_file(url1,file1)
        dl_file(url2,file2)
    else:
        #check for modification times
        modified_time = datetime.fromtimestamp(path.getmtime(swfile1))
    
        
       
    return file1, file2
       
    

       
    