# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:46:40 2025

@author: murph
"""

import numpy as np
import requests
import pathlib
import functools
import shutil
from datetime import datetime
from dateutil import tz

from tqdm import tqdm



def ydhms_days(t_ob):
    '''
    Convert the form of [year,doy,hour,min,sec] to days

    Uasge:
    days = ydhms_das(t_ob)
    '''
    
    t_arr = np.array([np.array(t.yday.split(':'),dtype=float)
             for t in t_ob])
    
    days = t_arr[:,1]+t_arr[:,2]/24. + t_arr[:,3]/1440. + t_arr[:,4]/86400. - 1
    
    return days



def vectorize(x):
    '''
    Vectorize a number(int, float) or a list to a numpy array.
    '''
    try:
        n = len(x)
        x = np.array(x)
    except:
        x = np.array([x])
    return x 



def dl_file(url, filename):
    """
    Download file from url to filename

    Parameters
    ----------
    url : url to file.
    filename : filename to save as.

    Raises
    ------
    for
        DESCRIPTION.
    RuntimeError
        DESCRIPTION.

    """
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))
    
    # if requests couldn't get file size drop encoding to get it
    if file_size == 0:
        rs = requests.head(url, headers={'Accept-Encoding': None})
        file_size = int(rs.headers.get('Content-Length', 0))
        
    
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    desc = "(Unknown total file size)" if file_size == 0 else ""
    post = f'Downloading: {url} to {filename}'
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc, postfix=post, position=0, leave=True) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
            
def wf_mtime(url):
    """
    Retrieves the last modified date of a file on the web.

    Args:
        url (str): The URL of the file.

    Returns:
        datetime or None: The last modified date as a datetime object,
                          or None if the header is not found or an error occurs.
    """
    try:
        response = requests.head(url)  # Use HEAD request for efficiency
        response.raise_for_status()  # Raise an exception for bad status codes

        last_modified_header = response.headers.get('Last-Modified')

        if last_modified_header:
            # Parse the date string from the header
            # Example format: 'Wed, 21 Oct 2015 07:28:00 GMT'
            return datetime.strptime(
                last_modified_header, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=tz.gettz('GMT'))
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None