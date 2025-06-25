# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:46:40 2025

@author: murph
"""

import numpy as np

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