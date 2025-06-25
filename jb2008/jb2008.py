import pandas as pd
import numpy as np
import numpy.typing as npt

import swifter

from astropy.coordinates import get_sun
from astropy.time import Time

from utils.utils import ydhms_days, vectorize
from utils import Const
from jb2008_subfuncs import jb2008_mod

from tqdm import tqdm
tqdm.pandas()


class jb2008():
    """
    JB2008 emprical global reference atmospheric model.
    
    Models atmospheric density and temperature. 
    
    This version is modified from the PyAtmos module 
    https://github.com/lcx366/ATMOS/tree/master/pyatmos
    
    Here the jb2008 class is vectorized and parallelised for larger
    calculations using swifter.
    
    Reference:
        Bowman, Bruce R., etc. : "A New Empirical Thermospheric
        Density Model JB2008 Using New Solar and Geomagnetic Indices",
        AIAA/AAS 2008, COSPAR CIRA 2008 Model
    
    Note:
        This code has been translated from PyATMOS which was translated
        from the fortran source code written by Bruce R Bowman
        (HQ AFSPC, Space Analysis Division), 2008
    

    Usage
    -----
    jb08 = jb2008(t,lat,lon,alt)
    jb08.predict_den()
    
    jb08.dat.head()

    Parameters
    ----------
    t : npt.ArrayLike, optional
        Time, string or datetime like. 
        The default is ['2001-01-01'].
    lat : npt.ArrayLike, optional
        Latitude in degrees. 
        The default is np.array([0]).
    lon : npt.ArrayLike, optional
        Longitude in degrees. 
        The default is np.array([0]).
    alt : npt.ArrayLike, optional
        Altitude in km. 
        The default is np.array([400]).

    Returns
    -------
    jb08 -> instance of class jb2008, where its attributes include
                                                                    
        dat -> pd.DataFrame
            DataFrame contianing the inputs to the jb2008 model.
            Time, latitude, longitude, and altitude where model predictions
            are derived. 
            jb2008 density and temperature predictions, 
            after predict() has been called.
            
            Note if the position and time do not change but altitude does, the
            exospheric temperature will be constant (ie, fixed with altitude at
            a fixe time and location).
            
    Examples
    --------
    >>> from jb2008 import jb2008
    >>>
    >>> 
    >>> t = '2014-07-22 22:18:45' # time(UTC) 
    >>> lat,lon,alt = 25,102,600 # latitude, longitude in [degree], and altitude in [km]
    >>> jb08 = jb2008(t,lat,lon,alt)
    >>> jb08.predict()
    >>>
    >>>
    >>> print(jb08.dat['DEN']) # [kg/m^3]
    >>> print(jb08.dat['EXO_TEMP') # [K] Exospheric Temperature above Input Position (K)     
    >>> print(jb08.dat['TEMP') # [K] Temperature at Input Position (K)              
    """
    
    def __init__(self, 
                 t: npt.ArrayLike=['2001-01-01'], 
                 lat: npt.ArrayLike=np.array([0]), 
                 lon: npt.ArrayLike=np.array([0]), 
                 alt: npt.ArrayLike=np.array([400])
                 ):
        """
        Initializing the jb2008 class.

        Parameters
        ----------
        t : npt.ArrayLike, optional
            Time, string or datetime like. 
            The default is ['2001-01-01'].
        lat : npt.ArrayLike, optional
            Latitude in degrees. 
            The default is np.array([0]).
        lon : npt.ArrayLike, optional
            Longitude in degrees. 
            The default is np.array([0]).
        alt : npt.ArrayLike, optional
            Altitude in km. 
            The default is np.array([400]).
        """
        
        
        #data directory
        self.direc = 'D:\\GitHub\\jb2008\\jb2008\\'
        t = pd.to_datetime(t)
        t = vectorize(t)
        lat = vectorize(lat)
        lon = vectorize(lon)
        alt = vectorize(alt)
        
        #vectorize/grid the data if the inputs are
        #not the same length       
        if not len(t) == len(lat) == len(lon) == len(alt):
            t, lat, lon, alt = np.meshgrid(t,lat,lon,alt)
            t = t.flatten()
            lat = lat.flatten().astype('float32')
            lon = lon.flatten().astype('float32')
            alt = alt.flatten().astype('float32')        
        
        
        #setup arrays we need
        t_ob = Time(t,location=(lon,lat))
        AMJD = t_ob.mjd
        
        #convert each time to decimal day
        YRDAY = ydhms_days(t_ob)
        
        #get sun position
        sunpos = get_sun(t_ob)
        
        #get satellite position
        sat_ra = t_ob.sidereal_time('mean').rad
        
        # load the swdata and get the required inputs
        swdata = self.read_sw()
        swinput = np.array([self.get_sw(swdata,dt) for dt in AMJD])
        
        # create a dataframe of all the input data
        # this will allow for some parallel computing using 
        # pandas and swifter
        
        # SUN = (sunpos.ra.rad,sunpos.dec.rad) -> tuple
        # SAT = (sat_ra,np.deg2rad(lat),alt) -> tuple
        cols = ['F10','F10B','S10','S10B','M10','M10B','Y10','Y10B','DTCVAL'] 
        jb_df = pd.DataFrame(swinput,
                             columns=cols, dtype='float32')
        
        jb_df['AMJD'] = AMJD
        jb_df['YRDAY'] = YRDAY
        jb_df['SUN_RA'] = np.array(sunpos.ra.rad, dtype='float32')
        jb_df['SUN_DEC'] = np.array(sunpos.dec.rad, dtype='float32')
        jb_df['SAT_RA'] = np.array(sat_ra, dtype='float32')
        jb_df['SAT_LAT'] = np.deg2rad(lat, dtype='float32')
        jb_df['SAT_ALT'] = alt
        
        self.dat = jb_df
        
    
    def predict(self):
        """
        Predict density and temperatures.
        
        For small number of predictions (<1000) use pandas apply.
        For larger number of predictions use swifter to determine fastest
            method. 

        Returns
        -------
        Adds DEN, TEMP, EXO_TEMP to the dat DataFrame
        
        """
        
        if self.dat.shape[0] < 1000:
            self.dat[['DEN','TEMP','EXO_TEMP']] = \
                self.dat.progress_apply(lambda x : 
                                        self.den_temp(x.AMJD, x.YRDAY,
                                        (x.SUN_RA,x.SUN_DEC),
                                        (x.SAT_RA,x.SAT_LAT,x.SAT_ALT),
                                        x.F10,x.F10B,x.S10,x.S10B,
                                        x.M10,x.M10B,x.Y10,x.Y10B,
                                        x.DTCVAL),
                                        axis=1, result_type="expand")
        
        else:
            self.dat[['DEN','TEMP','EXO_TEMP']] = \
                self.dat.swifter.apply(lambda x : 
                                       self.den_temp(x.AMJD, x.YRDAY,
                                       (x.SUN_RA,x.SUN_DEC),
                                       (x.SAT_RA,x.SAT_LAT,x.SAT_ALT),
                                       x.F10,x.F10B,x.S10,x.S10B,
                                       x.M10,x.M10B,x.Y10,x.Y10B,
                                       x.DTCVAL),
                                       axis=1, result_type="expand")
                
        
                                        

    
    def den_temp(self,AMJD,YRDAY,SUN,SAT,F10,F10B,S10,S10B,
                           M10,M10B,Y10,Y10B,DSTDTC):
        """
        Wrapper for the jb2008 model function.
        
        Required to use pandas apply function without falling over
        when an exception occurs.

        Parameters
        ----------
        AMJD : float
            Date and Time, in modified Julian Days and Fraction 
            (MJD = JD-2400000.5)
        YRDAY : float
            Decimal Day of Year.
        SUN : list/tuple
            Location of sun.
            SUN[0] - Right Ascension of Sun (radians)
            SUN[1] - Declination of Sun (radians)
        SAT : list/tuple
            Location of prediction
            SAT[0] - Right Ascension of Position (radians) 
            SAT[1] - Geocentric Latitude of Position (radians)
            SAT[2] - Height of Position (km)
        F10 : float
            10.7-cm Solar Flux (1.0E-22*W/(M**2*Hz)) (Tabular time 
            1.0 day earlier).
        F10B : float
            10.7-cm Solar Flux, ave. 81-day centered on the input time 
            (Tabular time 1.0 day earlier)
        S10 : float
            EUV index (26-34 nm) scaled to F10 (Tabular time 1.0 day earlier)
        S10B : float
            EUV 81-day ave. centered index (Tabular time 1.0 day earlier)
        M10 : float
            MG2 index scaled to F10 (Tabular time 2.0 days earlier)
        M10B : float
            MG2 81-day ave. centered index (Tabular time 2.0 days earlier)
        Y10 : float
            Solar X-Ray & Lya index scaled to F10 (Tabular time 5.0 days
            earlier)
        Y10B : float
            Solar X-Ray & Lya 81-day ave. centered index (Tabular time 5.0 
            days earlier)
        DTCVAL : float
            Temperature change computed from Dst index

        Returns
        -------
        list
            Modelled density, temperature, and exospheric temperature at
            SAT location.
            
            Density - kg/m^3
            Temperatures - K

        """
        
        try:
            temp, den = jb2008_mod(AMJD,YRDAY,SUN,SAT,F10,F10B,S10,S10B,
                                   M10,M10B,Y10,Y10B,DSTDTC)
        except:
            den = np.nan
            temp = [np.nan,np.nan]
            
        return [np.float32(den), np.float32(temp[1]), np.float32(temp[0])]
        
    
    
    def get_sw(self, sw_data,t_mjd):
        """
        Extract the necessary parameters describing the solar activity 
        and geomagnetic activity from the space weather data.

        Parameters
        ----------
        sw_data : list of 2 numpy arrays.
            Space Weather data read in from read_sw().
            [Solar Data Array, Geomagnetic Data Array]
        t_mjd : TYPE
            Time point to extract the neccessary data for.

        Returns
        -------
        F10 : float
            10.7-cm Solar Flux (1.0E-22*W/(M**2*Hz)) (Tabular time 1.0 day earlier).
        F10B : float
            10.7-cm Solar Flux, ave. 81-day centered on the input time (Tabular time 1.0 day earlier)
        S10 : float
            EUV index (26-34 nm) scaled to F10 (Tabular time 1.0 day earlier)
        S10B : float
            EUV 81-day ave. centered index (Tabular time 1.0 day earlier)
        M10 : float
            MG2 index scaled to F10 (Tabular time 2.0 days earlier)
        M10B : float
            MG2 81-day ave. centered index (Tabular time 2.0 days earlier)
        Y10 : float
            Solar X-Ray & Lya index scaled to F10 (Tabular time 5.0 days earlier)
        Y10B : float
            Solar X-Ray & Lya 81-day ave. centered index (Tabular time 5.0 days earlier)
        DTCVAL : float
            Temperature change computed from Dst index
        """

        sw_data1,sw_data2 = sw_data
        sw_mjd = sw_data1[:,0] - 2400000.5
        J_, = np.where(sw_mjd-0.5 < t_mjd)
        j = J_[-1]
    
        # USE 1 DAY LAG FOR F10 AND S10 FOR JB2008
        dlag = j-1
        F10,F10B,S10,S10B = sw_data1[dlag,1:5] 
    
        # USE 2 DAY LAG FOR M10 FOR JB2008
        dlag = j-2
        M10,M10B = sw_data1[dlag,5:7] 
    
        # USE 5 DAY LAG FOR Y10 FOR JB2008
        dlag = j-5
        Y10,Y10B = sw_data1[dlag,7:9] 
        
        t_dmjd = t_mjd - sw_mjd[j] + 0.5
        x = Const.x
        y = sw_data2[j:j+2].flatten()
        DTCVAL = np.interp(t_dmjd,x,y)  
            
        return F10,F10B,S10,S10B,M10,M10B,Y10,Y10B,DTCVAL
    
    def read_sw(self):
        """
        Read and Parse Solar and Geomagnetic Data for model.
        
        Returns
        -------
        sw_data1 : numpy array
            Numpy Array contianing solar data: F10, S10, M10, Y10.
        sw_data2 : numpy array
            Numpy Array continaing geomagnetic data

        """
        
        swfile1 = self.direc + 'SOLFSMY.TXT'
        swfile2 = self.direc + 'DTCFILE.TXT'
        sw_data1 = np.loadtxt(swfile1,usecols=range(2,11))
        sw_data2 = np.loadtxt(swfile2,usecols=range(3,27),dtype=int)
    
        return (sw_data1,sw_data2)
        
