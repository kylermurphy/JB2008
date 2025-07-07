import os, sys
import pandas as pd
import numpy as np

from swifter import set_defaults
set_defaults(
    npartitions=None,
    dask_threshold=1,
    scheduler="processes",
    progress_bar=True,
    progress_bar_desc=None,
    allow_dask_on_strings=False,
    force_parallel=True,
)


jb_path = "D:\\GitHub\\jb2008\\jb2008"
sys.path.append(os.path.dirname(jb_path))

from jb2008.jb2008 import jb2008

def jb08_prof(fn='D:/data/SatDensities/satdrag_database_grace_B_v3.hdf5',
              fo=False,
              alt_min=250,
              alt_max=750,
              columns=['DateTime_gr','lat','lon','alt','dens_x'], 
              chunk_size=20000,
              small_batch=False):

    # get altitudes for generating profiles
    if alt_max < alt_min:
        alt_min=250
        alt_max=750
    alts = np.arange(alt_min,alt_max+1,1)  
    
    if (small_batch) & (not fo):
        fo = 'D:/small_batch_test_jb2008.hdf5'
        print('Small Batch Processing')
    elif not fo:
        fo, f_ext = os.path.splitext(fn)
        fo = fo+'_jb2008'+f_ext
        

    # get itterator for reading
    # in the data 
    gr_it = pd.read_hdf(fn,columns=columns, iterator=True, chunksize=chunk_size)
    
    lp = 0
    for df in gr_it:
        # create mesh grids used for deriving modelled densities
        dt_pre, alt_pre = np.meshgrid(df['DateTime_gr'],alts, indexing = 'ij')
        lat_pre, _ = np.meshgrid(df['lat'],alts, indexing = 'ij') 
        lon_pre, _ = np.meshgrid(df['lon'],alts, indexing = 'xy')
        
        # modeled density profile
        jb_alt = jb2008(t=dt_pre.flatten(),lat=lat_pre.flatten(),
                lon=lat_pre.flatten(),alt=alt_pre.flatten())
        jb_alt.predict()
        
        # modeled density at the satellite
        jb_gr = jb2008(t=df['DateTime_gr'],lat=df['lat'],lon=df['lon'],alt=df['alt']/1000.)
        jb_gr.predict()
        
        # grab the densities
        # reshape the densities for broadcasting
        den_r = np.reshape(jb_alt.dat['DEN'].to_numpy(),dt_pre.shape)
        den_test = den_r[0,:]
        den_comp = jb_alt.dat.iloc[0:len(den_test)]['DEN'].to_numpy()
        
        # make sure the reshaping worked
        if (den_test == den_comp).all():
            print(f'Converting profile: {lp}')
            den_cm = jb_gr.dat['DEN'].to_numpy() # champ modelled density
            den_df = df['dens_x'].to_numpy() # champ density
            den_ratio = den_df/den_cm # ratio between densities
        
            den_pro = (den_ratio*den_r.transpose()).transpose() # corrected model profile
            
            df_wr = pd.DataFrame({'DateTime':dt_pre.flatten(),
                                  'lat': lat_pre.flatten(),
                                  'lon': lon_pre.flatten(),
                                  'alt': alt_pre.flatten(),
                                  'den_cor':den_pro.flatten(),
                                  'den_mod':den_r.flatten()}, 
                                 dtype=np.float32)
            df_wr.reset_index(drop=True).to_hdf(fo, key='profiles',
                                                append=True, format='Table',
                                                complevel=9)
            lp = lp+1

            if small_batch and lp > 1:
                break
        else: 
            print('no no no')
            continue
        
        
    return df_wr, den_cm, den_df, jb_alt, jb_gr
    
p_df, jb_gr, jb_alt =  jb08_prof(chunk_size=1000, small_batch=True)   
        
        
        
        