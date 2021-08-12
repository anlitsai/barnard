#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 05:22:35 2019

@author: altsai
"""

import os
import sys
import shutil
import numpy as np
import csv
import time
import math
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
#from astropy.coordinates import ICRS, Galactic, FK4, FK5 # Low-level frames
#from astropy.coordinates import Angle, Latitude, Longitude  # Angles
#from astropy.coordinates import match_coordinates_sky
from astropy.table import Table
from photutils import CircularAperture
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
from photutils import SkyCircularAnnulus
# https://photutils.readthedocs.io/en/stable/aperture.html
#from phot import aperphot
# http://www.mit.edu/~iancross/python/phot.html

import matplotlib.pyplot as plt
import matplotlib.axes as ax
from astropy.io import fits
from astropy.wcs import WCS
#from photutils import DAOStarFinder
#from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

from numpy.polynomial.polynomial import polyfit
from astropy.stats import sigma_clipped_stats
from photutils.psf import IterativelySubtractedPSFPhotometry
from statistics import mode
from astropy.visualization import simple_norm
from photutils.utils import calc_total_error
from astropy.stats import mad_std
import matplotlib.gridspec as gridspec
from photutils.aperture import aperture_photometry

import julian
#from datetime import datetime
#from datetime import timedelta
#from datetime import date
import datetime
#from astropy.stats import sigma_clipped_stats
from scipy import stats




dir_obj='aper_phot'

print('... will generate files in: ./'+dir_obj)
if os.path.exists(dir_obj):
    shutil.rmtree(dir_obj)
os.makedirs(dir_obj,exist_ok=True)



print('-----------------------')



ra_deg=269.50081
dec_deg=4.6849722
r_circle_as=30*u.arcsec
r_inner_as=40*u.arcsec
r_outer_as=50*u.arcsec


ra_pix_210715B=159.78148
dec_pix_210715B=360.68308
ra_bkg_pix_210715B=179.5409
dec_bkg_pix_210715B=335.44882

ra_pix_210716B=220.72169
dec_pix_210716B=344.74605
ra_bkg_pix_210716B=254.1174
dec_bkg_pix_210716B=312.78107

ra_pix_210718B=795.68548
dec_pix_210718B=363.76752
ra_bkg_pix_210718B=815.4449
dec_bkg_pix_210718B=338.53326

r_pix=10.344822

r_in_pix=13.793096
r_out_pix=17.24137
#position= SkyCoord(ra_deg,dec_deg,unit=(u.deg),frame='icrs')    
positions_pix_210715B=[(ra_pix_210715B,dec_pix_210715B),(ra_bkg_pix_210715B,dec_bkg_pix_210715B)]
positions_pix_210716B=[(ra_pix_210716B,dec_pix_210716B),(ra_bkg_pix_210716B,dec_bkg_pix_210716B)]
positions_pix_210718B=[(ra_pix_210718B,dec_pix_210718B),(ra_bkg_pix_210718B,dec_bkg_pix_210718B)]

#print(positions_pix)


#aper_annu_pix=[ra_pix,dec_pix]
#aperture_pix=[ra_]
#apper=[aperture_pix,aper_annu_pix]

#aperture_pix = CircularAperture(positions_pix, r=r_pix)
aperture_pix_210715B = CircularAperture(positions_pix_210715B, r=r_pix)
aperture_pix_210716B = CircularAperture(positions_pix_210716B, r=r_pix)
aperture_pix_210718B = CircularAperture(positions_pix_210718B, r=r_pix)



    

#sys.exit(0)
#aperture=SkyCircularAperture(position, r=r_circle_as)
#aperture=SkyCircularAperture(positions, r=r_circle_as)
#print(aperture)

#aper_annu=SkyCircularAnnulus(position,r_inner_as,r_outer_as)
#print(aper_annu)
#sys.exit(0)
#aper_annu_pix=aper_annu.to_pixel(wcs)



#cmd_search_file_raw='find ./ | grep 2107|grep barnards|grep fit|cut -d / -f4' 
cmd_search_file_raw='find ./ | grep 2107 |grep barnards|grep fit' 
print(cmd_search_file_raw)
#sys.exit(0)
list_file_raw=os.popen(cmd_search_file_raw,"r").read().splitlines() #[0]
print(list_file_raw)

n_file_raw=len(list_file_raw)
print('# of files = ', n_file_raw)
#sys.exit(0)

count_median=np.array([0.]*n_file_raw)

bkg_err=np.array([0.]*n_file_raw)
bkg_err_ratio=np.array([0.]*n_file_raw)

count_target_subtract_bkg=np.array([0.]*n_file_raw)
MJD=np.array([0.]*n_file_raw)
folder=['']*n_file_raw
filename=['']*n_file_raw

            
fig=plt.figure(figsize=(6,4))


for i in range(n_file_raw):
    print('======')
    filename[i]=list_file_raw[i]
    print(i, ')', filename[i])
    hdu=fits.open(filename[i])[0]
    imhead=hdu.header
    imdata=hdu.data
    MJD[i]=imhead['MJD']
#    wcs = WCS(imhead)

    folder_name=filename[i].split('/',-1)[1]
#    print('folder name = ',folder_name)
    folder[i]=folder_name
    

    if '210715B' in folder_name:
        aperture_pix=aperture_pix_210715B
    elif '210716B' in folder_name:
        aperture_pix=aperture_pix_210716B
    elif '210718B' in folder_name:
        aperture_pix=aperture_pix_210718B
    else:
        xx=0
        
#    aperture_pix='aperture_pix_'+folder
#    print(aperture_pix)



    phot_table=aperture_photometry(imdata,aperture_pix)
#    print(phot_table)
    count_target=phot_table['aperture_sum'][0]
    count_bkg=phot_table['aperture_sum'][1]
    count_target_subtract_bkg[i]=count_target-count_bkg
#    print('target - bkg = ',count_target_subtract_bkg[i])




    bkg_mean,bkg_median,bkg_std=sigma_clipped_stats(imdata,sigma=3.)
#    print('... bkg_mean,bkg_median,bkg_std')
#    print('...','%.4f' %bkg_mean,'%.4f' %bkg_median,'%.4f' %bkg_std)
    # bkg_std = sqrt((bkg-bkg_mean)^2/n) = background noise

    count_median[i]=bkg_median   
    imdata[imdata<bkg_median*0.9]=bkg_median
    
    bkg_err_ratio[i]=bkg_std/bkg_median
#    print('... bkg_err_ratio = bkg_std/bkg_median =','%.4f' %bkg_err_ratio[i])    

    err_imdata=bkg_err_ratio[i]*imdata
#    print()    
    
    
    
print('======')


df_out=pd.DataFrame({'MJD':MJD.tolist(),'counts':count_target_subtract_bkg.tolist(),'count_median':count_median,'folder':folder,'filename':filename})
#df_out=pd.DataFrame(columns=['MJD','counts','folder','filename'])
#df_out['MJD']=MJD.tolist()
#df_out['counts']=count_target_subtract_bkg.tolist()
#df_out['folder']=folder.tolist()
#df_out['filename']=filename.tolist()
#df_out=df_out[(np.abs(stats.zscore(df_out['counts']))<3)]
index_count_cut=df_out[df_out['counts']<5000].index
df_out=df_out.drop(index_count_cut).reset_index(drop=True)


file_out='quick_aperture.txt'
df_out.to_csv(file_out,sep='|')


plt.scatter(df_out['MJD'],df_out['counts'])
plt.xlabel('MJD')
plt.ylabel('counts (Barnard - background)')
plt.show()
plt.savefig('quick_aperture.png')    



print('... save plot to quick_aperture.png')
print('... finished ...')
#==============================

