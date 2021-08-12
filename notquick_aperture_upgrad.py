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



ra_bkg_pix_210715B=[194.46075,200.68155,208.21563,216.92475,224.38427,224.38427,234.33755,238.40763,245.87259,248.36091,253.33755,255.82587,258.31419,265.40763,270.38427,265.40763,270.38427,270.38427,272.87259]
dec_bkg_pix_210715B=[329.66947,329.66947,327.94146,327.94146,325.69282,325.69282,323.20449,322.69282,320.2045,317.71618,315.22786,312.73954,310.25122,308.18114,303.2045,300.71618,295.73954,290.7629,288.27458]
ra_pix_210715B=[161.06343,167.28423,174.81831,183.52743,190.98695,190.98695,200.94023,205.01031,212.47527,214.96359,219.94023,222.42855,224.91687,232.01031,236.98695,232.01031,236.98695,236.98695,239.47527]
dec_pix_210715B=[361.63446,361.63446,359.90645,359.90645,357.65781,357.65781,355.16948,354.65781,352.16949,349.68117,347.19285,344.70453,342.21621,340.14613,335.16949,332.68117,327.70453,322.72789,320.23957]
ra_ref_pix_210715B=[38.87152,45.09232,52.626399,61.335519,68.79504,68.79504,78.748319,82.8184,90.283359,92.771678,97.748317,100.23664,102.72496,109.8184,114.79504,109.8184,114.79504,114.79504,117.28336]
dec_ref_pix_210715B=[168.00978,168.00978,166.28177,166.28177,164.03313,164.03313,161.5448,161.03313,158.54481,156.05649,153.56817,151.07985,148.59153,146.52145,141.54481,139.05649,134.07985,129.10321,126.61489]

#ra_pix_210715B=159.78148
#dec_pix_210715B=360.68308
#ra_bkg_pix_210715B=179.5409
#dec_bkg_pix_210715B=335.44882

'------'


ra_bkg_pix_210716B=[247.1174,247.1174,250.1174]
dec_bkg_pix_210716B=[318.78107,315.78107,312.78107]
ra_pix_210716B=[213.72169,213.72169,216.72169]
dec_pix_210716B=[350.74605,347.74605,344.74605]
ra_ref_pix_210716B=[91.528695,91.528695,94.528695]
dec_ref_pix_210716B=[157.12133,154.12133,151.12133]

'------'


ra_bkg_pix_210718B=[834.0801,843.0801,850.0801,856.0801,860.0801,868.0801,876.0801,883.0801,890.0801,892.0801,896.0801,904.0801,905.0801,904.0801,910.0801,910.0801,913.0801,913.0801,913.0801]
dec_bkg_pix_210718B=[331.44106,329.44106,329.44106,328.24106,328.24106,325.24106,324.24106,320.24106,316.24106,313.24106,311.24106,307.24106,303.24106,299.24106,295.24106,291.24106,286.24106,280.24106,280.24106]
ra_pix_210718B=[800.6578,809.6578,816.6578,822.6578,826.6578,834.6578,842.6578,849.6578,856.6578,858.6578,862.6578,870.6578,871.6578,870.6578,876.6578,876.6578,879.6578,879.6578,879.6578]
dec_pix_210718B=[363.38235,361.38235,361.38235,360.18235,360.18235,357.18235,356.18235,352.18235,348.18235,345.18235,343.18235,339.18235,335.18235,331.18235,327.18235,323.18235,318.18235,312.18235,312.18235]
ra_ref_pix_210718B=[678.61095,687.61095,694.61095,700.61345,704.61345,712.61345,720.61345,727.61345,734.61345,736.61345,740.61345,748.61345,749.61345,748.61345,754.61345,754.61345,757.61345,757.61345,757.61345]
dec_ref_pix_210718B=[169.67521,167.67521,167.67521,166.47353,166.47353,163.47353,162.47353,158.47353,154.47353,151.47353,149.47353,145.47353,141.47353,137.47353,133.47353,129.47353,124.47353,118.47353,118.47353]





#ra_pix_210718B=795.68548
#dec_pix_210718B=363.76752
#ra_bkg_pix_210718B=815.4449
#dec_bkg_pix_210718B=338.53326

r_pix=10.344822

r_in_pix=13.793096
r_out_pix=17.24137
#position= SkyCoord(ra_deg,dec_deg,unit=(u.deg),frame='icrs')    
#positions_pix_210715B=[(ra_pix_210715B,dec_pix_210715B),(ra_bkg_ix_210715B,dec_bkg_pix_210715B)]
#positions_pix_210716B=[(ra_pix_210716B,dec_pix_210716B),(ra_bkg_pix_210716B,dec_bkg_pix_210716B)]
#positions_pix_210718B=[(ra_pix_210718B,dec_pix_210718B),(ra_bkg_pix_210718B,dec_bkg_pix_210718B)]

#print(positions_pix)


#aper_annu_pix=[ra_pix,dec_pix]
#aperture_pix=[ra_]
#apper=[aperture_pix,aper_annu_pix]

#aperture_pix = CircularAperture(positions_pix, r=r_pix)
#aperture_pix_210715B = CircularAperture(positions_pix_210715B, r=r_pix)
#aperture_pix_210716B = CircularAperture(positions_pix_210716B, r=r_pix)
#aperture_pix_210718B = CircularAperture(positions_pix_210718B, r=r_pix)



    

#sys.exit(0)
#aperture=SkyCircularAperture(position, r=r_circle_as)
#aperture=SkyCircularAperture(positions, r=r_circle_as)
#print(aperture)

#aper_annu=SkyCircularAnnulus(position,r_inner_as,r_outer_as)
#print(aper_annu)
#sys.exit(0)
#aper_annu_pix=aper_annu.to_pixel(wcs)



#cmd_search_file_raw='find ./ | grep 2107|grep barnards|grep fit|cut -d / -f4' 
cmd_search_file_raw='find ./ | grep 2107 |grep B|grep barnards|grep fit|sort' 
print(cmd_search_file_raw)
#sys.exit(0)
list_file_raw=os.popen(cmd_search_file_raw,"r").read().splitlines() #[0]
print(list_file_raw)

n_file_raw=len(list_file_raw)
print('# of files = ', n_file_raw)
#sys.exit(0)

cmd_search_n_210715B='find ./ | grep 210715B |grep barnards|grep fit|wc -l' 
print(cmd_search_n_210715B)
n_file_210715B=int(os.popen(cmd_search_n_210715B,"r").read().splitlines()[0])
print('# of files = ', n_file_210715B)      
      
      
      
count_median=np.array([0.]*n_file_raw)

bkg_err=np.array([0.]*n_file_raw)
bkg_err_ratio=np.array([0.]*n_file_raw)

count_target_subtract_bkg=np.array([0.]*n_file_raw)
count_ref_subtract_bkg=np.array([0.]*n_file_raw)
count_bkg=np.array([0.]*n_file_raw)
count_ref=np.array([0.]*n_file_raw)

MJD=np.array([0.]*n_file_raw)
folder=['']*n_file_raw
path_file=['']*n_file_raw

          


for i in range(n_file_raw):
    print('======')
    path_file[i]=list_file_raw[i]
    print(i, '/',n_file_raw, ')', path_file[i])
    hdu=fits.open(path_file[i])[0]
    imhead=hdu.header
    imdata=hdu.data
    MJD[i]=imhead['MJD']
#    wcs = WCS(imhead)

    folder_name=path_file[i].split('/',-1)[1]
#    print('folder name = ',folder_name)
    folder[i]=folder_name
    
    
    fitsname=path_file[i].split('/',-1)[-1]
#    print(fitsname)
    

    if '210715B' in fitsname:
#        print('folder name = ',folder_name)
        number_fits=int(fitsname.split('.',-1)[0].split('_',-1)[-1])
#        for j in range(n_file_210715B):
        number_fits_100=int(number_fits/100) 
#        print(number_fits,number_fits_100)
        ra_pix=ra_pix_210715B[number_fits_100]
        dec_pix=dec_pix_210715B[number_fits_100]
        ra_bkg_pix=ra_bkg_pix_210715B[number_fits_100]
        dec_bkg_pix=dec_bkg_pix_210715B[number_fits_100]
        ra_ref_pix=ra_ref_pix_210715B[number_fits_100]
        dec_ref_pix=dec_ref_pix_210715B[number_fits_100]
#        print(ra_pix,dec_pix)
    elif '210716B' in fitsname:        
        number_fits=int(fitsname.split('.',-1)[0].split('_',-1)[-1])
#        for j in range(n_file_210715B):
        number_fits_100=int(number_fits/100) 
#        print(number_fits,number_fits_100)
        ra_pix=ra_pix_210716B[number_fits_100]
        dec_pix=dec_pix_210716B[number_fits_100]
        ra_bkg_pix=ra_bkg_pix_210716B[number_fits_100]
        dec_bkg_pix=dec_bkg_pix_210716B[number_fits_100]
        ra_ref_pix=ra_ref_pix_210716B[number_fits_100]
        dec_ref_pix=dec_ref_pix_210716B[number_fits_100]
#        print(ra_pix,dec_pix)
    elif '210718B' in fitsname:
        number_fits=int(fitsname.split('.',-1)[0].split('_',-1)[-1])
#        for j in range(n_file_210715B):
        number_fits_100=int(number_fits/100) 
#        print(number_fits,number_fits_100)
        ra_pix=ra_pix_210718B[number_fits_100]
        dec_pix=dec_pix_210718B[number_fits_100]
        ra_bkg_pix=ra_bkg_pix_210718B[number_fits_100]
        dec_bkg_pix=dec_bkg_pix_210718B[number_fits_100]
        ra_ref_pix=ra_ref_pix_210718B[number_fits_100]
        dec_ref_pix=dec_ref_pix_210718B[number_fits_100]
#        print(ra_pix,dec_pix)
    
#    print('dec_bkg_pix = ',dec_bkg_pix)
    
    positions_pix=[(ra_pix,dec_pix),(ra_bkg_pix,dec_bkg_pix),(ra_ref_pix,dec_ref_pix)]
#    print(positions_pix)
    aperture_pix = CircularAperture(positions_pix, r=r_pix)
#    print(aperture_pix)
    


        
#    aperture_pix='aperture_pix_'+folder
#    print(aperture_pix)



    phot_table=aperture_photometry(imdata,aperture_pix)
#    print(phot_table)
    count_target=phot_table['aperture_sum'][0]
    count_bkg[i]=phot_table['aperture_sum'][1]
    count_ref[i]=phot_table['aperture_sum'][2]
    count_target_subtract_bkg[i]=count_target-count_bkg[i]
#    print('target - bkg = ',count_target_subtract_bkg[i])
    count_ref_subtract_bkg[i]=count_ref[i]-count_bkg[i]
    



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


df_out=pd.DataFrame({'MJD':MJD.tolist(),'counts_target':count_target_subtract_bkg.tolist(),'counts_bkg':count_bkg,'counts_ref':count_ref_subtract_bkg.tolist(),'count_median':count_median,'folder':folder,'path_file':path_file})
#df_out=pd.DataFrame(columns=['MJD','counts','folder','path_file'])
#df_out['MJD']=MJD.tolist()
#df_out['counts']=count_target_subtract_bkg.tolist()
#df_out['folder']=folder.tolist()
#df_out['path_file']=path_file.tolist()
#df_out=df_out[(np.abs(stats.zscore(df_out['counts']))<3)]

index_count_target_cut=df_out[df_out['counts_target']<2000].index
df_out=df_out.drop(index_count_target_cut).reset_index(drop=True)


file_out1='quick_aperture.txt'
df_out.to_csv(file_out1,sep='|')


file_out2='Barnard_counts.txt'
df_out2=df_out
df_out2=df_out2.drop(['count_median','path_file','folder'],axis=1)

df_out2.to_csv(file_out2,sep='|')



'''
fig=plt.figure(figsize=(6,4))
plt.scatter(df_out['MJD'],df_out['counts_target'],label='Barnards')
#plt.scatter(df_out['MJD'],df_out['counts_bkg'],label='background')
plt.xlabel('MJD')
plt.ylabel('counts (Barnard - background)')
plt.legend(loc='best')
plt.show()
'''


fig,axs=plt.subplots(3,1,figsize=(6,8))
fig.subplots_adjust(hspace=0.5,wspace=0.5)
#axs=axs.ravel()


folders=['210715B','210716B','210718B']
n_folders=len(folders)

for i in range(n_folders):
    i_folder=folders[i]
     #,secols=[1,2,3]) 
    plt_mjd=df_out[df_out['folder'].str.contains(i_folder)]['MJD']
#    print(JD1)
    plt_counts_target=df_out[df_out['folder'].str.contains(i_folder)]['counts_target']
#    print(Mag)
    plt_counts_ref=df_out[df_out['folder'].str.contains(i_folder)]['counts_ref']



#    axs[i].errorbar(JD0,R0,yerr=eR0,linestyle='--',label='no w',lw=1)
#    axs[i].errorbar(JD,Mag,yerr=err,linestyle='--',lw=1)  
    axs[i].scatter(plt_mjd,plt_counts_target,label='Barnard - bkg') 
    axs[i].scatter(plt_mjd,plt_counts_ref,label='ref. star - bkg') 
    
    axs[i].set_title(i_folder)  

    axs[i].set_xlabel('MJD')
    axs[i].set_ylabel('counts')
#    axs[i].invert_yaxis()
    axs[i].legend(loc='lower right')


plt.savefig('quick_aperture.png')    

print('... save plot to quick_aperture.png')
print('... finished ...')
#==============================

