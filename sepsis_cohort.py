"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MIMIC-III Sepsis Cohort Extraction.

sourced from: 
https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_sepsis3_def_160219.m
IDENTIFIES THE COHORT OF PATIENTS WITH SEPSIS in MIMIC-III as used in the AI Clinician (Komorowski, et al [Nature, 2018])
(c) Matthieu Komorowski, Imperial College London 2015-2019

Adapted to Python, and minimally modified, by Jayakumar Subramanian and Taylor Killian
        
GENERATES:
    # MIMICraw = MIMIC RAW DATA m*47 array with columns in right order
    # MIMICzs = MIMIC ZSCORED m*47 array with columns in right order, matching MIMICraw

PURPOSE:
------------------------------
This creates a list of icustayIDs of patients who develop sepsis at some point 
in the ICU. records charttime for onset of sepsis. Uses sepsis3 criteria

STEPS: 
There are two phases of the following procedure: 
  - First to compute the SOFA scores for each patient present in the extracted .csv files
  - Second, recompute the reformatting and filling of missing values with only the presumed septic patients
% -------------------------------
% IMPORT DATA FROM CSV FILES
% FLAG PRESUMED INFECTION
% PREPROCESSING
% REFORMAT in 4h time slots
% COMPUTE SOFA at each time step
% FLAG SEPSIS

note: the process generates the same features as the final MDP dataset, most of which are not used to compute SOFA
External files required: Reflabs, Refvitals, sample_and_hold (all saved in the ReferenceFiles folder)

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the license for details. 

Note: The size of the cohort will depend on which version of MIMIC-III is used.
The original cohort from the 2018 Nature Medicine publication was built using MIMIC-III v1.3.
"""

import argparse
import pyprind

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy import stats

from fancyimpute import KNN

parser = argparse.ArgumentParser()
parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
parser.add_argument("--save_intermediate", action="store_true", help="If specified, save off intermediate tables used to construct final patient table")
pargs = parser.parse_args()

print('Loading processed files created from database using "preprocess.py"')
abx           = pd.read_csv('processed_files/abx.csv',           sep = '|')
culture       = pd.read_csv('processed_files/culture.csv',       sep = '|')
microbio      = pd.read_csv('processed_files/microbio.csv',      sep = '|')
demog         = pd.read_csv('processed_files/demog.csv',         sep = '|')
ce010         = pd.read_csv('processed_files/ce010000.csv',      sep = '|')
ce1020        = pd.read_csv('processed_files/ce1000020000.csv',  sep = '|')
ce2030        = pd.read_csv('processed_files/ce2000030000.csv',  sep = '|')
ce3040        = pd.read_csv('processed_files/ce3000040000.csv',  sep = '|')
ce4050        = pd.read_csv('processed_files/ce4000050000.csv',  sep = '|')
ce5060        = pd.read_csv('processed_files/ce5000060000.csv',  sep = '|')
ce6070        = pd.read_csv('processed_files/ce6000070000.csv',  sep = '|')
ce7080        = pd.read_csv('processed_files/ce7000080000.csv',  sep = '|')
ce8090        = pd.read_csv('processed_files/ce8000090000.csv',  sep = '|')
ce90100       = pd.read_csv('processed_files/ce90000100000.csv', sep = '|')
MV            = pd.read_csv('processed_files/mechvent.csv',      sep = '|')
inputpreadm   = pd.read_csv('processed_files/preadm_fluid.csv',  sep = '|')
inputMV       = pd.read_csv('processed_files/fluid_mv.csv',      sep = '|')
inputCV       = pd.read_csv('processed_files/fluid_cv.csv',      sep = '|')
vasoMV        = pd.read_csv('processed_files/vaso_mv.csv',       sep = '|')
vasoCV        = pd.read_csv('processed_files/vaso_cv.csv',       sep = '|')
UOpreadm      = pd.read_csv('processed_files/preadm_uo.csv',     sep = '|')
UO            = pd.read_csv('processed_files/uo.csv',            sep = '|')
labU          = [pd.read_csv('processed_files/labs_ce.csv', sep = '|') , pd.read_csv('processed_files/labs_le.csv', sep = '|')]

labU[1].rename(columns = {'timestp': 'charttime'}, inplace=True) 
labU = pd.concat(labU, sort=False, ignore_index=True)


# Initial data manipulations
microbio['charttime'] = microbio['charttime'].fillna(microbio['chartdate'])
del microbio['chartdate']
bacterio = pd.concat([microbio, culture], sort=False, ignore_index=True)

demog['morta_90'].fillna(0, inplace=True)
demog['morta_hosp'].fillna(0, inplace=True)
demog['elixhauser'].fillna(0, inplace=True)

# Keep only the first icustay of an admission (CRITICAL FIX FROM MATLAB CODE)
demog = demog.drop_duplicates(subset=['admittime','dischtime'],keep='first')

# Get list of all icustayids since that's what we iterate over through the rest of this script
# The old code had a continuous range of icustayids so it was easy to loop through them with a range(numIDS),
# Since we're only keeping the first icustay of a patient's admission, this is now different...
icustayidlist = list(demog.icustay_id.values)

# Calculate the accurate readmission using the demographics data 
# (the SQL code from Komorowski, et al incorrectly cumulatively counts how many icu stays each patient has (preprocess.py:line 414) 
# and does a coarse boolean check if this number is >1). A readmission is now correctly defined by 
# whether the patient has returned to the ICU within 30 days of being previously discharged.

# This is done by grouping all the discharge times for each patient and using them in a comparison 
# with the current row's admission time to see if it's within the 30 day cutoff
subj_dischtime_list = demog.sort_values(by='admittime').groupby('subject_id').apply(lambda df: np.unique(df.dischtime.values)) # Create list of discharge times for each patient (output is a dict keyed by 'subject_id')

def determine_readmission(s, dischtimes=subj_dischtime_list,cutoff=3600*24*30):
    '''
    determine_readmisson evaluates each row of the provided dataframe (designed to operate on the demographics table)
    and chooses whether the current admission occurs within the cutoff of the previous discharge 
    (here, cutoff=30 days is the default)
    '''
    subject, admission, discharge = s[['subject_id','admittime','dischtime']]
    
    # Check for readmission
    subj_stay_idx = np.where(dischtimes[subject]==discharge)[0][0]
    s['re_admission'] = 0
    if subj_stay_idx > 0:
        if (admission - dischtimes[subject][subj_stay_idx-1]) <= cutoff:
            s['re_admission'] = 1
            
    return s
# Apply the above function to determine the appropriate readmissions
demog = demog.apply(determine_readmission,axis=1)

########################################################################
#                    ADDITIONAL HELPER FUNCTIONS
########################################################################

def SAH(input, vitalslab_hold, adjust=0):
    '''Matthieu Komorowski - Imperial College London 2017 
    will copy a value in the rows below if the missing values are within the
    hold period for this variable (e.g. 48h for weight, 2h for HR...)
    vitalslab_hold = 2x55 cell (with row1 = strings of names ; row 2 = hold time)'''
    temp = np.copy(input)
    hold = vitalslab_hold.values[0, :]
    nrow, ncol = temp.shape

    lastcharttime = np.zeros(ncol)
    lastvalue = np.zeros(ncol)
    oldstayid = temp[0, 1]

    bar_SAH = pyprind.ProgBar(ncol-(3+adjust))
    for i in range(3+adjust,ncol):
        bar_SAH.update()
        for j in range(nrow):
            if oldstayid != temp[j, 1]:
                lastcharttime = np.zeros(ncol)
                lastvalue = np.zeros(ncol)
                oldstayid = temp[j, 1]
            if not np.isnan(temp[j, i]):
                lastcharttime[i] = temp[j, 2]
                lastvalue[i] = temp[j, i]
            if j > 0:
                if (np.isnan(temp[j, i])) and (temp[j, 1] == oldstayid) and ((temp[j, 2] - lastcharttime[i]) <= hold[i-(3+adjust)]*3600):
                    temp[j,i] = lastvalue[i]
    return temp

def fixgaps(x):
    '''FIXGAPS Linearly interpolates gaps in a time series
    YOUT=FIXGAPS(YIN) linearly interpolates over NaN
    in the input time series (may be complex), but ignores
    trailing and leading NaN.
    R. Pawlowicz 6/Nov/99'''
    y = np.copy(x)
    bd = np.isnan(x)
    gd = np.arange(len(x))[~bd]
    bd[:min(gd)] = False
    bd[max(gd)+1:] = False
    y[bd] = interp1d(gd,x[gd])(np.arange(len(x))[bd])
    return y

def deloutabove(a, col_no, a_max):
    a[a[:,col_no] > a_max, col_no] = np.nan 
    return a

def deloutbelow(a, col_no, a_min):
    a[a[:,col_no] < a_min, col_no] = np.nan 
    return a

# Compute normalized rate of infusion
# if 100 ml of hypertonic fluid (600 mosm/l) is given at 100 ml/h (given in 1h) it is 200 ml of NS equivalent
# so the normalized rate of infusion is 200 ml/h (different volume in same duration)
inputMV['norm_rate_of_infusion'] = inputMV['tev']*inputMV['rate']/inputMV['amount']

# Fill-in missing ICUSTAY IDs in bacterio
print('Filling-in missing ICUSTAY IDs in bacterio')
bar = pyprind.ProgBar(len(bacterio.index.tolist()))
# Raw Translation
for i in bacterio.index.tolist():
    bar.update()
    if np.isnan(bacterio.loc[i, 'icustay_id']):
        o         = bacterio.loc[i, 'charttime'] 
        subjectid = bacterio.loc[i, 'subject_id']
        hadmid    = bacterio.loc[i, 'hadm_id']
        ii        = demog.index[demog['subject_id'] == subjectid].tolist()
        jj        = demog.index[(demog['subject_id'] == subjectid) & (demog['hadm_id'] == hadmid)].tolist()
        for j in range(len(ii)):
            if (o >= demog.loc[ii[j], 'intime'] - 48*3600) and (o <= demog.loc[ii[j], 'outtime'] + 48*3600):
                bacterio.loc[i,'icustay_id'] = demog.loc[ii[j], 'icustay_id']
            elif len(ii)==1:   # If we cant confirm from admission and discharge time but there is only 1 admission: it's the one!!
                bacterio.loc[i,'icustay_id'] = demog.loc[ii[j], 'icustay_id']

print('Filling-in missing ICUSTAY IDs in bacterio - 2')                
bar = pyprind.ProgBar(len(bacterio.index.tolist()))
for i in bacterio.index.tolist():
    bar.update()
    if np.isnan(bacterio.loc[i, 'icustay_id']):
        subjectid = bacterio.loc[i, 'subject_id']
        hadmid    = bacterio.loc[i, 'hadm_id']
        jj        = demog.index[(demog['subject_id'] == subjectid) & (demog['hadm_id'] == hadmid)].tolist()
        if len(jj) == 1:
            bacterio.loc[i,'icustay_id'] = demog.loc[jj[0], 'icustay_id']

# Fill-in missing ICUSTAY IDs in Antibiotics administration
print('Filling-in missing ICUSTAY IDs in ABx')
bar = pyprind.ProgBar(len(abx.index.tolist()))
for i in abx.index.tolist():
    bar.update()
    if np.isnan(abx.loc[i,'icustay_id']):
        o      = abx.loc[i,'startdate']  #time of event
        hadmid = abx.loc[i,'hadm_id']
        ii     = demog.index[demog['hadm_id'] == hadmid].tolist()
        for j in range(len(ii)):
            if o >= demog.loc[ii[j],'intime'] - 48*3600 and o <= demog.loc[ii[j], 'outtime'] + 48*3600:
                abx.loc[i, 'icustay_id'] = demog.loc[ii[j], 'icustay_id']
            elif len(ii) == 1:   #if we cant confirm from admission and discharge time but there is only 1 admission: it's the one!!
                abx.loc[i, 'icustay_id'] = demog.loc[ii[j], 'icustay_id']


########################################################################
#   Find presumed onset of infection according to sepsis3 guidelines
########################################################################

# METHOD:
# Loop through all administered antibiotics as soon as
# a sample is present within the time window break the loop.

print('Full ICU -- Finding presumed onset of infection according to sepsis3 guidelines')

onset = dict()
num_onset = 0
bar = pyprind.ProgBar(len(icustayidlist))
for icustayid in icustayidlist:
    bar.update()
    onset[icustayid] = np.zeros(3)
    ab = abx.loc[abx['icustay_id'] == icustayid, 'startdate']   # Start time of abx for this icustayid
    bact = bacterio.loc[bacterio['icustay_id'] == icustayid, 'charttime']   # Time of sample
    subj_bact = bacterio.loc[bacterio['icustay_id'] == icustayid,'subject_id'] 
    
    if len(ab) > 0 and len(bact) > 0:   # If we have data for both: proceed
        # Pairwise distances between antibiotic adminstration and requested cultures, in hours
        D = cdist(ab.values.reshape(ab.values.shape[0],1),bact.values.reshape(bact.values.shape[0],1))/3600  
        for i in range(D.shape[0]):  #looping through all rows of adminsitered antibiotics, from early to late
            M, I = np.min(D[i,:]), np.argmin(D[i,:])        # minimum distance in this row
            ab1 = ab.iloc[i]       # timestamp of this value in list of antibiotics
            bact1 = bact.iloc[I]   # timestamp in list of cultures
            if M <= 24 and ab1 <= bact1:      # if ab was first and delay < 24h
                onset[icustayid][0] = subj_bact.iloc[0]
                onset[icustayid][1] = icustayid  
                onset[icustayid][2] = ab1     # Onset of infection = abx time
                num_onset += 1
                break
            elif M <= 72 and ab1 >= bact1:    # elseif sample was first and delay < 72h
                onset[icustayid][0] = subj_bact.iloc[0]   
                onset[icustayid][1] = icustayid
                onset[icustayid][2] = bact1       # Onset of infection = sample time
                num_onset += 1
                break

# Sum of records found
print('Full ICU -- Number of preliminary, presumed septic trajectories: ', num_onset)


# Replacing item_ids with column numbers from reference tables
print('Full ICU -- Replacing item_ids with column numbers from reference tables')

# Replace itemid in labs with column number
# This will accelerate process later
Reflabs = pd.read_csv("ReferenceFiles/Reflabs.tsv", sep = '\t', header=None)
Reflabs_values = np.unique(Reflabs.fillna(-10000))[1:]
Reflabs_id_dict = {}
for r in Reflabs_values:
    try:
        Reflabs_id_dict[r] = np.max(np.where(Reflabs.values == r)[0]) + 1 # for row: +1 due to Index correction python        
    except:
        print(r)
        break
itemid_col = labU.columns.tolist().index('itemid')
labU_temp = labU.values
for c in range(labU_temp.shape[0]):
    labU_temp[c,itemid_col] = Reflabs_id_dict[labU_temp[c,itemid_col]]
for i, c in enumerate(labU.columns.tolist()):
    labU.loc[:,c] = labU_temp[:,i]


# Replace itemid in vitals with col number
Refvitals = pd.read_csv("ReferenceFiles/Refvitals.tsv", sep = '\t', header=None)
Refvitals_values = np.unique(Refvitals.fillna(-10000))[1:]
Refvitals_id_dict = {}
for r in Refvitals_values:
    Refvitals_id_dict[r] = np.max(np.where(Refvitals.values == r)[0]) + 1 # +1 due to index correction for Python from MATLAB
ce_dfs = [ce010, ce1020, ce2030, ce3040, ce4050, ce5060, ce6070, ce7080, ce8090, ce90100]
for ce_df in ce_dfs:
    itemid_col = ce_df.columns.tolist().index('itemid')
    ce_df_temp = ce_df.values
    for c in range(ce_df_temp.shape[0]):
        ce_df_temp[c,itemid_col] = Refvitals_id_dict[ce_df_temp[c,itemid_col]]
    for i, c in enumerate(ce_df.columns.tolist()):
        ce_df.loc[:,c] = ce_df_temp[:,i]

# ########################################################################
#           INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT
# ########################################################################

print(' Full ICU --  Making an array with all unique charttime (1 per row) and all items in columns.')
reformat = np.nan*np.ones((2000000,68))  # Final table 
qstime = dict()
winb4 = 25   # Lower limit for inclusion of data (24h before time flag)
winaft = 49  # Upper limit (48h after)
irow = 0  # Recording row for summary table
bar = pyprind.ProgBar(len(icustayidlist))
for icustayid in icustayidlist:
    qstime[icustayid] = np.zeros(4)
    bar.update()
    qst = onset[icustayid][2] #flag for presumed infection
    if qst > 0:  # if we have a flag
        d1 = demog.loc[demog['icustay_id'] == icustayid, ['age', 'dischtime']].values[0] # Age of patient + discharge time
        if d1[0] > 6574:  # If older than 18 years old
            # CHARTEVENTS
            if (icustayid-200000) < 10000:
                temp=ce010
            elif (icustayid-200000) < 20000:
                temp=ce1020
            elif (icustayid-200000) < 30000:
                temp=ce2030
            elif (icustayid-200000) < 40000:
                temp=ce3040
            elif (icustayid-200000) < 50000:
                temp=ce4050
            elif (icustayid-200000) < 60000:
                temp=ce5060
            elif (icustayid-200000) < 70000:
                temp=ce6070
            elif (icustayid-200000) < 80000:
                temp=ce7080
            elif (icustayid-200000) < 90000:
                temp=ce8090
            else:
                temp=ce90100
            temp = temp[temp['icustay_id'] == icustayid]

            ii = (temp['charttime'] >= qst - (winb4+4)*3600) & (temp['charttime'] <= qst + (winaft+4)*3600) # Time period of interest -4h and +4h
            temp = temp.loc[ii]   # Only time period of interest

            # LABEVENTS
            ii = labU['icustay_id'] == icustayid
            temp2 = labU.loc[ii]
            ii = (temp2['charttime'] >= qst - (winb4+4)*3600) & (temp2['charttime'] <= qst + (winaft+4)*3600) # Time period of interest -4h and +4h
            temp2 = temp2.loc[ii]   # Only time period of interest

            # Mech Vent + ?extubated
            ii = MV['icustay_id'] == icustayid
            temp3 = MV.loc[ii]
            ii = (temp3['charttime'] >= qst - (winb4+4)*3600) & (temp3['charttime'] <= qst + (winaft+4)*3600) # Time period of interest -4h and +4h
            temp3 = temp3.loc[ii]   #only time period of interest
            t = np.unique(pd.concat([temp['charttime'], temp2['charttime'], temp3['charttime']], ignore_index=True).values) # List of unique timestamps from all 3 sources / sorted in ascending order
 
            if len(t) > 0:
                for i in range(len(t)):
                    #CHARTEVENTS
                    ii = temp['charttime'] == t[i]
                    col = temp.loc[ii,'itemid']
                    value = temp.loc[ii,'valuenum']
                    reformat[irow, 0] = i+1 #timestep  
                    reformat[irow, 1] = icustayid
                    reformat[irow, 2] = t[i] #charttime
                    reformat[irow, 2+col.astype(int).values] = value.values # Store available values

                    # LAB VALUES
                    ii = temp2['charttime'] == t[i]
                    col = temp2.loc[ii, 'itemid']
                    value = temp2.loc[ii, 'valuenum']
                    reformat[irow, 30+col.astype(int).values] = value.values  # Store available values
                
                    # Mechanical Ventilation  
                    ii = temp3['charttime'] == t[i]
                    if np.nansum(ii) > 0:
                        col = temp3.loc[ii, 'MechVent']
                        value = temp3.loc[ii, 'Extubated']
                        reformat[irow, 66] = col.values[0] # Store available values
                        reformat[irow, 67] = value.values[0] # Store available values
                    else:
                        reformat[irow, 66]= np.nan
                        reformat[irow, 67]= np.nan
                    irow += 1

                qstime[icustayid][0] = qst # Flag for presumed infection / this is time of sepsis if SOFA >=2 for this patient
                # SAVE FIRST and LAST TIMESTAMPS, in QSTIME, for each ICUSTAYID
                qstime[icustayid][1] = t[0]   # First timestamp
                qstime[icustayid][2] = t[-1]  # Last timestamp
                qstime[icustayid][3] = d1[1]  # Discharge time

reformat = np.delete(reformat, range(irow, len(reformat)) ,axis=0)  # Delete unused rows

########################################################################
#                                   OUTLIERS 
########################################################################
print('Full ICU -- Handling outliers')

# Weight
reformat = deloutabove(reformat, 4, 300) 

# Heart Rate
reformat = deloutabove(reformat, 7, 250)

# Blood Pressure
reformat = deloutabove(reformat, 8, 300)
reformat = deloutbelow(reformat, 9, 0) 
reformat = deloutabove(reformat, 9, 200)
reformat = deloutbelow(reformat, 10, 0) 
reformat = deloutabove(reformat, 10, 200) 

# Respiratory Rate
reformat = deloutabove(reformat, 11, 80) 

# SpO2
reformat = deloutabove(reformat, 12, 150) 
reformat[reformat[:, 12]>100, 12] = 100

# Temperature
reformat[(reformat[:, 13] > 90) & (np.isnan(reformat[:, 14])), 14] = reformat[(reformat[:, 13] > 90) & (np.isnan(reformat[:, 14])), 13]
reformat = deloutabove(reformat, 13, 90) 

# Interface / is in col 22
# FiO2
reformat = deloutabove(reformat, 22, 100) 
reformat[reformat[:, 22] < 1 , 22] = reformat[reformat[:,22] < 1 , 22]*100

reformat = deloutbelow(reformat, 22, 20) 
reformat = deloutabove(reformat, 23, 1.5)

# O2 FLOW
reformat = deloutabove(reformat, 24, 70)

#PEEP
reformat=deloutbelow(reformat, 25, 0) 
reformat=deloutabove(reformat, 25, 40) 

# Total Volume
reformat=deloutabove(reformat, 26, 1800)

# Mean Volume
reformat=deloutabove(reformat, 27, 50)

# Potassium
reformat=deloutbelow(reformat, 31, 1) 
reformat=deloutabove(reformat, 31, 15) 

# Sodium
reformat=deloutbelow(reformat, 32, 95) 
reformat=deloutabove(reformat, 32, 178)

# Chloride
reformat=deloutbelow(reformat, 33, 70) 
reformat=deloutabove(reformat, 33, 150)

# Glucose
reformat=deloutbelow(reformat, 34, 1) 
reformat=deloutabove(reformat, 34, 1000)

# Creatinine
reformat=deloutabove(reformat, 36, 150)

# Magnesium
reformat=deloutabove(reformat, 37, 10)

# Calcium
reformat=deloutabove(reformat, 38, 20)

# Ionized Calcium
reformat=deloutabove(reformat, 39, 5) 

# CO2
reformat=deloutabove(reformat, 40, 120) 

# SGPT/SGOT
reformat=deloutabove(reformat, 41, 10000) 
reformat=deloutabove(reformat, 42, 10000) 

# Hb/Ht
reformat=deloutabove(reformat, 49, 20) 
reformat=deloutabove(reformat, 50, 65) 

# White Blood Cells
reformat=deloutabove(reformat, 52, 500) 

# Platelets
reformat=deloutabove(reformat, 53, 2000)

# INR
reformat=deloutabove(reformat, 57, 20) 

# pH
reformat=deloutbelow(reformat, 58, 6.7) 
reformat=deloutabove(reformat, 58, 8) 

# pO2
reformat=deloutabove(reformat, 59, 700)

# pCO2
reformat=deloutabove(reformat, 60, 200)

# Base Excess
reformat=deloutbelow(reformat, 61, -50)

# Lactate
reformat=deloutabove(reformat, 62, 30) 

####################################################################
# More data manipulation / imputation from existing values

# Estimate GCS from RASS - data from Wesley JAMA 2003
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] >= 0), 5] = 15
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -1), 5] = 14
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -2), 5] = 12
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -3), 5] = 11
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -4), 5] = 6
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -5), 5] = 3

# FiO2
reformat[(~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:, 23])), 23] = reformat[(~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:, 23])), 22] / 100
reformat[(~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22])), 22] = reformat[(~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22])), 23] * 100

print('Full ICU -- Doing sample and hold')
sample_and_hold = pd.read_csv('ReferenceFiles/sample_and_hold.csv', index_col = None)

reformatsah = SAH(reformat,sample_and_hold)  # Do SAH first to handle this task

# NO FiO2, YES O2 flow, no interface OR cannula
ii = np.where((np.isnan(reformatsah[:, 22])) & (~np.isnan(reformatsah[:, 24])) & ((reformatsah[:, 21] == 0) | (reformatsah[:, 21] == 2)))[0] #As np.where returns a tuple
reformat[ii[reformatsah[ii, 24] <= 15], 22] = 70
reformat[ii[reformatsah[ii, 24] <= 12], 22] = 62
reformat[ii[reformatsah[ii, 24] <= 10], 22] = 55
reformat[ii[reformatsah[ii, 24] <= 8], 22]  = 50
reformat[ii[reformatsah[ii, 24] <= 6], 22]  = 44
reformat[ii[reformatsah[ii, 24] <= 5], 22]  = 40
reformat[ii[reformatsah[ii, 24] <= 4], 22]  = 36
reformat[ii[reformatsah[ii, 24] <= 3], 22]  = 32
reformat[ii[reformatsah[ii, 24] <= 2], 22]  = 28
reformat[ii[reformatsah[ii, 24] <= 1], 22]  = 24

# NO FiO2, NO O2 flow, no interface OR cannula
ii = np.where((np.isnan(reformatsah[:, 22])) & np.isnan(reformatsah[:, 24]) & ((reformatsah[:, 21] == 0) | (reformatsah[:, 21] == 2)))[0]  #no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = 21

# NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
ii = np.where((np.isnan(reformatsah[:, 22])) & (~np.isnan(reformatsah[:, 24])) & 
((reformatsah[:, 21] == 1) | (reformatsah[:, 21]==3) | (reformatsah[:, 21] == 4) | (reformatsah[:, 21] == 5) | (reformatsah[:, 21]==6) | (reformatsah[:, 21]==9) | (reformatsah[:, 21]==10)))[0]
reformat[ii[reformatsah[ii, 24]<=15], 22] = 75
reformat[ii[reformatsah[ii, 24]<=12], 22] = 69
reformat[ii[reformatsah[ii, 24]<=10], 22] = 66
reformat[ii[reformatsah[ii, 24]<=8], 22]  = 58
reformat[ii[reformatsah[ii, 24]<=6], 22]  = 40
reformat[ii[reformatsah[ii, 24]<=4], 22]  = 36

# NO FiO2, NO O2 flow, face mask OR ....OR ventilator
ii = np.where(np.isnan(reformatsah[:, 22]) & np.isnan(reformatsah[:, 24]) & ((reformatsah[:, 21] == 1) | (reformatsah[:, 21] == 3) | 
(reformatsah[:, 21] == 4) | (reformatsah[:, 21] == 5) | (reformatsah[:, 21] == 6) | (reformatsah[:, 21] == 9) | (reformatsah[:, 21] == 10)))[0]  #no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = np.nan

# NO FiO2, YES O2 flow, Non rebreather mask
ii = np.where(np.isnan(reformatsah[:, 22]) & (~np.isnan(reformatsah[:, 24])) & (reformatsah[:, 21] == 7))[0]
reformat[ii[reformatsah[ii, 24] >= 10], 22] = 90
reformat[ii[reformatsah[ii, 24] >= 15], 22] = 100
reformat[ii[reformatsah[ii, 24] < 10], 22]  = 80
reformat[ii[reformatsah[ii, 24] <= 8], 22]  = 70
reformat[ii[reformatsah[ii, 24] <= 6], 22]  = 60

# NO FiO2, NO O2 flow, NRM
ii= np.where(np.isnan(reformatsah[:, 22]) & np.isnan(reformatsah[:, 24]) & (reformatsah[:, 21]==7))[0]  #no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = np.nan

# Update FiO2 columns again
ii = (~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:,23]))
reformat[ii, 23] = reformat[ii, 22]/100
ii = (~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22]))
reformat[ii, 22] = reformat[ii, 23]*100

# Blood Pressure
ii = (~np.isnan(reformat[:, 8])) & (~np.isnan(reformat[:, 9])) & np.isnan(reformat[:, 10])
reformat[ii, 10] = (3*reformat[ii, 9] - reformat[ii, 8])/2
ii = (~np.isnan(reformat[:, 8])) & (~np.isnan(reformat[:, 10])) & np.isnan(reformat[:, 9])
reformat[ii, 9] = (reformat[ii, 8] + 2*reformat[ii, 10])/3
ii = (~np.isnan(reformat[:, 9])) & (~np.isnan(reformat[:, 10])) & np.isnan(reformat[:, 8])
reformat[ii, 8] = 3*reformat[ii, 9] - 2*reformat[ii, 10]

# Temperature
# Some values recorded in the wrong column
ii = (reformat[:, 14] > 25) & (reformat[:, 14] < 45) # tempF close to 37deg??!
reformat[ii, 13] = reformat[ii, 14]
reformat[ii, 14] = np.nan
ii = reformat[:, 13] >70  # tempC > 70, likely recorded in Farenheit
reformat[ii, 14] = reformat[ii, 13]
reformat[ii, 13] = np.nan
ii = (~np.isnan(reformat[:, 13])) & np.isnan(reformat[:, 14])
reformat[ii, 14] = reformat[ii, 13]*1.8+32
ii = (~np.isnan(reformat[:, 14])) & np.isnan(reformat[:, 13])
reformat[ii, 13] = (reformat[ii, 14] - 32)/1.8

# Hb/Ht
ii = (~np.isnan(reformat[:,49])) & np.isnan(reformat[:, 50])
reformat[ii, 50] = (reformat[ii, 49] * 2.862) + 1.216
ii = (~np.isnan(reformat[:, 50])) & np.isnan(reformat[:, 49])
reformat[ii, 49] = (reformat[ii, 50] - 1.216)/2.862

# Bilirubin
ii = (~np.isnan(reformat[:, 43])) & np.isnan(reformat[:, 44])
reformat[ii, 44] = (reformat[ii, 43]*0.6934)-0.1752
ii = (~np.isnan(reformat[:, 44])) & np.isnan(reformat[:, 43])
reformat[ii, 43] = (reformat[ii, 44] + 0.1752)/0.6934

########################################################################
#                      SAMPLE AND HOLD on RAW DATA
########################################################################
print('Full ICU -- SAMPLE AND HOLD on RAW DATA')
reformat = SAH(reformat[:,0:68],sample_and_hold)

########################################################################
#                             DATA COMBINATION
########################################################################
print('Full ICU -- Data combination')
# WARNING: the time window of interest has been defined above (here -24 -> +48)! 
timestep = 4  # Resolution of timesteps, in hours
irow = 0   
icustayidlist = np.unique(reformat[:,1]).astype(np.int32)
reformat2 = np.nan*np.ones((reformat.shape[0], 85))  # Output array  
num_patients = len(icustayidlist)  # Number of patients
# Adding 2 empty cols for future shock index=HR/SBP and P/F
reformat = np.hstack([reformat, np.nan*np.ones((reformat.shape[0], 2))])
bar = pyprind.ProgBar(num_patients)
for i in range(len(icustayidlist)):
    bar.update()
    icustayid = icustayidlist[i] 
    
    #CHARTEVENTS AND LAB VALUES
    temp = reformat[reformat[:,1] == icustayid,:]   # subtable of interest
    beg = temp[0,2]   # timestamp of first record
    
    #IV FLUID STUFF
    iv = inputMV['icustay_id'] == icustayid         # rows of interest in inputMV
    input = inputMV[iv]                             # subset of interest
    iv = inputCV['icustay_id'] == icustayid         # rows of interest in inputCV
    input2 = inputCV[iv]                            # subset of interest
    startt = input['starttime']                     # start of all infusions and boluses
    endt = input['endtime']                         # end of all infusions and boluses
    rate = input['norm_rate_of_infusion']           # rate of infusion (is NaN for boluses) || corrected for tonicity
    pread = inputpreadm[inputpreadm['icustay_id'] == icustayid]['inputpreadm'] # preadmission volume

    input = input.values
    input2 = input2.values

    if len(pread) >0:  # store the value, if available
        totvol = np.nansum(pread)
    else:
        totvol=0   # if not documented: it's zero
       
    # Compute volume of fluid given before start of record!!!
    t0 = 0
    t1 = beg
    # Input from MetaVision (4 ways to compute)
    infu = np.nansum(rate*(endt-startt)*((endt<=t1) & (startt >= t0))/3600 + rate*(endt-t0)*((startt <= t0) & (endt <= t1) & (endt >= t0))/3600 +
                            rate*(t1-startt)*((startt >= t0) & (endt >=t1) & (startt <= t1))/3600 + rate*(t1-t0)*((endt >= t1) & (startt <= t0))/3600)
    # All boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
    bolus = np.nansum(input[np.isnan(input[:, 5]) & (input[:, 1] >= t0) & (input[:, 1] <= t1), 6]) + np.nansum(input2[(input2[:, 1] >= t0) & (input2[:, 1] <= t1), 4])
    totvol = np.nansum([totvol,infu,bolus]) 
    
    #########################################################################################
    #VASOPRESSORS    
    iv = vasoMV['icustay_id'] == icustayid  # rows of interest in vasoMV
    vaso1 = vasoMV[iv]    # subset of interest
    iv = vasoCV['icustay_id'] == icustayid   # rows of interest in vasoCV
    vaso2 = vasoCV[iv]    # subset of interest
    startv = vaso1['starttime'].values  # start of VP infusion
    endv = vaso1['endtime'].values      # end of VP infusions
    ratev = vaso1['rate_std'].values  # rate of VP infusion
    
    # DEMOGRAPHICS / gender, age, elixhauser, re-admit, died in hosp?, died within
    # 48h of out_time (likely in ICU or soon after), died within 90d after admission?        
    demogi = demog['icustay_id'] == icustayid
    dem = np.array(
            list(demog.gender[demogi].values) +
            list(demog.age[demogi].values) +
            list(demog.elixhauser[demogi].values) +
            list(demog.re_admission[demogi].values) + 
            list(demog.morta_hosp[demogi].values) + 
            list(abs(demog.dod[demogi].values - demog.outtime[demogi].values) < (24*3600*2)) +
            list(demog.morta_90[demogi].values) + 
            [(qstime[icustayid][3] - qstime[icustayid][2])/3600]
            )
        
        
    #URINE OUTPUT
    iu = UO['icustay_id'] == icustayid   #rows of interest in inputMV
    output = UO[iu]    #subset of interest
    pread = UOpreadm[UOpreadm['icustay_id'] == icustayid-200000]['value'].values #preadmission UO ????????????????? Why no + 200000 for icustayid here?
    if len(pread) > 0:     #store the value, if available
        UOtot = np.nansum(pread)
    else:
        UOtot = 0
    #adding the volume of urine produced before start of recording!    
    UOnow = np.nansum(output[(output['charttime']>=t0) & (output['charttime'] <= t1)]['value'].values) #t0 and t1 defined above
    UOtot = np.nansum([UOtot, UOnow])
    
    
    for j in range(0, 79, timestep): #0:timestep:79 #-28 until +52 = 80 hours in total
        t0 = 3600*j + beg   #left limit of time window
        t1 = 3600*(j + timestep) + beg   #right limit of time window
        ii = (temp[:, 2] >= t0) & (temp[:, 2] <= t1)  #index of items in this time period
        if sum(ii)>0:
            #ICUSTAY_ID, OUTCOMES, DEMOGRAPHICS
            reformat2[irow, 0] = (j/timestep)+1    # 'bloc' = timestep (1,2,3...)
            reformat2[irow, 1] = icustayid         # icustay_ID
            reformat2[irow, 2] = 3600*j+ beg       # t0 = lower limit of time window
            reformat2[irow, 3:11] = dem            # demographics and outcomes
            
            #CHARTEVENTS and LAB VALUES (+ includes empty cols for shock index and P/F)
            value = temp[ii]  # Records all values in this timestep
                
            if sum(ii) == 1:   # if only 1 row of values at this timestep
                reformat2[irow, 11:78] = value[:, 3:]
            else:
                reformat2[irow, 11:78] = np.nanmean(value[:, 3:], axis=0)  # mean of all available values
        
            # VASOPRESSORS
            # for CV: dose at timestamps.
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            #----t0---start----end-----t1----
            #----start---t0----end----t1----
            #-----t0---start---t1---end
            #----start---t0----t1---end----
            # MetaVision
            v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv<=t1)) | ((startv >= t0) & (startv <= t1))| ((startv <= t0) & (endv>=t1))
            # CareVue
            v2 = vaso2[(vaso2['charttime'] >= t0) & (vaso2['charttime'] <= t1)]['rate_std'].values
            if len(list(ratev[v]) +  list(v2)) > 0:
                v1 = np.nanmedian(list(ratev[v]) +  list(v2))
                v2 = np.nanmax(list(ratev[v]) + list(v2))
            else:
                v1 = np.nan
                v2 = np.nan
            
            if (~np.isnan(v1)) and (~np.isnan(v2)):
                reformat2[irow, 78] = v1    #median of dose of VP
                reformat2[irow, 79] = v2    #max dose of VP
        
            # INPUT FLUID
            # Input from MV (4 ways to compute)
            infu = np.nansum(rate*(endt-startt)*((endt <= t1) & (startt >= t0))/3600 
                                          + rate*(endt-t0)*((startt <= t0) & (endt <= t1) & (endt >= t0))/3600 
                                          + rate*(t1-startt)*((startt >= t0) & (endt >= t1) & (startt <= t1))/3600 
                                          + rate*(t1-t0)*((endt >=t1) & (startt <= t0))/3600)
            # All boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
            bolus = np.nansum(input[(np.isnan(input[:, 5])) & (input[:, 1] >= t0) & (input[:, 1] <= t1), 6]) + np.nansum(input2[(input2[:, 1] >= t0) & (input2[:, 1] <= t1), 4])
            # Cumulate all fluid given
            totvol = np.nansum([totvol, infu, bolus])
            reformat2[irow, 80] = totvol    #total fluid given
            reformat2[irow, 81] = np.nansum([infu, bolus])  #fluid given at this step
            
            # Urine Output
            UOnow = np.nansum(output[(output['charttime'] >= t0) & (output['charttime'] <= t1)]['value'].values)
            UOtot = np.nansum([UOtot, UOnow])
            reformat2[irow, 82] = UOtot    # Total Urine Output
            reformat2[irow, 83] = np.nansum(UOnow)  # Urine Output at this step

            #CUMULATED BALANCE
            reformat2[irow, 84] = totvol - UOtot

            irow += 1

reformat2 = np.delete(reformat2, range(irow, len(reformat2)) ,axis=0) 

########################################################################
#    CONVERT TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS
########################################################################
print('FULL ICU -- CONVERTING TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS')
dataheaders = ['Height_cm', 'Weight_kg', 'GCS','RASS','HR', 'SysBP', 'MeanBP', 'DiaBP',	'RR', 'SpO2', 'Temp_C', 'Temp_F', 'CVP', 'PAPsys', 'PAPmean', 'PAPdia', 'CI', 
'SVR', 'Interface', 'FiO2_100', 'FiO2_1', 'O2flow', 'PEEP', 'TidalVolume', 'MinuteVentil', 'PAWmean', 'PAWpeak', 'PAWplateau', 'Potassium', 'Sodium',
'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Direct_bili', 'Total_protein',
'Albumin', 'Troponin', 'CRP', 'Hb', 'Ht', 'RBC_count', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'ACT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',
'Arterial_BE', 'Arterial_lactate', 'HCO3', 'ETCO2', 'SvO2', 'MechVent', 'Extubated', 'Shock_Index', 'PaO2_FiO2']
dataheaders = ['bloc','icustayid','charttime','gender','age','elixhauser','re_admission', 'died_in_hosp', 'died_within_48h_of_out_time','mortality_90d','delay_end_of_record_and_discharge_or_death'] + \
    dataheaders + ['median_dose_vaso','max_dose_vaso','input_total','input_4hourly','output_total','output_4hourly','cumulated_balance']

reformat2t = pd.DataFrame(reformat2, columns=dataheaders)
miss = np.sum(np.isnan(reformat2), axis=0)/reformat2.shape[0]

# If values have less than 70% missing values (over 30% of values present): I keep them
reformat3 = reformat2[:,[True]*11 + (miss[11:74] < 0.7).tolist() + [True]*11]
reformat3t = pd.DataFrame(reformat3, columns= reformat2t.columns[[True]*11 + (miss[11:74] < 0.7).tolist() + [True]*11])

########################################################################
#             HANDLING OF MISSING VALUES  &  CREATE REFORMAT4T
########################################################################

# Do linear interpolation where missingness is low (kNN imputation doesnt work if all rows have missing values)
print('Full ICU -- Doing linear interpolation where missingness is low (kNN imputation doesnt work if all rows have missing values)')
miss = np.sum(np.isnan(reformat3), axis=0)/reformat3.shape[0]
ii = (miss>0) & (miss<0.05)  #less than 5% missingness
mechventcol = reformat3t.columns.tolist().index('mechvent')

for i in range(10,mechventcol): # Correct column by column
    if ii[i]==1:
        reformat3[:,i] = fixgaps(reformat3[:,i])

reformat3t[reformat3t.columns[10:mechventcol]] = reformat3[:,10:mechventcol]

# KNN IMPUTATION -  Done on chunks of 10K records.
print('Full ICU -- KNN imputation')

reformat3t_cols = reformat3t.columns.tolist()
mechventcol = reformat3t_cols.index('mechvent')
ref = np.copy(reformat3[:,11:mechventcol])  #columns of interest

bar_knn = pyprind.ProgBar(len(range(0,reformat3.shape[0],9999)))
for i in range(0,reformat3.shape[0],9999):   #dataset divided in 10K rows chunks (otherwise too large)
    bar_knn.update()
    ref[i:i+9999,:] = KNN(k=1).fit_transform(ref[i:i+9999,:])

reformat3t[reformat3t_cols[11:mechventcol]] = ref 

reformat4t = reformat3t.copy()
reformat4 = reformat4t.values

########################################################################
#        COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS...
########################################################################
print('FULL ICU -- COMPUTING SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS...')
# CORRECT GENDER
reformat4t['gender'] = reformat4t['gender'] - 1

# CORRECT AGE > 200 yo
ii = reformat4t['age'] > 150*365.25
reformat4t.loc[ii,'age'] = 91.4*365.25

# FIX MECHVENT
reformat4t['mechvent'].fillna(0, inplace=True)
reformat4t.loc[reformat4t['mechvent'] > 0, 'mechvent'] = 1

# FIX Elixhauser missing values
reformat4t['elixhauser'].loc[np.isnan(reformat4t['elixhauser'])] = np.nanmedian(reformat4t['elixhauser'])   #use the median value / only a few missing data points 

# Vasopressors / no NAN
reformat4t['median_dose_vaso'].fillna(0, inplace=True)
reformat4t['max_dose_vaso'].fillna(0, inplace=True)

# Recompute P/F with no missing values...
reformat4t['PaO2_FiO2'] = reformat4t['paO2']/reformat4t['FiO2_1']

# Recompute SHOCK INDEX without NAN and INF
reformat4t['Shock_Index'] = reformat4t['HR']/reformat4t['SysBP']

reformat4t.loc[np.isinf(reformat4t['Shock_Index']), 'Shock_Index'] = np.NaN

d = np.nanmean(reformat4t['Shock_Index'])
reformat4t['Shock_Index'].fillna(d, inplace=True)

# SOFA - at each timepoint we need (in this order):  
# P/F,  MV,  PLT,  TOT_BILI,  MAP,  NORAD(max),  GCS,  CR,  UO
s = reformat4t[['PaO2_FiO2', 'Platelets_count', 'Total_bili', 'MeanBP', 'max_dose_vaso', 'GCS', 'Creatinine', 'output_4hourly']].values
p = np.arange(5)
s1=np.array([s[:,0]>400, (s[:, 0]>=300) & (s[:, 0]<400), (s[:, 0]>=200) & (s[:, 0]<300), (s[:, 0]>=100) & (s[:, 0]<200), s[:, 0]<100 ])   #count of points for all 6 criteria of sofa
s2=np.array([s[:,1]>150, (s[:, 1]>=100) & (s[:, 1]<150), (s[:, 1]>=50) & (s[:, 1]<100), (s[:, 1]>=20) & (s[:, 1]<50), s[:, 1]<20 ])
s3=np.array([s[:, 2]<1.2, (s[:, 2]>=1.2) & (s[:, 2]<2), (s[:, 2]>=2) & (s[:, 2]<6), (s[:, 2]>=6) & (s[:, 2]<12), s[:, 2]>12 ])
s4=np.array([s[:, 3]>=70, (s[:, 3]<70) & (s[:, 3]>=65), (s[:, 3]<65), (s[:, 4]>0) & (s[:, 4]<=0.1), s[:, 4]>0.1 ])
s5=np.array([s[:, 5]>14, (s[:, 5]>12) & (s[:, 5]<=14), (s[:, 5]>9) & (s[:, 5]<=12), (s[:, 5]>5) & (s[:, 5]<=9), s[:, 5]<=5])
s6=np.array([s[:, 6]<1.2, (s[:, 6]>=1.2) & (s[:, 6]<2), (s[:, 6]>=2) & (s[:, 6]<3.5), ((s[:, 6]>=3.5) & (s[:, 6]<5))|(s[:, 7]<84), (s[:, 6]>5)|(s[:, 7]<34)])

num_columns = reformat4t.shape[1]   #nr of variables in data
newcols_reformat4 = np.zeros((reformat4t.shape[0],7)) 
for i in range(reformat4t.shape[0]):
    t = max(p[s1[:, i]], default=0) + max(p[s2[:, i]], default=0) + max(p[s3[:, i]], default=0) + max(p[s4[:, i]], default=0) + max(p[s5[:, i]], default=0) + max(p[s6[:, i]], default=0)  #SUM OF ALL 6 CRITERIA
    if t > 0:
        newcols_reformat4[i, :] = [max(p[s1[:, i]], default=0), max(p[s2[:, i]], default=0), max(p[s3[:, i]], default=0), max(p[s4[:, i]], default=0), max(p[s5[:, i]], default=0), max(p[s6[:, i]], default=0), t]

# SIRS - at each timepoint |  need: temp HR RR PaCO2 WBC 
s = reformat4t[['Temp_C', 'HR', 'RR', 'paCO2', 'WBC_count']].values

s1=np.array([(s[:, 0]>=38) | (s[:,0]<=36)])   # Count of points for all criteria of SIRS
s2=np.array([s[:, 1]>90])
s3=np.array([(s[:, 2]>=20) | (s[:, 3]<=32)])
s4=np.array([(s[:, 4]>=12) | (s[:, 4]<4)])
newcols_sirs = (1*s1) + (1*s2) + (1*s3) + (1*s4)

# Adds 2 cols for SOFA and SIRS
# Records values
reformat4t['SOFA'] = newcols_reformat4[:,-1]
reformat4t['SIRS'] = newcols_sirs[0]


########################################################################
#                   EXCLUSION OF SOME PATIENTS 
########################################################################

# Check for patients with extreme UO = outliers = to be deleted (>40 litres of UO per 4h!!)
a = reformat4t['output_4hourly'] > 12000 
i = reformat4t[a]['icustayid'].unique() 
i = reformat4t['icustayid'].isin(i) 
reformat4t.drop(reformat4t.index[i], inplace=True) 

# Some have bili = 999999
a = reformat4t['Total_bili'] > 10000 
i = reformat4t[a]['icustayid'].unique()
i = reformat4t['icustayid'].isin(i) 
reformat4t.drop(reformat4t.index[i], inplace=True)

# Check for patients with extreme INTAKE = outliers = to be deleted (>10 litres of intake per 4h!!)
a = reformat4t['input_4hourly'] > 10000 
i = reformat4t[a]['icustayid'].unique()
i = reformat4t['icustayid'].isin(i) 
reformat4t.drop(reformat4t.index[i], inplace=True)
########################################################################

# Exclude early deaths from possible withdrawals 
print('Full ICU -- Excluding early deaths from possible withdrawals')
# Stats per patient
q = reformat4t['bloc']==1

num_of_trials = len(reformat4t['icustayid'].unique()) 
a = reformat4t[['icustayid', 'mortality_90d', 'max_dose_vaso', 'SOFA']].values
a = pd.DataFrame(a, columns = ['id', 'mortality_90d', 'vaso', 'sofa']) 
d = a.groupby(['id']).max() 
d_count = a.groupby(['id']).count()

# Find the patients who match the Sepsis 3 criteria
e = np.zeros(num_of_trials)
for i in range(num_of_trials):
    if d['mortality_90d'].iloc[i] == 1:
        ii = (reformat4t['icustayid'] == d.index[i]) & (reformat4t['bloc'] == d_count.iloc[i]['mortality_90d'])  #last row for this patient
        e[i] = np.sum((reformat4t['max_dose_vaso'][ii] == 0) & (d['vaso'].iloc[i] > 0.3) & (reformat4t['SOFA'][ii] >= d['sofa'].iloc[i]/2)) > 0
r = d.index[(e == 1) & (d_count['mortality_90d'] < 20)] # ids to be removed
ii = reformat4t['icustayid'].isin(r) 
reformat4t = reformat4t.loc[~ii] 

# Exclude patients who died in ICU during data collection period
print('Full ICU -- excluding patients who died in ICU during data collection period')
ii = (reformat4t['bloc'] == 1) & (reformat4t['died_within_48h_of_out_time'] == 1) & (reformat4t['delay_end_of_record_and_discharge_or_death'] < 24)
ii = reformat4t['icustayid'][ii].isin(icustayidlist).index 
ii = reformat4t['icustayid'].isin(ii)
reformat4t = reformat4t.loc[~ii] 

#######################################################################
#       CREATE SEPSIS COHORT FROM ALL ICU PATIENTS EXTRACTED
########################################################################
print('Creating sepsis cohort')
# Create array with 1 row per icu admission
# Keep only patients with flagged sepsis (max sofa during time period of interest >= 2)
# Assumed baseline SOFA is zero
sepsis = np.zeros((30000,5)) #NOTE: For other cohorts, this size may have to be changed
irow = 0

bar_cohort = pyprind.ProgBar(len(icustayidlist))
for icustayid in icustayidlist:
    bar_cohort.update()
    ii = reformat4t['icustayid'] == icustayid 
    if sum(ii) > 0:
        sofa = reformat4t['SOFA'][ii]
        sirs = reformat4t['SIRS'][ii]
        sepsis[irow, 0] = icustayid
        sepsis[irow, 1] = reformat4t['mortality_90d'][ii].iloc[0] # 90-day mortality
        sepsis[irow, 2] = np.max(sofa)
        sepsis[irow, 3] = np.max(sirs)
        sepsis[irow, 4] = qstime[icustayid][0]   #time of onset of sepsis #icustayid-1 not done to keep it consistent with earlier verified use of qstime and 0 added as onset of sepsis index.
        irow += 1

sepsis = np.delete(sepsis, range(irow, len(sepsis)) ,axis=0) # Remove extra rows
sepsis = pd.DataFrame(sepsis, columns=['icustayid', 'morta_90d', 'max_sofa', 'max_sirs', 'sepsis_time'])

# Delete all non-septic patients
ii = sepsis['max_sofa'] < 2
sepsis = sepsis[~ii]
# Final count of patients included
print('Final patient count:', sepsis.shape[0])  

# Save cohort
if pargs.save_intermediate:
    sepsis.to_csv('new_sepsis_mimiciii.csv', index=False) 

########################################################################
#           INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT
########################################################################
#gives an array with all unique charttime (1 per row) and all items in columns.
################## IMPORTANT !!!!!!!!!!!!!!!!!!
# Here we use -24 -> +48 to define the MDP
print('Sepsis Cohort -- Making an array with all unique charttime (1 per row) and all items in columns.')
reformat = np.nan*np.ones((2000000,69))  #final table 
qstime = dict()
winb4 = 25   #lower limit for inclusion of data (24h before time flag)
winaft = 49  # upper limit (48h after)
irow = 0  #recording row for summary table
bar = pyprind.ProgBar(sepsis.shape[0]+1)
for icustayidrow in range(1, sepsis.shape[0]+1):
    bar.update()
    qst = sepsis['sepsis_time'].iloc[icustayidrow-1] #;%,3); %flag for presumed infection
    icustayid = int(sepsis['icustayid'].iloc[icustayidrow-1])
    qstime[icustayid] = np.zeros(4)
    # CHARTEVENTS
    if (icustayid-200000) < 10000:
        temp=ce010
    elif (icustayid-200000) < 20000:
        temp=ce1020
    elif (icustayid-200000) < 30000:
        temp=ce2030
    elif (icustayid-200000) < 40000:
        temp=ce3040
    elif (icustayid-200000) < 50000:
        temp=ce4050
    elif (icustayid-200000) < 60000:
        temp=ce5060
    elif (icustayid-200000) < 70000:
        temp=ce6070
    elif (icustayid-200000) < 80000:
        temp=ce7080
    elif (icustayid-200000) < 90000:
        temp=ce8090
    else:
        temp=ce90100
    temp = temp[temp['icustay_id'] == icustayid]

    ii = (temp['charttime'] >= qst - (winb4+4)*3600) & (temp['charttime'] <= qst + (winaft+4)*3600) #time period of interest -4h and +4h
    temp = temp.loc[ii]   #only time period of interest

    # LAB EVENTS
    ii = labU['icustay_id'] == icustayid
    temp2 = labU.loc[ii]
    ii = (temp2['charttime'] >= qst - (winb4+4)*3600) & (temp2['charttime'] <= qst + (winaft+4)*3600) #time period of interest -4h and +4h
    temp2 = temp2.loc[ii]   #only time period of interest

    # Mech Vent + ?extubated
    ii = MV['icustay_id'] == icustayid
    temp3 = MV.loc[ii]
    ii = (temp3['charttime'] >= qst - (winb4+4)*3600) & (temp3['charttime'] <= qst + (winaft+4)*3600) # Time period of interest -4h and +4h
    temp3 = temp3.loc[ii]   #only time period of interest
    
    t = np.unique(pd.concat([temp['charttime'], temp2['charttime'], temp3['charttime']], ignore_index=True).values) # List of unique timestamps from all 3 sources / sorted in ascending order

    if len(t) > 0:
        for i in range(len(t)):
            # CHARTEVENTS
            ii = temp['charttime'] == t[i]
            col = temp.loc[ii, 'itemid']
            value = temp.loc[ii, 'valuenum']
            reformat[irow, 0] = i+1 # Timestep  
            reformat[irow, 1] = icustayid
            reformat[irow, 2] = t[i] # Charttime
            reformat[irow, 3] = qst  # Store the presumed onset time
            reformat[irow, 3+col.astype(int).values] = value.values # Store available values

            #LAB VALUES
            ii = temp2['charttime'] == t[i]
            col = temp2.loc[ii, 'itemid']
            value = temp2.loc[ii, 'valuenum']
            reformat[irow,31+col.astype(int).values] = value.values  #store available values
        
            #MV  
            ii = temp3['charttime'] == t[i]
            if np.nansum(ii) > 0:
                col = temp3.loc[ii, 'mechvent']
                value = temp3.loc[ii, 'extubated']
                reformat[irow, 67] = col.values[0] # Store available values
                reformat[irow, 68] = value.values[0] # Store available values
            else:
                reformat[irow, 67]= np.nan
                reformat[irow, 68]= np.nan
            irow += 1

        qstime[icustayid][0] = qst # Flag for presumed infection / this is time of sepsis if SOFA >=2 for this patient
        # WE SAVE FIRST and LAST TIMESTAMPS, in QSTIME, for each ICUSTAYID
        qstime[icustayid][1] = t[0]   # First timestamp
        qstime[icustayid][2] = t[-1]  # Last timestamp
        qstime[icustayid][3] = demog.loc[demog['icustay_id'] == icustayid, 'dischtime'].values[0] # Discharge time

reformat = np.delete(reformat, range(irow, len(reformat)) ,axis=0)  # Remove unused rows

########################################################################
#                                   OUTLIERS 
########################################################################
print('Sepsis Cohort -- Handling outliers')

# Weight
reformat = deloutabove(reformat, 5, 300)

# Heart Rate
reformat = deloutabove(reformat, 8, 250)

# Blood Pressure
reformat = deloutabove(reformat, 9, 300)
reformat = deloutbelow(reformat, 10, 0)
reformat = deloutabove(reformat, 11, 200)
reformat = deloutbelow(reformat, 11, 0)
reformat = deloutabove(reformat, 11, 200)

# Respiratory Rate
reformat = deloutabove(reformat, 12, 80) 

# SpO2
reformat = deloutabove(reformat, 13, 150)
reformat[reformat[:, 13]>100, 13] = 100 
reformat = deloutbelow(reformat, 13, 50) 

# Temperature
reformat[(reformat[:, 14] > 90) & (np.isnan(reformat[:, 15])), 15] = reformat[(reformat[:, 14] > 90) & (np.isnan(reformat[:, 15])), 14]
reformat = deloutabove(reformat, 14, 90) 
reformat = deloutbelow(reformat, 14, 25) 

# interface / is in col 23
# FiO2
reformat = deloutabove(reformat, 23, 100)
reformat[reformat[:, 23] < 1 , 23] = reformat[reformat[:, 23] < 1 , 23]*100
reformat = deloutbelow(reformat, 23, 20) 
reformat = deloutabove(reformat, 24, 1.5)

# O2 FLOW
reformat = deloutabove(reformat, 25, 70)

# PEEP
reformat=deloutbelow(reformat, 26, 0)
reformat=deloutabove(reformat, 26, 40)

# Total Volume
reformat=deloutabove(reformat, 27, 1800)

# Mean Volume
reformat=deloutabove(reformat, 28, 50)

# Potassium
reformat=deloutbelow(reformat, 32, 1)
reformat=deloutabove(reformat, 32, 15)

# Sodium
reformat=deloutbelow(reformat, 33, 95)
reformat=deloutabove(reformat, 33, 178)

# Chloride
reformat=deloutbelow(reformat, 34, 70)
reformat=deloutabove(reformat, 34, 150)

# Glucose
reformat=deloutbelow(reformat, 35, 1)
reformat=deloutabove(reformat, 35, 1000) 

# Creatinine
reformat=deloutabove(reformat, 37, 150)

# Magnesium
reformat=deloutabove(reformat, 38, 10)

# Calcium
reformat=deloutabove(reformat, 39, 20)

# Ionized Calcium
reformat=deloutabove(reformat, 40, 5)

# CO2
reformat=deloutabove(reformat, 41, 120)

# SGPT/SGOT
reformat=deloutabove(reformat, 42, 10000)
reformat=deloutabove(reformat, 43, 10000)

# Hb/Ht
reformat=deloutabove(reformat, 50, 20)
reformat=deloutabove(reformat, 51, 65)

# White Blood Cells
reformat=deloutabove(reformat, 53, 500)

# Platelets
reformat=deloutabove(reformat, 54, 2000)

# INR
reformat=deloutabove(reformat, 58, 20)

# pH
reformat=deloutbelow(reformat, 59, 6.7)
reformat=deloutabove(reformat, 59, 8) 

# pO2
reformat=deloutabove(reformat, 60, 700)

# pCO2
reformat=deloutabove(reformat, 61, 200)

# Base Excess
reformat=deloutbelow(reformat, 62, -50)

# Lactate
reformat=deloutabove(reformat, 63, 30)

####################################################################
# More data manipulation / imputation from existing values
# Estimate GCS from RASS - data from Wesley JAMA 2003
reformat[(np.isnan(reformat[:, 6])) & (reformat[:, 7] >= 0), 6] = 15
reformat[(np.isnan(reformat[:, 6])) & (reformat[:, 7] == -1), 6] = 14
reformat[(np.isnan(reformat[:, 6])) & (reformat[:, 7] == -2), 6] = 12
reformat[(np.isnan(reformat[:, 6])) & (reformat[:, 7] == -3), 6] = 11
reformat[(np.isnan(reformat[:, 6])) & (reformat[:, 7] == -4), 6] = 6
reformat[(np.isnan(reformat[:, 6])) & (reformat[:, 7] == -5), 6] = 3

# FiO2
reformat[(~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 24])), 24] = reformat[(~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 24])), 23] / 100
reformat[(~np.isnan(reformat[:, 24])) & (np.isnan(reformat[:, 23])), 23] = reformat[(~np.isnan(reformat[:, 24])) & (np.isnan(reformat[:, 23])), 24] * 100

# ESTIMATE FiO2 /// with use of interface / device (cannula, mask, ventilator....)
print('Sepsis Cohort -- Doing sample and hold')
reformatsah = SAH(reformat, sample_and_hold, adjust=1) # do SAH first to handle this task, setting `adjust` to 1 since we added another column

# NO FiO2, YES O2 flow, no interface OR cannula
ii = np.where((np.isnan(reformatsah[:,23])) & (~np.isnan(reformatsah[:,25])) & ((reformatsah[:,22] == 0) | (reformatsah[:,22] == 2)))[0] #As np.where returns a tuple
reformat[ii[reformatsah[ii, 25] <= 15], 23] = 70
reformat[ii[reformatsah[ii, 25] <= 12], 23] = 62
reformat[ii[reformatsah[ii, 25] <= 10], 23] = 55
reformat[ii[reformatsah[ii, 25] <= 8], 23]  = 50
reformat[ii[reformatsah[ii, 25] <= 6], 23]  = 44
reformat[ii[reformatsah[ii, 25] <= 5], 23]  = 40
reformat[ii[reformatsah[ii, 25] <= 4], 23]  = 36
reformat[ii[reformatsah[ii, 25] <= 3], 23]  = 32
reformat[ii[reformatsah[ii, 25] <= 2], 23]  = 28
reformat[ii[reformatsah[ii, 25] <= 1], 23]  = 24

# NO FiO2, NO O2 flow, no interface OR cannula
ii = np.where((np.isnan(reformatsah[:, 23])) & np.isnan(reformatsah[:, 25]) & ((reformatsah[:, 22] == 0) | (reformatsah[:, 22] == 2)))[0]  #no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 23] = 21

# NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
ii = np.where((np.isnan(reformatsah[:,23])) & (~np.isnan(reformatsah[:,25])) & 
((reformatsah[:, 22]==1) | (reformatsah[:, 22]==3) | (reformatsah[:, 22]==4) | (reformatsah[:, 22]==5) | (reformatsah[:, 22]==6) | (reformatsah[:, 22]==9) | (reformatsah[:, 22]==10)))[0]
reformat[ii[reformatsah[ii, 25]<=15], 23] = 75
reformat[ii[reformatsah[ii, 25]<=12], 23] = 69
reformat[ii[reformatsah[ii, 25]<=10], 23] = 66
reformat[ii[reformatsah[ii, 25]<=8], 23]  = 58
reformat[ii[reformatsah[ii, 25]<=6], 23]  = 40
reformat[ii[reformatsah[ii, 25]<=4], 23]  = 36

# NO FiO2, NO O2 flow, face mask OR ....OR ventilator
ii = np.where(np.isnan(reformatsah[:, 23]) & np.isnan(reformatsah[:, 25]) & ((reformatsah[:, 22] == 1) | (reformatsah[:, 22] == 3) | 
(reformatsah[:, 22] == 4) | (reformatsah[:, 22] == 5) | (reformatsah[:, 22] == 6) | (reformatsah[:, 22] == 9) | (reformatsah[:, 22] == 10)))[0]  # No FiO2 given and O2 flow given, no interface OR cannula
reformat[ii, 23] = np.nan

# NO FiO2, YES O2 flow, Non rebreather mask
ii = np.where(np.isnan(reformatsah[:, 23]) & (~np.isnan(reformatsah[:, 25])) & (reformatsah[:, 22] == 7))[0]
reformat[ii[reformatsah[ii, 25] >= 10], 23] = 90
reformat[ii[reformatsah[ii, 25] >= 15], 23] = 100
reformat[ii[reformatsah[ii, 25] < 10], 23]  = 80
reformat[ii[reformatsah[ii, 25] <= 8], 23]  = 70
reformat[ii[reformatsah[ii, 25] <= 6], 23]  = 60

# NO FiO2, NO O2 flow, NRM
ii= np.where(np.isnan(reformatsah[:, 23]) & np.isnan(reformatsah[:, 25]) & (reformatsah[:, 22]==7))[0]  #no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 23] = np.nan

# Update Fi02 columns again
ii = (~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 24]))
reformat[ii, 24] = reformat[ii, 23]/100
ii = (~np.isnan(reformat[:, 24])) & (np.isnan(reformat[:, 24]))
reformat[ii, 23] = reformat[ii, 24]*100

# BLOOD PRESSURE
ii = (~np.isnan(reformat[:, 9])) & (~np.isnan(reformat[:, 10])) & np.isnan(reformat[:, 11])
reformat[ii, 11] = (3*reformat[ii, 10] - reformat[ii, 9])/2
ii = (~np.isnan(reformat[:, 9])) & (~np.isnan(reformat[:, 11])) & np.isnan(reformat[:, 10])
reformat[ii, 10] = (reformat[ii, 9] + 2*reformat[ii, 11])/3
ii = (~np.isnan(reformat[:, 10])) & (~np.isnan(reformat[:, 11])) & np.isnan(reformat[:, 9])
reformat[ii, 9] = 3*reformat[ii, 10] - 2*reformat[ii, 11]

# TEMPERATURE
# some values recorded in the wrong column
ii = (reformat[:, 15] > 25) & (reformat[:, 15] < 45) #tempF close to 37deg??!
reformat[ii, 14] = reformat[ii, 15]
reformat[ii, 15] = np.nan
ii = reformat[:, 14] >70  # TempC > 70, likely recorded in Farenheit
reformat[ii, 15] = reformat[ii, 14]
reformat[ii, 14] = np.nan
ii = (~np.isnan(reformat[:, 14])) & np.isnan(reformat[:, 15])
reformat[ii, 15] = reformat[ii, 14] * 1.8 + 32
ii = (~np.isnan(reformat[:, 14])) & np.isnan(reformat[:, 13])
reformat[ii, 14] = (reformat[ii, 15] - 32)/1.8

# Hb/Ht
ii = (~np.isnan(reformat[:, 50])) & np.isnan(reformat[:,51])
reformat[ii, 51] = (reformat[ii, 50] * 2.862) + 1.216
ii = (~np.isnan(reformat[:, 51])) & np.isnan(reformat[:,50])
reformat[ii, 50] = (reformat[ii, 51] - 1.216)/2.862

# BILIRUBIN
ii = (~np.isnan(reformat[:, 44])) & np.isnan(reformat[:,45])
reformat[ii, 45] = (reformat[ii, 44]*0.6934)-0.1752
ii = (~np.isnan(reformat[:, 45])) & np.isnan(reformat[:,44])
reformat[ii, 44] = (reformat[ii, 45] + 0.1752)/0.6934

########################################################################
#                      SAMPLE AND HOLD on RAW DATA
########################################################################
print('SEPSIS COHORT -- SAMPLE AND HOLD on RAW DATA')
reformat = SAH(reformat[:, 0:69], sample_and_hold, adjust=1)  # Setting `adjust` to 1 to account for the added column for `presumed_onset`
#######################################################################
########################################################################
#                             DATA COMBINATION
########################################################################
print('Sepsis Cohort -- Data combination')
# WARNING: the time window of interest has been defined above (here -24 -> +48)! 
timestep = 4  # Resolution of timesteps, in hours
irow = 0   
icustayidlist = np.unique(reformat[:,1]).astype(np.int32)
reformat2 = np.nan*np.ones((reformat.shape[0], 86))  # Output array
num_patients = len(icustayidlist)  # Number of patients

# Adding 2 empty cols for future shock index=HR/SBP and P/F
reformat = np.hstack([reformat, np.nan*np.ones((reformat.shape[0], 2))])
bar = pyprind.ProgBar(num_patients)
for i in range(len(icustayidlist)):
    bar.update()
    icustayid = icustayidlist[i]  
    
    #CHARTEVENTS AND LAB VALUES
    temp = reformat[reformat[:,1] == icustayid,:]   # Subtable of interest
    beg = temp[0,2]   # Timestamp of first record
    
    #IV FLUID STUFF
    iv = inputMV['icustay_id'] == icustayid   # Rows of interest in inputMV
    input = inputMV[iv]    # Subset of interest
    iv = inputCV['icustay_id'] == icustayid   # Rows of interest in inputCV
    input2 = inputCV[iv]    # Subset of interest
    startt = input['starttime'] # Start of all infusions and boluses
    endt = input['endtime'] # End of all infusions and boluses
    rate = input['norm_rate_of_infusion']  # Rate of infusion (is NaN for boluses) || corrected for tonicity
    pread = inputpreadm[inputpreadm['icustay_id'] == icustayid]['inputpreadm'] # Preadmission volume

    input = input.values
    input2 = input2.values

    if len(pread) >0:             # Store the value, if available
        totvol = np.nansum(pread)
    else:
        totvol=0   # If not documented: it's zero

    # Compute volume of fluid given before start of record!!!
    t0 = 0
    t1 = beg
    # Input from MV (4 ways to compute)
    infu = np.nansum(rate*(endt-startt)*((endt<=t1) & (startt >= t0))/3600 + rate*(endt-t0)*((startt <= t0) & (endt <= t1) & (endt >= t0))/3600 +
                            rate*(t1-startt)*((startt >= t0) & (endt >=t1) & (startt <= t1))/3600 + rate*(t1-t0)*((endt >= t1) & (startt <= t0))/3600)
    # All boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
    bolus = np.nansum(input[np.isnan(input[:,5]) & (input[:,1] >= t0) & (input[:,1] <= t1),6]) + np.nansum(input2[(input2[:,1] >= t0) & (input2[:,1] <= t1),4])
    totvol = np.nansum([totvol,infu,bolus]) 
    
    #########################################################################################
    #VASOPRESSORS    
    iv = vasoMV['icustay_id'] == icustayid # rows of interest in vasoMV
    vaso1 = vasoMV[iv]    # subset of interest
    iv = vasoCV['icustay_id'] == icustayid   # rows of interest in vasoCV
    vaso2 = vasoCV[iv]    # subset of interest
    startv = vaso1['starttime'].values  # start of VP infusion
    endv = vaso1['endtime'].values      # end of VP infusions
    ratev = vaso1['rate_std'].values  # rate of VP infusion

    # DEMOGRAPHICS / gender, age, elixhauser, re-admit, onset time, died in hosp?, died within 48h of out_time (likely in ICU or soon after),
    # died within 90d after admission?, Length of stay after obs window     
    demogi = demog['icustay_id'] == icustayid
    dem = np.array(
            list(demog.gender[demogi].values) +
            list(demog.age[demogi].values) +
            list(demog.elixhauser[demogi].values) +
            list(demog.re_admission[demogi].values) + # Using corrected readmission (within 30 days of previous discharge)
            [temp[0,3]] +                    # Adding presumed sepsis onset
            list(demog.morta_hosp[demogi].values) + 
            list(abs(demog.dod[demogi].values - demog.outtime[demogi].values) < (24*3600*2)) +
            list(demog.morta_90[demogi].values) + 
            [(qstime[icustayid][3] - qstime[icustayid][2])/3600]      # Length of time after last observation and discharge?
            )
  
    # URINE OUTPUT
    iu = UO['icustay_id'] == icustayid   #rows of interest in inputMV
    output = UO[iu]    #subset of interest
    pread = UOpreadm[UOpreadm['icustay_id'] == icustayid-200000]['value'].values #preadmission UO ????????????????? Why no + 200000 for icustayid here?
    if len(pread) > 0:     #store the value, if available
        UOtot = np.nansum(pread)
    else:
        UOtot = 0
    # adding the volume of urine produced before start of recording!    
    UOnow = np.nansum(output[(output['charttime']>=t0) & (output['charttime'] <= t1)]['value'].values) #t0 and t1 defined above
    UOtot = np.nansum([UOtot, UOnow])
    
    # Loop over the relevant times for each patient where information is recorded
    for j in range(0, 79, timestep): # 0:timestep:79 % -28 until +52 = 80 hours in total (4 hours buffer on either side of our desired window)
        t0 = 3600*j + beg   # left limit of time window
        t1 = 3600*(j + timestep) + beg   # right limit of time window
        ii = (temp[:,2] >= t0) & (temp[:,2] <= t1)  # index of items in this time period
        if sum(ii)>0:
            # ICUSTAY_ID, OUTCOMES, DEMOGRAPHICS
            reformat2[irow,0] = (j/timestep)+1    # 'bloc' = timestep (1,2,3...)
            reformat2[irow,1] = icustayid         # icustay_ID
            reformat2[irow,2] = 3600*j+ beg       # t0 = lower limit of time window
            reformat2[irow,3:12] = dem            # demographics and outcomes
            
            # CHARTEVENTS and LAB VALUES (+ includes empty columns for shock index and P/F)
            value = temp[ii]  #records all values in this timestep
                
            if sum(ii) == 1:   # if only 1 row of values at this timestep
                reformat2[irow,12:79] = value[:,4:]
            else:
                reformat2[irow,12:79] = np.nanmean(value[:,4:], axis=0) # mean of all available values

            #VASOPRESSORS
            # for CV: dose at timestamps.
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            #----t0---start----end-----t1----
            #----start---t0----end----t1----
            #-----t0---start---t1---end
            #----start---t0----t1---end----
            #MV
            v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv<=t1)) | ((startv >= t0) & (startv <= t1))| ((startv <= t0) & (endv>=t1))
            #CV
            v2 = vaso2[(vaso2['charttime'] >= t0) & (vaso2['charttime'] <= t1)]['rate_std'].values
            if len(list(ratev[v]) +  list(v2)) > 0:
                v1 = np.nanmedian(list(ratev[v]) +  list(v2))
                v2 = np.nanmax(list(ratev[v]) + list(v2))
            else:
                v1 = np.nan
                v2 = np.nan
            if (~np.isnan(v1)) and (~np.isnan(v2)):
                reformat2[irow,79] = v1    #median of dose of VP
                reformat2[irow,80] = v2    #max dose of VP

            #INPUT FLUID
            #input from MV (4 ways to compute)
            infu = np.nansum(rate*(endt-startt)*((endt <= t1) & (startt >= t0))/3600 
                                          + rate*(endt-t0)*((startt <= t0) & (endt <= t1) & (endt >= t0))/3600 
                                          + rate*(t1-startt)*((startt >= t0) & (endt >= t1) & (startt <= t1))/3600 
                                          + rate*(t1-t0)*((endt >=t1) & (startt <= t0))/3600)
            #all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
            bolus = np.nansum(input[(np.isnan(input[:,5])) & (input[:,1] >= t0) & (input[:,1] <= t1),6]) + np.nansum(input2[(input2[:,1] >= t0) & (input2[:,1] <= t1),4])
            #sum fluid given
            totvol = np.nansum([totvol, infu, bolus])
            reformat2[irow,81] = totvol    #total fluid given
            reformat2[irow,82] = np.nansum([infu, bolus])  #fluid given at this step
            
            #UO
            UOnow = np.nansum(output[(output['charttime'] >= t0) & (output['charttime'] <= t1)]['value'].values)
            UOtot = np.nansum([UOtot, UOnow])
            reformat2[irow,83] = UOtot    #total UO
            reformat2[irow,84] = np.nansum(UOnow)   #UO at this step

            #CUMULATED BALANCE
            reformat2[irow,85] = totvol - UOtot    #cumulated balance

            irow += 1

reformat2 = np.delete(reformat2, range(irow, len(reformat2)) ,axis=0) 

#########################################################################
#             CONVERT TO TABLE AND KEEP ONLY WANTED VARIABLE
#########################################################################

dataheaders = [i[1:-1] for i in sample_and_hold.columns]
dataheaders.extend(['Shock_Index', 'PaO2_FiO2'])
dataheaders = ['bloc', 'icustayid', 'charttime', 'gender', 'age', 'elixhauser', 're_admission', 'presumed_onset', 'died_in_hosp', \
    'died_within_48h_of_out_time', 'mortality_90d', 'delay_end_of_record_and_discharge_or_death'] + \
    dataheaders + ['median_dose_vaso','max_dose_vaso','input_total','input_4hourly','output_total','output_4hourly','cumulated_balance']


reformat2t = pd.DataFrame(reformat2, columns=dataheaders)

# headers I want to keep
dataheaders5 = ['bloc', 'icustayid', 'charttime', 'gender', 'age', 'elixhauser', 're_admission', 'presumed_onset', 'died_in_hosp', 'died_within_48h_of_out_time', \
    'mortality_90d', 'delay_end_of_record_and_discharge_or_death', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2', 'Temp_C', 'FiO2_1', \
    'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', \
    'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'Arterial_lactate', 'mechvent', 'Shock_Index', \
    'PaO2_FiO2', 'median_dose_vaso', 'max_dose_vaso', 'input_total', 'input_4hourly', 'output_total', 'output_4hourly', 'cumulated_balance']

reformat3t = reformat2t[dataheaders5].copy() 

## SOME DATA MANIPULATION BEFORE IMPUTATION
#CORRECT GENDER
reformat3t['gender'] = reformat3t['gender'] - 1

# CORRECT AGE > 200 yo
ii = reformat3t['age'] > 150*365.25
reformat3t.loc[ii,'age'] = 91.4*365.25

# FIX MECHVENT
reformat3t['mechvent'].fillna(0, inplace=True)
reformat3t.loc[reformat3t['mechvent'] > 0, 'mechvent'] = 1

# FIX Elixhauser missing values
reformat3t['elixhauser'].loc[np.isnan(reformat3t['elixhauser'])] = np.nanmedian(reformat3t['elixhauser'])   #use the median value / only a few missing data points 

# Vasopressors / no NAN
reformat3t['median_dose_vaso'].fillna(0, inplace=True)
reformat3t['max_dose_vaso'].fillna(0, inplace=True)

# Check missingness proportions here
miss = pd.DataFrame([np.sum(np.isnan(reformat3t.values), axis=0)/reformat3t.shape[0]], columns=reformat3t.columns)
# Fill the values temporarily with zeros, otherwise kNN imp doesnt work
reformat3t['Shock_Index'] = np.zeros(reformat3t.shape[0])
reformat3t['PaO2_FiO2'] = np.zeros(reformat3t.shape[0])

########################################################################
#        HANDLING OF MISSING VALUES & CREATE REFORMAT4T
########################################################################

# Do linear interpolation where missingness is low (kNN imputation doesnt work if all rows have missing values)
reformat3 = reformat3t.values
miss = np.sum(np.isnan(reformat3), axis=0)/reformat3.shape[0]
ii = (miss>0) & (miss<0.05)  #less than 5% missingness
mechventcol = reformat3t.columns.tolist().index('mechvent')

for i in range(12, mechventcol): # correct col by col, otherwise it does it wrongly
    if ii[i]==1:
        reformat3[:, i] = fixgaps(reformat3[:, i])

reformat3t[reformat3t.columns[12:mechventcol]] = reformat3[:, 12:mechventcol]

# KNN IMPUTATION - Done on chunks of 10K records.
print('Sepsis Cohort -- KNN imputation with K = 1')

reformat3t_cols = reformat3t.columns.tolist()
mechventcol = reformat3t_cols.index('mechvent')
ref = np.copy(reformat3[:,12:mechventcol])  #columns of interest

bar_knn = pyprind.ProgBar(len(range(0,reformat3.shape[0],9999)))
for i in range(0,reformat3.shape[0],9999):   # Dataset divided in 10K rows chunks (otherwise too large)
    bar_knn.update()
    ref[i:i+9999,:] = KNN(k=1).fit_transform(ref[i:i+9999,:])

# Copy on the interpolated data
reformat3t[reformat3t_cols[12:mechventcol]] = ref 

reformat4t = reformat3t.copy()  # Make final table copy to compute derived features

########################################################################
#        COMPUTE SOME DERIVED FEATURES: P/F, Shock Index, SOFA, SIRS...
########################################################################
# Recompute P/F with no missing values...
reformat4t['PaO2_FiO2'] = reformat4t['paO2']/reformat4t['FiO2_1']

# Recompute SHOCK INDEX without NAN and INF
reformat4t['Shock_Index'] = reformat4t['HR']/reformat4t['SysBP']

reformat4t.loc[np.isinf(reformat4t['Shock_Index']), 'Shock_Index'] = np.NaN
d = np.nanmean(reformat4t['Shock_Index'])
reformat4t['Shock_Index'].fillna(d, inplace=True)

shock_idx = reformat4t['Shock_Index'] >= np.quantile(reformat4t['Shock_Index'], 0.999)
reformat4t.loc[shock_idx, 'Shock_Index'] = np.quantile(reformat4t['Shock_Index'], 0.999)

# SOFA - at each timepoint we need (in this order):  
# P/F,  MV,  PLT,  TOT_BILI,  MAP,  NORAD(max),  GCS, CR,  UO
s = reformat4t[['PaO2_FiO2', 'Platelets_count', 'Total_bili', 'MeanBP', 'max_dose_vaso', 'GCS', 'Creatinine', 'output_4hourly']].values
p = np.arange(5)
s1=np.array([s[:, 0]>400, (s[:, 0]>=300) & (s[:, 0]<400), (s[:, 0]>=200) & (s[:, 0]<300), (s[:, 0]>=100) & (s[:, 0]<200), s[:, 0]<100 ])   #count of points for all 6 criteria of sofa
s2=np.array([s[:, 1]>150, (s[:, 1]>=100) & (s[:, 1]<150), (s[:, 1]>=50) & (s[:, 1]<100), (s[:, 1]>=20) & (s[:, 1]<50), s[:, 1]<20 ])
s3=np.array([s[:, 2]<1.2, (s[:, 2]>=1.2) & (s[:, 2]<2), (s[:, 2]>=2) & (s[:, 2]<6), (s[:, 2]>=6) & (s[:, 2]<12), s[:, 2]>12 ])
s4=np.array([s[:, 3]>=70, (s[:, 3]<70) & (s[:, 3]>=65), (s[:, 3]<65), (s[:, 4]>0) & (s[:, 4]<=0.1), s[:, 4]>0.1 ])
s5=np.array([s[:,5]>14, (s[:, 5]>12) & (s[:, 5]<=14), (s[:, 5]>9) & (s[:, 5]<=12), (s[:, 5]>5) & (s[:, 5]<=9), s[:, 5]<=5])
s6=np.array([s[:,6]<1.2, (s[:,6 ]>=1.2) & (s[:, 6]<2), (s[:, 6]>=2) & (s[:, 6]<3.5), ((s[:, 6]>=3.5) & (s[:, 6]<5))|(s[:, 7]<84), (s[:, 6]>5)|(s[:, 7]<34)])

num_columns = reformat4t.shape[1]   # Number of variables in data
newcols_reformat4 = np.zeros((reformat4t.shape[0],7))
for i in range(reformat4t.shape[0]): 
    t = max(p[s1[:,i]], default=0) + max(p[s2[:,i]], default=0) + max(p[s3[:,i]], default=0) + max(p[s4[:,i]], default=0) 
        + max(p[s5[:,i]], default=0) + max(p[s6[:,i]], default=0)  #SUM OF ALL 6 CRITERIA
    if t > 0:
        newcols_reformat4[i,:] = [max(p[s1[:,i]], default=0), max(p[s2[:,i]], default=0), max(p[s3[:,i]], default=0), 
            max(p[s4[:,i]], default=0), max(p[s5[:,i]], default=0), max(p[s6[:,i]], default=0), t]

# SIRS - at each timepoint |  need: temp HR RR PaCO2 WBC 
s = reformat4t[['Temp_C', 'HR', 'RR', 'paCO2', 'WBC_count']].values

s1=np.array([(s[:,0]>=38) | (s[:,0]<=36)])   #count of points for all criteria of SIRS
s2=np.array([s[:,1]>90])
s3=np.array([(s[:,2]>=20) | (s[:,3]<=32)])
s4=np.array([(s[:,4]>=12) | (s[:,4]<4)])
newcols_sirs = (1*s1) + (1*s2) + (1*s3) + (1*s4)

# more IO corrections
reformat4t.loc[reformat4t['input_total'] < 0, 'input_total'] = 0
reformat4t.loc[reformat4t['input_4hourly'] < 0, 'input_4hourly'] = 0

# records values
reformat4t['SOFA'] = newcols_reformat4[:,-1]
reformat4t['SIRS'] = newcols_sirs[0]

########################################################################
#                     CREATE FINAL MIMIC_TABLE
########################################################################
MIMICtable = reformat4t.copy()
if pargs.save_intermediate:
    MIMICtable.to_csv('MIMICtable.csv', index=False)

#################   Convert training data and compute conversion factors    ######################

# all 47 columns of interest + additional meta columns to easier associate trajectories with other patient auxiliary info (eg. notes)
colmeta = ['presumed_onset', 'charttime', 'icustayid']  # Meta-data around patient stay
colbin = ['gender', 'mechvent', 'max_dose_vaso', 're_admission']  # Binary features
# Patient features that will be z-normalize
colnorm = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1',\ 
        'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', \
        'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',\
        'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index',\
        'PaO2_FiO2', 'cumulated_balance']
# Patient features that will be log-normalized
collog=['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',\ 
        'input_total', 'input_4hourly', 'output_total', 'output_4hourly']

# find patients who died in ICU during data collection period
icustayidlist = MIMICtable['icustayid']
icuuniqueids = icustayidlist.unique() #list of unique icustayids from MIMIC

MIMICraw = MIMICtable[colmeta + colbin + colnorm + collog]
MIMICraw = MIMICraw.values  # RAW values
MIMICzs = np.hstack([MIMICtable[colmeta].values, MIMICtable[colbin].values-0.5, stats.zscore(MIMICtable[colnorm].values), 
    stats.zscore(np.log(0.1 + MIMICtable[collog].values))])

MIMICzs[:,5] = np.log(MIMICzs[:, 5] + 0.6)  # MAX DOSE NORAD 
MIMICzs[:,46] = 2*MIMICzs[:,46]  # Increase weight of this variable

# compute conversion factors using MIMIC data
a = MIMICtable['input_4hourly'].values  # IV fluid
a = stats.rankdata(a[a>0])/len(a[a>0]) # excludes zero fluid (will be action 1)
iof = np.floor((a+0.2499999999)*4).astype(np.int) #converts iv volume in 4 actions
a = MIMICtable['input_4hourly'] > 0 # location of non-zero fluid in big matrix
io = np.ones(MIMICtable.shape[0]) # array of ones, by default     
io[a] = iof + 1 # where more than zero fluid given: save actual action

vc = MIMICtable['max_dose_vaso'].values
vcr = stats.rankdata(vc[vc != 0])/len(vc[vc != 0])
vcr = np.floor((vcr+0.249999999999)*4)
vcr[vcr == 0] = 1
vc[vc != 0] = vcr + 1
vc[vc == 0] = 1

actions = (io-1)*5 + (vc-1)

# Process MIMICzs into trajectory data for later use
MIMICzs             = pd.DataFrame(MIMICzs, columns=colmeta+colbin+colnorm+collog)
meta_df             = pd.DataFrame(MIMICzs[colmeta].values, columns = colmeta)
ob_df               = pd.DataFrame(MIMICzs[colbin+colnorm+collog].values, columns=colbin+colnorm+collog)
ac_df               = pd.DataFrame(actions, columns=['action'])
raw_data_df         = MIMICtable.copy()
num_actions         = 25
outcome_key         = 'died_within_48h_of_out_time' #Should be either 'died_within_48h_of_out_time' or ''mortality_90d''
meta_cols           = meta_df.columns.tolist()
ob_cols             = ob_df.columns.tolist()
raw_data_df['traj'] = (raw_data_df['bloc'] == 1).cumsum().values
meta_df['traj']     = (raw_data_df['bloc'] == 1).cumsum().values
ob_df['traj']       = (raw_data_df['bloc'] == 1).cumsum().values
ac_df['traj']       = (raw_data_df['bloc'] == 1).cumsum().values
trajectories        = raw_data_df['traj'].unique()
data = {}
data['meta_cols'] = meta_cols
data['obs_cols'] = ob_cols
data['traj'] = {}
print('Sepsis Cohort -- Making trajectory data')
bar = pyprind.ProgBar(len(trajectories))
for i in trajectories:
    bar.update()
    data['traj'][i] = {}
    data['traj'][i]['meta'] = meta_df[meta_df['traj']==i][meta_cols].values.T
    data['traj'][i]['obs'] = ob_df[ob_df['traj'] == i][ob_cols].values.T
    data['traj'][i]['actions'] = ac_df[ac_df['traj'] == i]['action'].values.astype(np.int32)
    data['traj'][i]['outcome'] = raw_data_df[raw_data_df['traj'] == i][outcome_key].values[0] 
    data['traj'][i]['rewards'] = np.zeros(len(data['traj'][i]['actions']))
    data['traj'][i]['rewards'][-1] = (1-2*data['traj'][i]['outcome'])

print('Sepsis Cohort -- Making final output file')
col_names = ['traj', 'step']
col_names.extend(['m:'+ i for i in data['meta_cols']])
col_names.extend(['o:'+ i for i in data['obs_cols']])
col_names.append('a:action')
col_names.append('r:reward')
all_data = []
bar = pyprind.ProgBar(len(data['traj'].keys()))
for i in data['traj'].keys():
    bar.update()
    for ctr in range(data['traj'][i]['actions'].shape[0]):
        all_data.append([])
        all_data[-1].append(i)
        all_data[-1].append(ctr)
        for m_index in range(data['traj'][i]['meta'].shape[0]):
            all_data[-1].append(data['traj'][i]['meta'][m_index, ctr])
        for o_index in range(data['traj'][i]['obs'].shape[0]):
            all_data[-1].append(data['traj'][i]['obs'][o_index, ctr])
        all_data[-1].append(data['traj'][i]['actions'][ctr])
        all_data[-1].append(data['traj'][i]['rewards'][ctr])
df = pd.DataFrame(all_data, columns=col_names)
df.to_csv('sepsis_final_data_withTimes.csv', index=False)

if pargs.process_raw: # If we want to convert the MIMICraw data into trajectories, repeat the above code without normalizing
    raw_df              = pd.DataFrame(MIMICraw, columns=colmeta+colbin+colnorm+collog)
    meta_df             = raw_df[colmeta]
    meta_df             = pd.DataFrame(meta_df.values, columns=colmeta)
    ob_df               = raw_df[colbin+colnorm+collog]
    ob_df               = pd.DataFrame(ob_df.values, columns=colbin+colnorm+collog)
    ac_df               = pd.DataFrame(actions, columns=['action'])
    raw_data_df         = MIMICtable.copy()
    num_actions         = 25
    outcome_key         = 'died_within_48h_of_out_time' #Should be either 'died_within_48h_of_out_time' or ''mortality_90d''
    meta_cols           = meta_df.columns.tolist()
    ob_cols             = ob_df.columns.tolist()
    raw_data_df['traj'] = (raw_data_df['bloc'] == 1).cumsum().values
    meta_df['traj']     = (raw_data_df['bloc'] == 1).cumsum().values
    ob_df['traj']       = (raw_data_df['bloc'] == 1).cumsum().values
    ac_df['traj']       = (raw_data_df['bloc'] == 1).cumsum().values
    trajectories        = raw_data_df['traj'].unique()
    data = {}
    data['meta_cols'] = meta_cols
    data['obs_cols'] = ob_cols
    data['traj'] = {}
    print('Sepsis Cohort -- Making RAW trajectory data')
    bar = pyprind.ProgBar(len(trajectories))
    for i in trajectories:
        bar.update()
        data['traj'][i] = {}
        data['traj'][i]['meta'] = meta_df[meta_df['traj']==i][meta_cols].values.T
        data['traj'][i]['obs'] = ob_df[ob_df['traj'] == i][ob_cols].values.T
        data['traj'][i]['actions'] = ac_df[ac_df['traj'] == i]['action'].values.astype(np.int32)
        data['traj'][i]['outcome'] = raw_data_df[raw_data_df['traj'] == i][outcome_key].values[0] 
        data['traj'][i]['rewards'] = np.zeros(len(data['traj'][i]['actions']))
        data['traj'][i]['rewards'][-1] = (1-2*data['traj'][i]['outcome'])

    print('Sepsis Cohort -- Making final RAW output file')
    col_names = ['traj', 'step']
    col_names.extend(['m:'+ i for i in data['meta_cols']])
    col_names.extend(['o:'+ i for i in data['obs_cols']])
    col_names.append('a:action')
    col_names.append('r:reward')
    all_data = []
    bar = pyprind.ProgBar(len(data['traj'].keys()))
    for i in data['traj'].keys():
        bar.update()
        for ctr in range(data['traj'][i]['actions'].shape[0]):
            all_data.append([])
            all_data[-1].append(i)
            all_data[-1].append(ctr)
            for m_index in range(data['traj'][i]['meta'].shape[0]):
                all_data[-1].append(data['traj'][i]['meta'][m_index, ctr])            
            for o_index in range(data['traj'][i]['obs'].shape[0]):
                all_data[-1].append(data['traj'][i]['obs'][o_index, ctr])
            all_data[-1].append(data['traj'][i]['actions'][ctr])
            all_data[-1].append(data['traj'][i]['rewards'][ctr])
    df = pd.DataFrame(all_data, columns=col_names)
    df.to_csv('sepsis_final_data_RAW_withTimes.csv', index=False)

####################################################################################################################################
