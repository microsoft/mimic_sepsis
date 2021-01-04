"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MIMIC-III Sepsis Cohort Extraction.

This file is sourced and modified from: https://github.com/matthieukomorowski/AI_Clinician
"""

import argparse
import os

import pandas as pd
import psycopg2 as pg


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", default='USERNAME', help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password", default='PASSWORD', help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimic' user={0} host='mimic' options='--search_path=mimimciii' password={1}".format(pargs.username,pargs.password))

# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# Extraction of sub-tables
# There are 43 tables in the Mimic III database. 
# 26 unique tables; the other 17 are partitions of chartevents that are not to be queried directly 
# See: https://mit-lcp.github.io/mimic-schema-spy/
# We create 15 sub-tables when extracting from the database

# From each table we extract subject ID, admission ID, ICU stay ID 
# and relevant times to assist in joining these tables
# All other specific information extracted will be documented before each section of the following code.


# NOTE: The next three tables are built to help identify when a patient may be 
# considered to be septic, using the Sepsis 3 criteria

# 1. culture
# These correspond to blood/urine/CSF/sputum cultures etc
# There are 18 chartevent tables in the Mimic III database, one unsubscripted and 
# the others subscripted from 1 to 17. We use the unsubscripted one to create the 
# culture subtable. The remaining 17 are just partitions and should not be directly queried.
# The labels corresponding to the 51 itemids in the query below are:
"""
 Itemid | Label
-----------------------------------------------------
    938 | blood cultures
    941 | urine culture
    942 | BLOOD CULTURES
   2929 | sputum culture
   3333 | Blood Cultures
   4855 | Urine culture
   6035 | Urinalysis sent
   6043 | surface cultures
  70006 | ANORECTAL/VAGINAL CULTURE
  70011 | BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)
  70012 | BLOOD CULTURE
  70013 | FLUID RECEIVED IN BLOOD CULTURE BOTTLES
  70014 | BLOOD CULTURE - NEONATE
  70016 | BLOOD CULTURE (POST-MORTEM)
  70024 | VIRAL CULTURE: R/O CYTOMEGALOVIRUS
  70037 | FOOT CULTURE
  70041 | VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS
  70055 | POSTMORTEM CULTURE
  70057 | Rapid Respiratory Viral Screen & Culture
  70060 | Stem Cell - Blood Culture
  70063 | STERILITY CULTURE
  70075 | THROAT CULTURE
  70083 | VARICELLA-ZOSTER CULTURE
  80220 | AFB GROWN IN CULTURE; ADDITIONAL INFORMATION TO FOLLOW
 225401 | Blood Cultured
 225437 | CSF Culture
 225444 | Pan Culture
 225451 | Sputum Culture
 225454 | Urine Culture
 225722 | Arterial Line Tip Cultured
 225723 | CCO PAC Line Tip Cultured
 225724 | Cordis/Introducer Line Tip Cultured
 225725 | Dialysis Catheter Tip Cultured
 225726 | Tunneled (Hickman) Line Tip Cultured
 225727 | IABP Line Tip Cultured
 225728 | Midline Tip Cultured
 225729 | Multi Lumen Line Tip Cultured
 225730 | PA Catheter Line Tip Cultured
 225731 | Pheresis Catheter Line Tip Cultured
 225732 | PICC Line Tip Cultured
 225733 | Indwelling Port (PortaCath) Line Tip Cultured
 225734 | Presep Catheter Line Tip Cultured
 225735 | Trauma Line Tip Cultured
 225736 | Triple Introducer Line Tip Cultured
 225768 | Sheath Line Tip Cultured
 225814 | Stool Culture
 225816 | Wound Culture
 225817 | BAL Fluid Culture
 225818 | Pleural Fluid Culture
 226131 | ICP Line Tip Cultured
 227726 | AVA Line Tip Cultured
"""
query = """
select subject_id, hadm_id, icustay_id,  extract(epoch from charttime) as charttime, itemid
from mimiciii.chartevents
where itemid in (6035, 3333, 938, 941, 942, 4855, 6043, 2929, 225401, 225437, 225444, 225451, 225454, 225814,
  225816, 225817, 225818, 225722, 225723, 225724, 225725, 225726, 225727, 225728, 225729, 225730, 225731,
  225732, 225733, 227726, 70006, 70011, 70012, 70013, 70014, 70016, 70024, 70037, 70041, 225734, 225735,
  225736, 225768, 70055, 70057, 70060, 70063, 70075, 70083, 226131, 80220)
order by subject_id, hadm_id, charttime
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir, 'culture.csv'),index=False,sep='|')


# 2. microbio (Microbiologyevents)
query = """
select subject_id, hadm_id, extract(epoch from charttime) as charttime, extract(epoch from chartdate) as chartdate 
from mimiciii.microbiologyevents
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir, 'microbio.csv'),index=False,sep='|')


# 3. abx (Antibiotics administration)
# gsn/GSN: Generic Sequence Number. This number provides a representation of the drug in various coding systems. 
# GSN is First DataBank's classification system. These are 6 digit codes for various drugs.
# ???  The codes here correspond to various antibiotics as sepsis onset is detected by administration of antibiotcs ???
query = """
select hadm_id, icustay_id, extract(epoch from startdate) as startdate, extract(epoch from enddate) as enddate
from mimiciii.prescriptions
where gsn in ('002542','002543','007371','008873','008877','008879','008880','008935','008941',
  '008942','008943','008944','008983','008984','008990','008991','008992','008995','008996',
  '008998','009043','009046','009065','009066','009136','009137','009162','009164','009165',
  '009171','009182','009189','009213','009214','009218','009219','009221','009226','009227',
  '009235','009242','009263','009273','009284','009298','009299','009310','009322','009323',
  '009326','009327','009339','009346','009351','009354','009362','009394','009395','009396',
  '009509','009510','009511','009544','009585','009591','009592','009630','013023','013645',
  '013723','013724','013725','014182','014500','015979','016368','016373','016408','016931',
  '016932','016949','018636','018637','018766','019283','021187','021205','021735','021871',
  '023372','023989','024095','024194','024668','025080','026721','027252','027465','027470',
  '029325','029927','029928','037042','039551','039806','040819','041798','043350','043879',
  '044143','045131','045132','046771','047797','048077','048262','048266','048292','049835',
  '050442','050443','051932','052050','060365','066295','067471')
order by hadm_id, icustay_id
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'abx.csv'),index=False,sep='|')


# 4. demog (Patient demographics)
# See https://github.com/MIT-LCP/mimic-code/blob/master/concepts/comorbidity/elixhauser-quan.sql
# This code calculates the Elixhauser comorbidities as defined in Quan et. al 2009
# This outputs a materialized view (table) with 58976 rows and 31 columns. The first column is 'hadm_id' and the 
# rest of the columns are as given below (Each entry is either 0 or 1):
# 2. 'congestive_heart_failure', 
# 3. 'cardiac_arrhythmias',
# 4. 'valvular_disease',
# 5. 'pulmonary_circulation', 
# 6. 'peripheral_vascular',
# 7. 'hypertension', 
# 8. 'paralysis', 
# 9. 'other_neurological'
# 10.'chronic_pulmonary',
# 11. 'diabetes_uncomplicated', 
# 12. 'diabetes_complicated', 
# 13. 'hypothyroidism',
# 14. 'renal_failure', 
# 15. 'liver_disease', 
# 16. 'peptic_ulcer', 
# 17. 'aids', 
# 18. 'lymphoma',
# 19. 'metastatic_cancer', 
# 20. 'solid_tumor', 
# 21. 'rheumatoid_arthritis',
# 22. 'coagulopathy', 
# 23. 'obesity', 
# 24. 'weight_loss', 
# 25. 'fluid_electrolyte',
# 26. 'blood_loss_anemia', 
# 27. 'deficiency_anemias', 
# 28. 'alcohol_abuse',
# 29. 'drug_abuse', 
# 30. 'psychoses', 
# 31. 'depression'
query = """
DROP MATERIALIZED VIEW IF EXISTS PUBLIC.ELIXHAUSER_QUAN CASCADE;
CREATE MATERIALIZED VIEW PUBLIC.ELIXHAUSER_QUAN AS
with icd as
(
  select hadm_id, seq_num, icd9_code
  from mimiciii.diagnoses_icd
  where seq_num != 1 -- we do not include the primary icd-9 code
)
, eliflg as
(
select hadm_id, seq_num, icd9_code
, CASE
  when icd9_code in ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4254','4255','4257','4258','4259') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('428') then 1
  else 0 end as CHF       /* Congestive heart failure */

, CASE
  when icd9_code in ('42613','42610','42612','99601','99604') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4260','4267','4269','4270','4271','4272','4273','4274',
  '4276','4278','4279','7850','V450','V533') then 1
  else 0 end as ARRHY

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('0932','7463','7464','7465','7466','V422','V433') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('394','395','396','397','424') then 1
  else 0 end as VALVE     /* Valvular disease */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4150','4151','4170','4178','4179') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('416') then 1
  else 0 end as PULMCIRC  /* Pulmonary circulation disorder */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('440','441') then 1
  else 0 end as PERIVASC  /* Peripheral vascular disorder */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('401') then 1
  else 0 end as HTN       /* Hypertension, uncomplicated */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('402','403','404','405') then 1
  else 0 end as HTNCX     /* Hypertension, complicated */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('3341','3440','3441','3442','3443','3444','3445','3446','3449') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('342','343') then 1
  else 0 end as PARA      /* Paralysis */

, CASE
  when icd9_code in ('33392') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('3319','3320','3321','3334','3335','3362','3481','3483','7803','7843') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('334','335','340','341','345') then 1
  else 0 end as NEURO     /* Other neurological */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4168','4169','5064','5081','5088') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('490','491','492','493','494','495','496','500','501','502','503','504','505') then 1
  else 0 end as CHRNLUNG  /* Chronic pulmonary disease */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2500','2501','2502','2503') then 1
  else 0 end as DM        /* Diabetes w/o chronic complications*/

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2504','2505','2506','2507','2508','2509') then 1
  else 0 end as DMCX      /* Diabetes w/ chronic complications */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2409','2461','2468') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('243','244') then 1
  else 0 end as HYPOTHY   /* Hypothyroidism */

, CASE
  when icd9_code in ('40301','40311','40391','40402','40403','40412','40413','40492','40493') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('5880','V420','V451') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('585','586','V56') then 1
  else 0 end as RENLFAIL  /* Renal failure */

, CASE
  when icd9_code in ('07022','07023','07032','07033','07044','07054') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('0706','0709','4560','4561','4562','5722','5723','5724','5728',
    '5733','5734','5738','5739','V427') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('570','571') then 1
  else 0 end as LIVER     /* Liver disease */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('5317','5319','5327','5329','5337','5339','5347','5349') then 1
  else 0 end as ULCER     /* Chronic Peptic ulcer disease (includes bleeding only if obstruction is also present) */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('042','043','044') then 1
  else 0 end as AIDS      /* HIV and AIDS */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2030','2386') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('200','201','202') then 1
  else 0 end as LYMPH     /* Lymphoma */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('196','197','198','199') then 1
  else 0 end as METS      /* Metastatic cancer */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in
  (
     '140','141','142','143','144','145','146','147','148','149','150','151','152'
    ,'153','154','155','156','157','158','159','160','161','162','163','164','165'
    ,'166','167','168','169','170','171','172','174','175','176','177','178','179'
    ,'180','181','182','183','184','185','186','187','188','189','190','191','192'
    ,'193','194','195'
  ) then 1
  else 0 end as TUMOR     /* Solid tumor without metastasis */

, CASE
  when icd9_code in ('72889','72930') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('446','714','720','725') then 1
  else 0 end as ARTH              /* Rheumatoid arthritis/collagen vascular diseases */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2871','2873','2874','2875') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('286') then 1
  else 0 end as COAG      /* Coagulation deficiency */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2780') then 1
  else 0 end as OBESE     /* Obesity      */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('7832','7994') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('260','261','262','263') then 1
  else 0 end as WGHTLOSS  /* Weight loss */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2536') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('276') then 1
  else 0 end as LYTES     /* Fluid and electrolyte disorders */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2800') then 1
  else 0 end as BLDLOSS   /* Blood loss anemia */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2801','2808','2809') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('281') then 1
  else 0 end as ANEMDEF  /* Deficiency anemias */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2652','2911','2912','2913','2915','2918','2919','3030',
    '3039','3050','3575','4255','5353','5710','5711','5712','5713','V113') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('980') then 1
  else 0 end as ALCOHOL /* Alcohol abuse */

, CASE
  when icd9_code in ('V6542') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('3052','3053','3054','3055','3056','3057','3058','3059') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('292','304') then 1
  else 0 end as DRUG /* Drug abuse */

, CASE
  when icd9_code in ('29604','29614','29644','29654') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2938') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('295','297','298') then 1
  else 0 end as PSYCH /* Psychoses */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2962','2963','2965','3004') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('309','311') then 1
  else 0 end as DEPRESS  /* Depression */
from icd
)
-- collapse the icd9_code specific flags into hadm_id specific flags
-- this groups comorbidities together for a single patient admission
, eligrp as
(
  select hadm_id, max(chf) as chf, max(arrhy) as arrhy, max(valve) as valve, max(pulmcirc) as pulmcirc, 
  max(perivasc) as perivasc, max(htn) as htn, max(htncx) as htncx, max(para) as para, max(neuro) as neuro, 
  max(chrnlung) as chrnlung, max(dm) as dm, max(dmcx) as dmcx, max(hypothy) as hypothy, max(renlfail) as renlfail, 
  max(liver) as liver, max(ulcer) as ulcer, max(aids) as aids, max(lymph) as lymph, max(mets) as mets, max(tumor) as tumor, 
  max(arth) as arth, max(coag) as coag, max(obese) as obese, max(wghtloss) as wghtloss, max(lytes) as lytes, 
  max(bldloss) as bldloss, max(anemdef) as anemdef, max(alcohol) as alcohol, max(drug) as drug, max(psych) as psych, max(depress) as depress
from eliflg
group by hadm_id
)
-- now merge these flags together to define elixhauser
-- most are straightforward.. but hypertension flags are a bit more complicated

select adm.hadm_id, chf as CONGESTIVE_HEART_FAILURE, arrhy as CARDIAC_ARRHYTHMIAS, valve as VALVULAR_DISEASE, 
pulmcirc as PULMONARY_CIRCULATION, perivasc as PERIPHERAL_VASCULAR
-- we combine "htn" and "htncx" into "HYPERTENSION"
, case
    when htn = 1 then 1
    when htncx = 1 then 1
  else 0 end as HYPERTENSION
, para as PARALYSIS, neuro as OTHER_NEUROLOGICAL, chrnlung as CHRONIC_PULMONARY
-- only the more severe comorbidity (complicated diabetes) is kept
, case
    when dmcx = 1 then 0
    when dm = 1 then 1
  else 0 end as DIABETES_UNCOMPLICATED
, dmcx as DIABETES_COMPLICATED, hypothy as HYPOTHYROIDISM, renlfail as RENAL_FAILURE, liver as LIVER_DISEASE, ulcer as PEPTIC_ULCER, 
aids as AIDS, lymph as LYMPHOMA, mets as METASTATIC_CANCER
-- only the more severe comorbidity (metastatic cancer) is kept
, case
    when mets = 1 then 0
    when tumor = 1 then 1
  else 0 end as SOLID_TUMOR
, arth as RHEUMATOID_ARTHRITIS, coag as COAGULOPATHY, obese as OBESITY, wghtloss as WEIGHT_LOSS, lytes as FLUID_ELECTROLYTE, 
bldloss as BLOOD_LOSS_ANEMIA, anemdef as DEFICIENCY_ANEMIAS, alcohol as ALCOHOL_ABUSE, drug as DRUG_ABUSE, psych as PSYCHOSES
, depress as DEPRESSION

from mimiciii.admissions adm
left join eligrp eli
  on adm.hadm_id = eli.hadm_id
order by adm.hadm_id;
"""
cursor = conn.cursor()
cursor.execute(query)

# This demographics table is built based on the Elixhauser_Quan table previously defined in lines 149-404
query = """
select ad.subject_id, ad.hadm_id, i.icustay_id ,extract(epoch from ad.admittime) as admittime, extract(epoch from ad.dischtime) as dischtime, ROW_NUMBER() over (partition by ad.subject_id order by i.intime asc) as adm_order, case when i.first_careunit='NICU' then 5 when i.first_careunit='SICU' then 2 when i.first_careunit='CSRU' then 4 when i.first_careunit='CCU' then 6 when i.first_careunit='MICU' then 1 when i.first_careunit='TSICU' then 3 end as unit,  extract(epoch from i.intime) as intime, extract(epoch from i.outtime) as outtime, i.los,
 EXTRACT(EPOCH FROM (i.intime-p.dob)::INTERVAL)/86400 as age, extract(epoch from p.dob) as dob, extract(epoch from p.dod) as dod,
 p.expire_flag,  case when p.gender='M' then 1 when p.gender='F' then 2 end as gender,
 CAST(extract(epoch from age(p.dod,ad.dischtime))<=24*3600  as int )as morta_hosp,  --died in hosp if recorded DOD is close to hosp discharge
 CAST(extract(epoch from age(p.dod,i.intime))<=90*24*3600  as int )as morta_90,
 congestive_heart_failure+cardiac_arrhythmias+valvular_disease+pulmonary_circulation+peripheral_vascular+hypertension+paralysis+other_neurological+chronic_pulmonary+diabetes_uncomplicated+diabetes_complicated+hypothyroidism+renal_failure+liver_disease+peptic_ulcer+aids+lymphoma+metastatic_cancer+solid_tumor+rheumatoid_arthritis+coagulopathy+obesity	+weight_loss+fluid_electrolyte+blood_loss_anemia+	deficiency_anemias+alcohol_abuse+drug_abuse+psychoses+depression as elixhauser
from mimiciii.admissions ad, mimiciii.icustays i, mimiciii.patients p, public.elixhauser_quan elix
where ad.hadm_id=i.hadm_id and p.subject_id=i.subject_id and elix.hadm_id=ad.hadm_id
order by subject_id asc, intime asc
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'demog.csv'),index=False,sep='|')


# 5. ce (Patient vitals from chartevents)
# Divided into 10 chunks for speed (indexed by ICU stay ID). Each chunk is around 170 MB.
# Each itemid here corresponds to single measurement type
for i in range(0,100000,10000):
  print(i)
  query= "select distinct icustay_id, extract(epoch from charttime) as charttime, itemid, case when value = 'None' then '0' when value = \
  'Ventilator' then '1' when value='Cannula' then '2' when value = 'Nasal Cannula' then '2' when value = 'Face Tent' then '3' when value = \
  'Aerosol-Cool' then '4' when value = 'Trach Mask' then '5' when value = 'Hi Flow Neb' then '6' when value = 'Non-Rebreather' then '7' when \
  value = '' then '8'  when value = 'Venti Mask' then '9' when value = 'Medium Conc Mask' then '10' else valuenum end as valuenum from \
  mimiciii.chartevents where icustay_id>="+str(200000+i)+" and icustay_id< " + str(210000+i) + " and value is not null and \
  itemid in  (467, 470, 471, 223834, 227287, 194, 224691, 226707, 226730, 581, 580, 224639, 226512, 198, 228096, \
  211, 220045, 220179, 225309, 6701, 6, 227243, 224167, 51, 455, 220181, 220052, 225312, 224322, 6702, 443, 52,	\
  456, 8368, 8441, 225310, 8555, 8440, 220210, 3337, 224422, 618, 3603, 615, 220277, 646, 834, 3655, 223762, \
  223761, 678, 220074, 113, 492, 491, 8448, 116, 1372, 1366, 228368, 228177, 626, 223835, 3420, 160, 727, 190, 220339, 506, \
  505, 224700, 224686, 224684, 684, 224421, 224687, 450, 448, 445, 224697, 444, 224695, 535, 224696, 543, 3083, 2566, \
  654, 3050, 681, 2311)  order by icustay_id, charttime "        
  
  d=pd.read_sql_query(query,conn)
  d.to_csv(os.path.join(exportdir, 'ce' + str(i)+str(i+10000) +'.csv'),index=False,sep='|')


# 6. labs_ce (Labs from chartevents)
# Each itemid here corresponds to single measurement type
query = """
select icustay_id, extract(epoch from charttime) as charttime, itemid, valuenum
from mimiciii.chartevents
where valuenum is not null and icustay_id is not null and 
itemid in  (829, 1535, 227442, 227464, 4195, 3726, 3792, 837, 220645, 4194,	
3725, 3803, 226534, 1536, 4195, 3726, 788, 220602, 1523, 4193, 3724,
226536, 3747, 225664, 807, 811, 1529, 220621, 226537, 3744, 781, 1162, 225624,	
3737, 791, 1525, 220615, 3750, 821, 1532, 220635, 786, 225625, 1522, 3746, 816, 225667,	
3766, 777, 787, 770, 3801, 769, 3802, 1538, 848, 225690, 803, 1527, 225651, 3807,	
1539, 849, 772, 1521, 227456, 3727, 227429, 851, 227444, 814, 220228, 813,	
220545, 3761, 226540, 4197, 3799, 1127, 1542, 220546, 4200, 3834, 828, 227457,	
3789, 825, 1533, 227466, 3796, 824, 1286, 1671, 1520, 768, 220507, 815, 1530, 227467, 780,	
1126, 3839, 4753, 779, 490, 3785, 3838, 3837, 778, 3784, 3836, 3835, 776, 224828, 3736,	
4196, 3740, 74, 225668, 1531, 227443, 1817, 228640, 823, 227686)
order by icustay_id, charttime, itemid
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'labs_ce.csv'),index=False,sep='|')


# 7. labs_le (Labs from lab events)
query = """
select xx.icustay_id, extract(epoch from f.charttime) as timestp, f.itemid, f.valuenum
from(
select subject_id, hadm_id, icustay_id, intime, outtime
from mimiciii.icustays
group by subject_id, hadm_id, icustay_id, intime, outtime
) as xx inner join  mimiciii.labevents as f on f.hadm_id=xx.hadm_id and f.charttime>=xx.intime-interval '1 day' 
and f.charttime<=xx.outtime+interval '1 day'  and f.itemid in  (50971, 50822, 50824, 50806, 50931, 51081, 50885, 51003, 51222,
50810, 51301, 50983, 50902, 50809, 51006, 50912, 50960, 50893, 50808, 50804, 50878, 50861, 51464, 50883, 50976, 50862, 51002, 50889,
50811, 51221, 51279, 51300, 51265, 51275, 51274, 51237, 50820, 50821, 50818, 50802, 50813, 50882, 50803) and valuenum is not null
order by f.hadm_id, timestp, f.itemid
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'labs_le.csv'),index=False,sep='|')


# 8. uo (Real-time Urine Output)
query = """
select icustay_id, extract(epoch from charttime) as charttime, itemid, value
from mimiciii.outputevents
where icustay_id is not null and value is not null and itemid in (40055, 43175, 40069, 40094, 40715,
40473, 40085, 40057, 40056, 40405, 40428, 40096, 40651, 226559, 226560, 227510, 226561, 227489,
226584, 226563, 226564, 226565, 226557, 226558)
order by icustay_id, charttime, itemid
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'uo.csv'),index=False,sep='|')


# 9. preadm_uo (Pre-admission Urine Output)
query = """
select distinct oe.icustay_id, extract(epoch from oe.charttime) as charttime, oe.itemid, oe.value , 
60*24*date_part('day',ic.intime-oe.charttime)  + 60*date_part('hour',ic.intime-oe.charttime) + date_part('min',ic.intime-oe.charttime) as datediff_minutes
from mimiciii.outputevents oe, mimiciii.icustays ic
where oe.icustay_id=ic.icustay_id and itemid in (	40060, 226633)	
order by icustay_id, charttime, itemid
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'preadm_uo.csv'),index=False,sep='|')


# 10. fluid_mv (Real-time input from metavision)
# This extraction converts the different rates and dimensions to a common unit
"""
Records with no rate = STAT
Records with rate = INFUSION
fluids corrected for tonicity
"""
query = """
with t1 as
(
select icustay_id, extract(epoch from starttime) as starttime, extract(epoch from endtime) as endtime, itemid, amount, rate,
case when itemid in (30176, 30315) then amount *0.25
when itemid in (30161) then amount *0.3
when itemid in (30020, 30015, 225823, 30321, 30186, 30211, 30353, 42742, 42244, 225159) then amount *0.5 --
when itemid in (227531) then amount *2.75
when itemid in (30143, 225161) then amount *3
when itemid in (30009, 220862) then amount *5
when itemid in (30030, 220995, 227533) then amount *6.66
when itemid in (228341) then amount *8
else amount end as tev -- total equivalent volume
from mimiciii.inputevents_mv
-- only real time items !!
where icustay_id is not null and amount is not null and itemid in (225158, 225943, 226089, 225168,
225828, 220862, 220970, 220864, 225159, 220995, 225170, 225825, 227533, 225161, 227531, 225171, 225827,
225941, 225823, 228341, 30018, 30021, 30015, 30296, 30020, 30066, 30001, 30030,
30060, 30005, 30321, 30006, 30061, 30009, 30179, 30190, 30143, 30160, 30008, 30168, 30186, 30211, 30353, 30159, 30007,
30185, 30063, 30094, 30352, 30014, 30011, 30210, 46493, 45399, 46516, 40850, 30176, 30161, 30381, 30315, 42742, 30180,
46087, 41491, 30004, 42698, 42244)
)
select icustay_id, starttime, endtime, itemid, round(cast(amount as numeric),3) as amount,
round(cast(rate as numeric),3) as rate,round(cast(tev as numeric),3) as tev -- total equiv volume
from t1
order by icustay_id, starttime, itemid
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'fluid_mv.csv'),index=False,sep='|')

# 11. fluid_cv (Real-time input from carevue)
# This extraction converts the different rates and dimensions to a common units
"""
In CAREVUE, all records are considered STAT doses!!
fluids corrected for tonicity
"""
query = """
with t1 as
(
select icustay_id, extract(epoch from charttime) as charttime, itemid, amount,
case when itemid in (30176, 30315) then amount *0.25
when itemid in (30161) then amount *0.3
when itemid in (30020, 30321, 30015, 225823, 30186, 30211, 30353, 42742, 42244, 225159) then amount *0.5
when itemid in (227531) then amount *2.75
when itemid in (30143, 225161) then amount *3
when itemid in (30009, 220862) then amount *5
when itemid in (30030, 220995, 227533) then amount *6.66
when itemid in (228341) then amount *8
else amount end as tev -- total equivalent volume
from mimiciii.inputevents_cv
-- only RT itemids
where amount is not null and itemid in (225158, 225943, 226089, 225168, 225828, 220862, 220970,
220864, 225159, 220995, 225170, 227533, 225161, 227531, 225171, 225827, 225941, 225823,
225825, 228341, 30018, 30021, 30015, 30296, 30020, 30066, 30001, 30030, 30060, 30005, 30321, 30006, 30061,
30009, 30179, 30190, 30143, 30160, 30008, 30168, 30186, 30211, 30353, 30159, 30007, 30185, 30063, 30094, 30352, 30014,
30011, 30210, 46493, 45399, 46516, 40850, 30176, 30161, 30381, 30315, 42742, 30180, 46087, 41491, 30004, 42698, 42244)
order by icustay_id, charttime, itemid
)

select icustay_id, charttime, itemid, round(cast(amount as numeric),3) as amount, round(cast(tev as numeric),3) as tev -- total equivalent volume
from t1

"""
d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'fluid_cv.csv'),index=False,sep='|')

# 12. preadm_fluid (Pre-admission fluid intake)
query = """
with mv as
(
select ie.icustay_id, sum(ie.amount) as sum
from mimiciii.inputevents_mv ie, mimiciii.d_items ci
where ie.itemid=ci.itemid and ie.itemid in (30054, 30055, 30101, 30102, 30103, 30104, 30105, 30108, 226361,
226363, 226364, 226365, 226367, 226368, 226369, 226370, 226371, 226372, 226375, 226376, 227070, 227071, 227072)
group by icustay_id
), cv as
(
select ie.icustay_id, sum(ie.amount) as sum
from mimiciii.inputevents_cv ie, mimiciii.d_items ci
where ie.itemid=ci.itemid and ie.itemid in (30054, 30055, 30101, 30102, 30103, 30104, 30105, 30108, 226361,
226363, 226364, 226365, 226367, 226368, 226369, 226370, 226371, 226372, 226375, 226376, 227070, 227071, 227072)
group by icustay_id
)

select pt.icustay_id,
case when mv.sum is not null then mv.sum
when cv.sum is not null then cv.sum
else null end as inputpreadm
from mimiciii.icustays pt
left outer join mv
on mv.icustay_id=pt.icustay_id
left outer join cv
on cv.icustay_id=pt.icustay_id
order by icustay_id
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'preadm_fluid.csv'),index=False,sep='|')


# 13. vaso_mv (Vasopressors from metavision)
# This extraction converts the different rates and dimensions to a common units
"""
Drugs converted in noradrenaline-equivalent
Body weight assumed 80 kg when missing
"""
query = """
select icustay_id, itemid, extract(epoch from starttime) as starttime, extract(epoch from endtime) as endtime, -- rate, -- ,rateuom,
case when itemid in (30120, 221906, 30047) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3)  -- norad
when itemid in (30120, 221906, 30047) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3)  -- norad
when itemid in (30119, 221289) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3) -- epi
when itemid in (30119, 221289) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3) -- epi
when itemid in (30051, 222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),3) -- vasopressin, in U/h
when itemid in (30051, 222315) and rateuom='units/min' then round(cast(rate*5 as numeric),3) -- vasopressin
when itemid in (30051, 222315) and rateuom='units/hour' then round(cast(rate*5/60 as numeric),3) -- vasopressin
when itemid in (30128, 221749, 30127) and rateuom='mcg/kg/min' then round(cast(rate*0.45 as numeric),3) -- phenyl
when itemid in (30128, 221749, 30127) and rateuom='mcg/min' then round(cast(rate*0.45 / 80 as numeric),3) -- phenyl
when itemid in (221662, 30043, 30307) and rateuom='mcg/kg/min' then round(cast(rate*0.01 as numeric),3)  -- dopa
when itemid in (221662, 30043, 30307) and rateuom='mcg/min' then round(cast(rate*0.01/80 as numeric),3) else null end as rate_std-- dopa
from mimiciii.inputevents_mv
where itemid in (30128, 30120, 30051, 221749, 221906, 30119, 30047, 
  30127, 221289, 222315, 221662, 30043, 30307) and rate is not null and statusdescription <> 'Rewritten'
order by icustay_id, itemid, starttime
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'vaso_mv.csv'),index=False,sep='|')


# 14. vaso_cv (Vasopressors from carevue)
# This extraction converts the different rates and dimensions to a common units
"""
Same comments as above
"""
query = """
select icustay_id,  itemid, extract(epoch from charttime) as charttime, -- rate, -- rateuom,
case when itemid in (30120, 221906, 30047) and rateuom='mcgkgmin' then round(cast(rate as numeric),3) -- norad
when itemid in (30120, 221906, 30047) and rateuom='mcgmin' then round(cast(rate/80 as numeric),3)  -- norad
when itemid in (30119, 221289) and rateuom='mcgkgmin' then round(cast(rate as numeric),3) -- epi
when itemid in (30119, 221289) and rateuom='mcgmin' then round(cast(rate/80 as numeric),3) -- epi
when itemid in (30051, 222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),3) -- vasopressin, in U/h
when itemid in (30051, 222315) and rateuom='Umin' and rate < 0.2 then round(cast(rate*5  as numeric),3) -- vasopressin
when itemid in (30051, 222315) and rateuom='Uhr' then round(cast(rate*5/60  as numeric),3) -- vasopressin
when itemid in (30128, 221749, 30127) and rateuom='mcgkgmin' then round(cast(rate*0.45  as numeric),3) -- phenyl
when itemid in (30128, 221749, 30127) and rateuom='mcgmin' then round(cast(rate*0.45 / 80  as numeric),3) -- phenyl
when itemid in (221662, 30043, 30307) and rateuom='mcgkgmin' then round(cast(rate*0.01   as numeric),3) -- dopa
when itemid in (221662, 30043, 30307) and rateuom='mcgmin' then round(cast(rate*0.01/80  as numeric),3) else null end as rate_std-- dopa
-- case when rateuom='mcgkgmin' then 1 when rateuom='mcgmin' then 2 end as uom
from mimiciii.inputevents_cv
where itemid in (30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289, 222315, 221662, 30043, 30307) and rate is not null
order by icustay_id, itemid, charttime

"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'vaso_cv.csv'),index=False,sep='|')


# 15. mechvent (Mechanical ventilation)
query = """
select
    icustay_id, extract(epoch from charttime) as charttime    -- case statement determining whether it is an instance of mech vent
    , max(
      case
        when itemid is null or value is null then 0 -- can't have null values
        when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
        when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
        when itemid in
          (
          445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
          , 639, 654, 681, 682, 683, 684, 224685, 224684, 224686 -- tidal volume
          , 218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
          , 221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187 -- Insp pressure
          , 543 -- PlateauPressure
          , 5865, 5866, 224707, 224709, 224705, 224706 -- APRV pressure
          , 60, 437, 505, 506, 686, 220339, 224700 -- PEEP
          , 3459 -- high pressure relief
          , 501, 502, 503, 224702 -- PCV
          , 223, 667, 668, 669, 670, 671, 672 -- TCPCV
          , 157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810 -- ETT
          , 224701 -- PSVlevel
          )
          THEN 1
        else 0
      end
      ) as MechVent
      , max(
        case when itemid is null or value is null then 0
          when itemid = 640 and value = 'Extubated' then 1
          when itemid = 640 and value = 'Self Extubation' then 1
        else 0
        end
        )
        as Extubated
      , max(
        case when itemid is null or value is null then 0
          when itemid = 640 and value = 'Self Extubation' then 1
        else 0
        end
        )
        as SelfExtubated

  from mimiciii.chartevents ce
  where value is not null
  and itemid in
  (
      640 -- extubated
      , 720 -- vent type
      , 467 -- O2 delivery device
      , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
      , 639, 654, 681, 682, 683, 684, 224685, 224684, 224686 -- tidal volume
      , 218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
      , 221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187 -- Insp pressure
      , 543 -- PlateauPressure
      , 5865, 5866, 224707, 224709, 224705, 224706 -- APRV pressure
      , 60, 437, 505, 506, 686, 220339, 224700 -- PEEP
      , 3459 -- high pressure relief
      , 501, 502, 503, 224702 -- PCV
      , 223, 667, 668, 669, 670, 671, 672 -- TCPCV
      , 157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810 -- ETT
      , 224701 -- PSVlevel
  )
  group by icustay_id, charttime
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'mechvent.csv'),index=False,sep='|')
