# Sepsis Cohort from MIMIC III

This repo provides code for generating the sepsis cohort from MIMIC III dataset. Our main goal is to facilitate reproducibility of results in the literature. 

This is a pure-python implementation based on a corrected version (by the first contributor below) of the original Matlab repo accompanying "The AI Clinician" paper ([Komorowski, et al](https://www.nature.com/articles/s41591-018-0213-5?sf200531662=1)):

https://github.com/matthieukomorowski/AI_Clinician


### Core updates and modifications to the above repo include:

- Pure python re-implementation;
- Numerous bug fixes;
- Add description to the item IDs (essential to clarify what is what);
- A point-by-point check with the original code to assure same data generation before imputation; 
- Deprecate original imputation, which are not reproducible;
- Add KNN imputation to produce higher quality data.

---

[LICENSE](https://github.com/microsoft/mimic_sepsis/blob/master/LICENSE)



[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

---

## Contributing

This code has been developed as part of the RL4H initiative at MSR Montreal. Most of the core work has been done by

- Jayakumar Subramanian (jayakumar.subramanian@gmail.com), Research Intern, MSR Montreal
- Taylor Killian (twkillian@cs.toronto.edu), Ph.D. Student, University of Toronto

---

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Requirements

We recommend using the [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution for python dependencies. From this standard distribution we use both the `os` and `argparse` libraries. All other needed libraries used in this code can be found in `requirements.txt`.

## How to use

#### 1) MIMIC-III Database
You need to first set up and configure MIMIC III database. The details are provided here:

https://mimic.physionet.org/

The MIMIC database is publicly available; however, accessing MIMIC requires additional steps which are explained at the hosting webpage.

We chose to use a PostgresSQL server to manage the database (hence our use of the `psycopg2` library requirement--see `requirements.txt`). Other options and formats are available, see the [MIMIC repository](https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic) for examples and alternatives.

After downloading and setting up the SQL files and performing all the steps from the physionet link above, you should be able to use this codebase without too much additional set-up. 

#### 2) Run `preprocess.py`

This script accesses the MIMIC database and extracts sub-tables for use in defining the final septic patient cohort in the next step.

There are 43 tables in the Mimic III database, 26 are unique and the other 17 are partitions of chartevents that are not to be queried directly (see: [https://mit-lcp.github.io/mimic-schema-spy/](https://mit-lcp.github.io/mimic-schema-spy/) for further guidance).

Ulitmately, we create 15 sub-tables when extracting from the database. These subtables are stored in a subfolder `processed_files/` that can be created manually. This script will create the subfolder if it doesn't already exist.

The [preamble](https://github.com/microsoft/mimic_sepsis/blob/main/preprocess.py#L17-L26) of this file will likely be the only editing needed to direct toward where a user's access to the MIMIC database is defined as well as where they choose save off the intermediate files.

Depending on the I/O readout speed and network connectivity (assuming that the MIMIC database is saved on a server) this script can take several hours to run completely.

#### 3) Run `sepsis_cohort.py`

Using the sepsis3 criteria, this script uses the preprocessed intermediate tables produced in the prior step to define a cohort of septic patients. This cohort definition was spefically designed for use in sequential decision making purposes, yet this cohort definition code does not partition temporally spaced observations as individual data points. This script instead populates a table of patients who develop sepsis at some point during their treatment in the ICU and includes all observations 24 hours before until 48 hours after presumed onset of sepsis. Further preprocessing is required to represent this data in MDP format, an example where this is done can be found at: https://github.com/MLforHealth/rl_representations/.

External files required: `Reflabs.tsv`, `Refvitals.tsv`, `sample_and_hold.csv` (all saved in the `ReferenceFiles/` sub-folder)

The final cohort table is saved in a user specified location in `.csv` format where the columns are z-normalized. The user can specify to also save off an unormalized copy of the same table.

Note: The size of the cohort will depend on which version of MIMIC-III is used. The original cohort from the 2018 Nature Medicine publication was built using MIMIC-III v1.3.

Again, depending on system characteristics, this script may take 2-3 hours to run to completion.
