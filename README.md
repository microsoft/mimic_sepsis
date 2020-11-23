# Sepsis Cohort from MIMIC III

This repo provides code for generating the sepsis cohort from MIMIC III dataset. Our main goal is to facilitate reproducibility of results in the literature. This is a purely python repo based on (a corrected version by the first contributor below) of the following Matlab repo:

https://github.com/matthieukomorowski/AI_Clinician

In addition to numerous bug fixes in the above code, we added KNN imputation to produce a higher quality data.

---

[LICENSE](https://github.com/microsoft/mimic_sepsis/blob/master/LICENSE)



[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

---

## Contributing

This code has been developed as part of RL4H initiative at MSR Montreal. Most of the core work has done by

- Jayakumar Subramanian (jayakumar.subramanian@gmail.com), Research Intern, MSR Montreal
- Taylor Killian (twkillian@gmail.com), Ph.D. Student, University of Toronto

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

We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution for python dependencies. (complete list will be added.)


## How to use

You need to first set up and configure MIMIC III database. The details are provided here:

https://mimic.physionet.org/

Remark that MIMIC is publicly available; however, accessing MIMIC requires additional steps which are explained above. 

After downloading the SQL files and performing all the steps from the link above, you should be able to use this codebase. 
