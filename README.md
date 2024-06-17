# WENDY
code files for WENDY method, used for inferring gene regulatory networks (GRN) from single-cell gene expression data

-----------------------------------------------------------
major code files:

wendy_tutorial.py: a tutorial for using WENDY method

wendy_alg.py: main function of WENDY method, used for testing

wendy_solver.py: the numerical solver used in WENDY method

-----------------------------------------------------------
code files for comparing different methods:

SINC_comparison.py: used to compare different methods on SINC data

DREAM4_comparison.py: used to compare different methods on DREAM4 data

time_comparison.py: used to compare time costs of different methods

THP1_comparison.py: used to compare different methods on THP-1 data

hESC_comparison.py: used to compare different methods on hESC data

-----------------------------------------------------------
code for other methods, for comparison:

sincerities.py: code for SINCERITIES method

xgbgrn.py: code for NonlinearODEs method

GENIE3.py: code for GENIE3 method

dynGENIE3.py: code for dynGENIE3 method

-----------------------------------------------------------
auxiliary code files:

data_generation.py: generate test data

evaluation.py: evaluate inferred GRNs

-----------------------------------------------------------
data sets:

folder SINC: GRNs from https://academic.oup.com/bioinformatics/article/34/2/258/4158033, has 40 numpy matrices (GRNs)

folder DREAM4: GRNs and corresponding expression data from https://www.synapse.org/#!Synapse:syn3049712/wiki/74628, has 10 numpy matrices (GRNs) DREAM4_A....npy and 10 numpy matrices (expression data) DREAM4_data....npy

example_data_0.npy, example_data_1.npy: two example data sets, used for wendy_tutorial.py

folder THP1: GRN (THP1_A.npy) and corresponding expression data (THP1_data.npy) from https://link.springer.com/article/10.1186/gb-2013-14-10-r118

folder hESC: GRN (hESC_A.npy) and corresponding expression data (hESC_data....npy for six time points) from https://link.springer.com/article/10.1186/s13059-016-1033-x
