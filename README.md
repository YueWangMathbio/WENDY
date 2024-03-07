# WENDY
code files for WENDY method, used for inferring gene regulatory networks (GRN) from single-cell gene expression data

major code files:
wendy_tutorial.py: a tutorial for using WENDY method
wendy_alg.py: main function of WENDY method, used for testing
wendy_solver.py: the numerical solver used in WENDY method


code files for comparing different methods:
SINC_comparison.py: used to compare different methods on SINC data
DREAM4_comparison.py: used to compare different methods on DREAM4 data
time_comparison.py: used to compare time costs of different methods


code for other methods, for comparison:
sincerities.py: code for SINCERITIES method
xgbgrn.py: code for NonlinearODEs method
GENIE3.py: code for GENIE3 method
dynGENIE3.py: code for dynGENIE3 method


auxiliary code files:
data_generation.py: generate test data
evaluation.py: evaluate inferred GRNs


data sets:
folder GRN: GRNs from https://academic.oup.com/bioinformatics/article/34/2/258/4158033
has 40 numpy matrices (GRNs)
folder DREAM4: GRNs and corresponding expression data from https://www.synapse.org/#!Synapse:syn3049712/wiki/74628
has 10 numpy matrices (GRNs) DREAM4_A....npy and 10 numpy matrices (expression data) DREAM4_data....npy
example_data_0.npy, example_data_1.npy: two example data sets, used for wendy_tutorial.py


experimental data and code:
exp_data.py: calculate GRNs from experimental data
exp_plot.py: combine figures
grn0.png, grn1.png, grn2.png: figures of inferred GRNs for experimental data
folder Experimental data/Exp data: three experimental data sets, revised from https://github.com/hmatsu1226/SCODE/blob/master/README.md
.../data0: single-cell expression levels of mouse embryonic stem cells, https://www.nature.com/articles/s41467-018-02866-0
.../data1: single-cell expression levels of mouse embryonic fibroblast cells, https://www.nature.com/articles/nature18323
.../data2: single-cell expression levels of human embryonic stem cells, https://link.springer.com/article/10.1186/s13059-016-1033-x
each data set has three files:
.../expdata.txt: gene expression levels
.../tf.txt: name of genes
.../time.txt: measured time and inferred time (not used) of each cell
folder Experimental data/Exp GRN: used to store plotted GRNs
