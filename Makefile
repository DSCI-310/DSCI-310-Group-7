# author: DSCI310 2021W2 Group7
# date: 2022-04-01

#includes the items for the exploratory analysis
all : results/figures/fig1.png results/figures/k_accuracy.png results/figures/dt_accuracy.png results/csv/head.csv results/csv/knn_classification_report.csv results/csv/dt_classification_report.csv results/csv/dt_cross_validate_result.csv results/csv/knn_cross_validate_result.csv results/cvs/lr_classification_report.csv results/csv/svm_classification_report.csv analysis/_build/html/index.html

#get the data from the web
data/raw/zoo.csv: src/read_data_script.py
	python src/read_data_script.py "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data" "data/raw/zoo.data"

#pre-process data set
data/processed/zoo.csv: src/pre_processing.py data/raw/zoo.csv
	python src/pre_processing.py data/raw/student-mat.csv 0.2 "data/"

#get exploratory data analysis tables and figures

#perform classification and get reports



analysis/_build/html/index.html: analysis/_config.yml analysis/_toc.yml analysis/exploratory_analysis.ipynb analysis/discussion.ipynb analysis/references.bib
	jupyter-book build analysis/


clean :
	rm -f data/raw/student-mat.csv
	rm -f data/*.csv
	rm -rf results
	rm -rf analysis/_build
