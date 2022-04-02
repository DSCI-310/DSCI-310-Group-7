# author: DSCI-group7
# date: 2022-04-01

#includes the items for the exploratory data analysis and the final analysis 
all : results/figures/explore_numeric.png results/figures/explore_cat.png results/exploratory-stu-mat.csv results/coeff-table.csv results/figures/predvsfinal.png results/cvtable.csv results/finaltable.csv analysis/_build/html/index.html

#get the data from the web
data/raw/student-mat.csv: src/gatherdata.py
	python src/gatherdata.py "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip" "data/raw/student-mat.csv"

#split data into train test splits
data/student-mat-test.csv data/student-mat-train.csv: src/splitter.py data/raw/student-mat.csv
	python src/splitter.py data/raw/student-mat.csv 0.2 "data/"

#get exploratory data analysis tables and figures
results/figures/explore_numeric.png results/figures/explore_cat.png results/exploratory-stu-mat.csv: src/generate_exploratory.py data/student-mat-test.csv data/student-mat-train.csv
	python src/generate_exploratory.py "data/student-mat-train.csv" "results/"

#perform regression and get final tables and plots
results/coeff-table.csv results/figures/predvsfinal.png results/cvtable.csv results/finaltable.csv: src/regression.py data/student-mat-test.csv data/student-mat-train.csv
	python src/regression.py "data/" "results/"
    
analysis/_build/html/index.html: analysis/_config.yml analysis/_toc.yml analysis/exploratory_analysis.ipynb analysis/methods.ipynb analysis/results.ipynb analysis/references.bib
	jupyter-book build analysis/
    
    
clean :
	rm -f data/raw/student-mat.csv
	rm -f data/*.csv
	rm -rf results
	rm -rf analysis/_build