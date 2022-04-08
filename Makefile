# author: DSCI310 2021W2 Group7
# date: 2022-04-01

#includes the items for the exploratory analysis
all: results/figures/fig1.png results/figures/k_accuracy.png results/figures/dt_accuracy.png results/csv/head.csv results/csv/knn_classification_report.csv results/csv/dt_classification_report.csv results/csv/dt_cross_validate_result.csv results/csv/knn_cross_validate_result.csv results/csv/lr_classification_report.csv results/csv/svm_classification_report.csv analysis/_build/html/index.html analysis/_build/latex/python.pdf doc/zoo_analysis.html doc/zoo_analysis.pdf


#read data and proprocess data
data/processed/zoo.csv: data/raw/zoo.data src/read_data_script.py
	python src/read_data_script.py data/raw/zoo.data data/processed/zoo.csv

#generate exploratory analysis plot
results/figures/fig1.png: data/processed/zoo.csv src/eda_visualization_script.py
	python src/eda_visualization_script.py data/processed/zoo.csv results/figures/fig1.png

#perform classification and get reports based on knn
results/csv/head.csv results/figures/k_accuracy.png results/csv/knn_classification_report.csv: data/processed/zoo.csv src/knn_script.py
	python src/knn_script.py data/processed/zoo.csv results/

#perform classification and get reports based on decistion tree
results/figures/dt_accuracy.png results/csv/dt_classification_report.csv: data/processed/zoo.csv src/dt_script.py
	python src/dt_script.py data/processed/zoo.csv results/

#perform classification and get reports based on svm and linear regresssion
results/csv/lr_classification_report.csv results/csv/svm_classification_report.csv: data/processed/zoo.csv src/svm_lr_script.py
	python src/svm_lr_script.py data/processed/zoo.csv results/

#render reports
analysis/_build/html/index.html: analysis/_config.yml analysis/_toc.yml analysis/analysis.ipynb analysis/discussion.md analysis/references.bib
	jupyter-book build analysis/
analysis/_build/latex/python.pdf: analysis/_config.yml analysis/_toc.yml analysis/analysis.ipynb analysis/discussion.md analysis/references.bib
	jb build analysis/ --builder pdflatex

# render R Markdown report in HTML and PDF
doc/zoo_analysis.html doc/zoo_analysis.pdf: doc/zoo_analysis.Rmd doc/references.bib
	Rscript -e "rmarkdown::render('doc/zoo_analysis.Rmd', c('bookdown::html_document2', 'bookdown::pdf_document2'))"

# remove all the files created by mikefile
clean:
	rm -rf data/processed/zoo.csv
	rm -rf results/csv/*
	rm -rf results/figures/*
	rm -rf analysis/_build
	rm -rf doc/zoo_analysis.html doc/zoo_analysis.pdf
