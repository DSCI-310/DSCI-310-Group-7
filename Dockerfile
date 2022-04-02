# Group7 Dockerfile
FROM jupyter/scipy-notebook:8f0a73e76d17

RUN conda install --yes -c conda-forge\
    pandas=1.4.1 \
    scikit-learn=1.0.2 \
    matplotlib=3.5.1 \
    numpy=1.22.2 \
    pytest=7.0.1 \ 
    jupyter-book=0.12.1 \

RUN apt-get update && apt-get install -y r-base
    
RUN Rscript -e "install.packages('knitr', version='1.38')" \
    Rscript -e "install.packages('reticulate', version='1.24')" \
    Rscript -e "install.packages('tidyverse', version='1.3.1')"
    