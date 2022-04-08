# Group7 Dockerfile
FROM jupyter/scipy-notebook:8f0a73e76d17

RUN conda install --yes -c conda-forge\
    pandas=1.4.1 \
    scikit-learn=1.0.2 \
    matplotlib=3.5.1 \
    numpy=1.22.2 \
    pytest=7.0.1 \ 
    statsmodels=0.13.2 \
    jupyter-book=0.12.1 \


FROM jupyter/r-notebook:r-4.1.2
RUN Rscript -e "require(devtools); install_version('knitr', version = '1.38', repos = 'http://cran.us.r-project.org')" -1
RUN Rscript -e "require(devtools); install_version('reticulate', version = '1.24', repos = 'http://cran.us.r-project.org')" -1
RUN Rscript -e "require(devtools); install_version('tidyverse', version = '1.3.1', repos = 'http://cran.us.r-project.org')" -1
RUN Rscript -e "require(devtools); install_version('bookdown', version = '0.25', repos = 'http://cran.us.r-project.org')" -1 
