# Group7 Dockerfile
FROM jupyter/scipy-notebook:8f0a73e76d17

RUN conda install --yes -c conda-forge\
    pandas=1.4.1 \
    scikit-learn=1.0.2 \
    matplotlib=3.5.1 \
    numpy=1.22.2 \
    pytest=7.0.1 \
    jupyter-book=0.12.1 \
    knitr=1.38\
    reticulate=1.24
    
