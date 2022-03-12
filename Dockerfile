# Group7 Dockerfile
RUN conda install --yes -c conda-forge \
    pandas=1.4.1 \
    scikit-learn=1.0.2 \
    matplotlib=3.5.1 \
    numpy=1.22.2 \
    pytest=7.0.1 \
    statsmodels=0.13.2

