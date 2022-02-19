# Group7 Dockerfile
FROM jupyter/scipy-notebook 
RUN conda install --yes -c conda-forge pandas=1.4.1
RUN conda install --yes -c conda-forge scikit-learn=1.0.2
RUN conda install --yes -c conda-forge matplotlib=3.5.1
RUN conda install --yes -c conda-forge numpy=1.22.2