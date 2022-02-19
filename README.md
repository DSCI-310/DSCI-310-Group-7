# Prediction on the Animal Species
### Group Project Repository for DSCI 310 (Group 7)
<br>

## Contributors / Authors: 

·Elaine Zhou

·Jossie Jiang

·Swakhar Poddar

·Weihao Sun

<br>

## Short Summary
The data set we used is the Zoo (1990) provided by UCL Machine Learning Repositry. It stores data about 7 classes of animals and their related factors inlcuding animal name, hair, feathers and so on. In this project, we picked classification as our method to classify a given animal to its most likely type. We also used multiple analysis to identify the type of the animals using 16 variables including hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, and catsize as our predictors. To best predict the class of a new observation, we implemented and compared a list of algorithms. Every type of algorithms will be mentioned in the main project [analysis](zoo_analysis.ipynb). 

For further detailed information, See:

[code of conduct](CODE_OF_CONDUCT.md)

[contributing](CONTRIBUTING.md)

<br>

## How to Run the Analysis
1. A list of the dependencies needed to run the analysis:
   
   a. Jupyter notebook with python
   
   b. pandas package in version 1.4.1
   
   c. scikit-learn module in version 1.0.2
   
   d. matplotlib package in version 3.5.1
   
   e. numpy package in version 1.22.2
   
   see [dockerfile](Dockerfile) and [docker image](https://hub.docker.com/repository/docker/sasiburi/dsci-310-group-7)
3. Steps to run: 
   
   a. Run "docker run --rm -p 8787:8787 jupyter/scipy-notebook" in terminal
   
   b. Open up a web browser and access the server through http://127.0.0.1:8888/lab?token=399e23fe91a267a070037bd5196feaf0f2decd6e136ffcb0
3. Licenses contained in [license](LICENSE.md):
   
   a. an MIT license for the project code 
   
   b. a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International(CC BY-NC-ND 4.0) license for the project report

   





