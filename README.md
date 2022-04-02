# Prediction on the Animal Species
### Group Project Repository for DSCI 310 (Group 7)
<br>

## Contributors / Authors: 

- Elaine Zhou

- Jossie Jiang

- Swakhar Poddar

- Weihao Sun

<br>

## Abstract
The data set we used is the Zoo (1990) provided by UCL Machine Learning Repositry. It stores data about 7 classes of animals and their related factors inlcuding animal name, hair, feathers and so on. In this project, we picked classification as our method to classify a given animal to its most likely type. All of 16 factors including hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, and catsize were selected as our predictors. To best predict the class of a new observation, we implemented and evalutated models based on a list of algorithms including k-Nearest Neighbor(k-NN), Decision Tree, Support Vector Machine and Logistic Regression. After a comparison among accuracies of different models, we finally found that algorithm k-NN produced the most accurate result of predicting animal type. Please kindly review our [analysis](analysis/zoo_analysis.ipynb) for more details. 

Documents about Project Manners:

[code of conduct](CODE_OF_CONDUCT.md)

[contributing](CONTRIBUTING.md)

<br>

## Usage

There are two suggested ways to run

1. Via Docker

a. Run in your terminal

```
docker pull sasiburi/dsci-310-group-7:latest
```

*Note: You can replace `latest` by another specific version.  Please kindly read the **History of Releases** for more versions.*

<br>

b. Run the docker image

Clone this repo by command:

```
git clone https://github.com/DSCI-310/DSCI-310-Group-7.git
```

*Note*: you can find detailed instructions of repository clone from [here](https://github.com/DSCI-310/DSCI-310-Group-7.git).

Make sure you are at the **root** of the cloned repo, then open it under the reproducible environment by following command:

```
docker run --rm -p 8888:8888 -v ${PWD}:/home/jovyan/work sasiburi/dsci-310-group-7:latest
```

*Note: if port 8888 has been allocated, you can replace 8888 by another random four-digit number.*

<br>

c. Access the file via JupyterLab

Open link provided on the console

*Note: If token is required, it can be obtained on your terminal. For example, `token=3912f59232fe3b260fda201da4e822e69bfed02e649dc56b`, then `3912f59232fe3b260fda201da4e822e69bfed02e649dc56b` is the token.*

<br>



2. Via MakeFile

a. Still Clone this repo. Same code as above.

b. Installed all the dependencies listed as blow.

c. `cd` to the root of the local repo, and then run command:

```
make all
```

d. reset the repo to clean, run the command:

```
make clean
```



## History of Releases

Releases relate to each milestone are listed as follows:

1. Milestone 1: [v0.5.0](https://github.com/DSCI-310/DSCI-310-Group-7/releases/tag/v0.5.0)
2. Milestone 2: [v2.0.0](https://github.com/DSCI-310/DSCI-310-Group-7/releases/tag/v2.0.0)

more details could be found on the right-hand side panel - **Releases**.

<br>

## Dependencies

1. A list of the dependencies packaged in the image:

| Package Name | Version |
| ------------ | ------- |
| python       | 3.9.7   |
| pandas       | 1.4.1   |
| scikit-learn | 1.0.2   |
| matplotlib   | 3.5.1   |
| numpy        | 1.22.2  |
| pytest       | 7.0.1   |

   see [dockerfile](Dockerfile) and [docker image](https://hub.docker.com/repository/docker/sasiburi/dsci-310-group-7)

##  [licenses](LICENSE.md):

a. an MIT license for the project code 

b. a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International(CC BY-NC-ND 4.0) license for the project report







