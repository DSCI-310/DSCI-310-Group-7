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
The data set we used is the Zoo (1990) provided by UCL Machine Learning Repository. It stores data about 7 classes of animals and their related factors including animal name, hair, feathers and so on. In this project, we picked classification as our method to classify a given animal to its most likely type. All of 16 factors including hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, and catsize were selected as our predictors. To best predict the class of a new observation, we implemented and evaluated models based on a list of algorithms including k-Nearest Neighbor(k-NN), Decision Tree, Support Vector Machine and Logistic Regression. After a comparison among accuracies of different models, we finally found that algorithm k-NN produced the most accurate result of predicting animal type. There are several ways to repeat/reproduce our analysis, please kindly find details in **Usage** section.

Documents about Project Manners:

[code of conduct](CODE_OF_CONDUCT.md)

[contributing](CONTRIBUTING.md)

<br>

## Analysis Results
The final report can be viewed from the following:

[Jbook Html](analysis/_build/html/index.html)

[Jbook Pdf](analysis/_build/latex/python.pdf)

[Rmd Html](doc/zoo_analysis.html)

[Rmd PDF](doc/zoo_analysis.pdf)

<br>

## Dependencies

A list of the dependencies packaged in the image:

| Package Name | Version |
| ------------ | ------- |
| python       | 3.9.7   |
| pandas       | 1.4.1   |
| scikit-learn | 1.0.2   |
| matplotlib   | 3.5.1   |
| numpy        | 1.22.2  |
| pytest       | 7.0.1   |
| R            | 4.1.2   |
| knitr        | 1.38    |
| reticulate   | 1.24    |
| tidyverse    | 1.3.1   |
| jupyter-book | 0.12.1. |


   see [dockerfile](Dockerfile) and [docker image](https://hub.docker.com/repository/docker/sasiburi/dsci-310-group-7)

<br>

## Usage

### 0. Preparation

- You should sign up/on a [Docker](https://hub.docker.com) account.

- Install Docker in your computer.

- **Clone this repo**

  You can find the detailed instructions of repository clone from [here](https://github.com/DSCI-310/DSCI-310-Group-7.git)

  ```
  git clone https://github.com/DSCI-310/DSCI-310-Group-7.git
  ```

  

<br>

Then, choose one of following two suggested ways to repeat/reproduce this analysis:

### 1. Via Docker

**a. Pull down the docker image**

```
docker pull sasiburi/dsci-310-group-7:v2.9.0
```
You can replace `latest` by another specific version.  Please kindly read the **History of Releases** for more versions.

**c. Run the docker image**

`cd` to the **root** of the cloned repo, then run the command:

```
docker run --rm -p 8888:8888 -v ${PWD}:/home/jovyan/work sasiburi/dsci-310-group-7:v2.9.0
```

Open the link provided on your console. Now you should able to see the repo under `work/`. 

*Note: if you see the following error message: `docker: Error response from daemon: Mounts denied: 
The path /YOURPATH is not shared from the host and is not known to Docker.`*

Try following steps:
1. Open Preferences
2. Click Resources -> FILE SHARING
3. Add your /path/to/exported/directory
4. Restart Docker and try the command above again

- If a token is required, it can be obtained on your console. For example, `token=3912f59232fe3b260fda201da4e822e69bfed02e649dc56b`, then `3912f59232fe3b260fda201da4e822e69bfed02e649dc56b` is the token.
- If the port `8888` has been occupied, you can replace the first `8888` by another four-digit number.

<br>



### 2. Via Makefile

**a. Install all the dependencies**

**b. reproduce the analysis**

`cd` to the **root** of the cloned repo, then run the command:

```
make all
```
**c. reset the repo**

```
make clean
```



## History of Releases

Releases relate to each milestone are listed as follows:

1. Milestone 1: [v0.5.0](https://github.com/DSCI-310/DSCI-310-Group-7/releases/tag/v0.5.0)
2. Milestone 2: [v0.10.0](https://github.com/DSCI-310/DSCI-310-Group-7/releases/tag/v0.10.0)
3. Milestone 3: [v2.9.0](https://github.com/DSCI-310/DSCI-310-Group-7/releases/tag/v2.9.0)

more details could be found on the right-hand-side panel, **Releases** or our [docker web](https://hub.docker.com/r/sasiburi/dsci-310-group-7/tags).

<br>

##  [licenses](LICENSE.md):

a. an MIT license for the project code 

b. a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International(CC BY-NC-ND 4.0) license for the project report







