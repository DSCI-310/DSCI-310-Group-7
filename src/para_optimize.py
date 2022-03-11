#' Optimize hyper-parameters for a model
#'
#' Creates a new data frame with two columns, 
#' listing the classes present in the input data frame,
#' and the number of observations for each class.
#'
#' @param mod A model already defined to be optimized (e.g. KNN).
#' @param params A dictionary of hyper-parameters to be evaluated. 
#'               Keys are hyper-parameter names. Values are the hyper-parameter values
#'               (e.g. dict(n_neighbors=list(range(1, 21)))).
#' @param n The number of folders in Searching. It might change due to data frame size (e.g. 5).
#' @param X_train Dataframe of training set without the targets.
#' @param y_train Dataframe of targets in the training set.
#'
#' @return A dictionary with best hyper-parameters. 
#'   The first column (named class) lists the classes from the input data frame.
#'   The second column (named count) lists the number of observations for each class from the input data frame.
#'   It will have one row for each class present in input data frame.
#'
#' @export
#'
#' @examples
#' para_optimize(knn, param_grid, 5)
Def para_optimize(mod, params, n, X_train, y_train):
    grid = GridSearchCV(mod, params, cv=n, scoring='accuracy')
    grid.fit(X_train, y_train)
    best = grid.best_params_
    return best

