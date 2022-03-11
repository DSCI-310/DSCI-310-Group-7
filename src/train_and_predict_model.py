#' train model and predict based on a specified algorithm
#'
#' @param string the name of algorithm will be used
#' @param integer the number of K
#' @param *array sequence of opriginal results in test set
#' @param *array sequence of predicted results in test set
#'
#' @return an array of accuracies in terms of different Ks based on a specified algorithm
#'
#' @export
#'
#' @examples
#' TrainAndPredictModel('KNN', 81, X_train, y_train, y_test)
def train_and_predict_model(algorithm, Ks, X_train, y_train, y_test):
    for n in range(1, Ks):
        if algorithm == 'DT':
            decTree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
            decTree.fit(X_train,y_train)
            yhat=decTree.predict(X_test)
        elif algorithm == 'KNN':
            neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
            yhat=neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    return mean_acc
    

        

