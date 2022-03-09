#' Plot a linear relationship
#'
#' 
#' @param integer the number of neighbours we want to try for KNN method to find the best k  
#' @param array a list of numbers which represents the mean accuracy of the KNN neighbour distance
#' @param array a list of numbers which represents the standard deviation of the KNN method's accuracy
#' @param string the label of x axis
#' @param string the label of y axis
#' @param string the name of the plot
#'
#' @return An image of the linear relationship between two parameters of the given data frame. 
#'         The plot should have a x label, a y label and a title.
#'
#' @export
#'
#' @examples
#' line_plot(...)
Def line_plot(k, mean, std, x, y, name): 
    plt.plot(range(1,k),data,'g')
    plt.fill_between(range(1, k), data - 1 * std_acc, data + 1 * data, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(name)
    plt.tight_layout()
    plt.show()
