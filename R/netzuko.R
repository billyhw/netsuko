#' Compute the linear predictors to be activated via the activation function
#'
#' @param z The inputs or hidden units
#' @param w The weights
#' @return The linear predictors
#' @note For Internal Use
#' @export
get_s = function(x, w) x %*% w

#' Compute the soft max activation for output predictive probabilities
#' @param a The linear predictors from the last hidden layer
#'
#' @return The output probabilities
#' @note For Internal Use
#' @export
soft_max = function(a) exp(a)/rowSums(exp(a))

#' Compute logistic activation given linear predictors
#'
#' @param s The linear predictors
#' @return The unit activations
#' @note For Internal Use
#' @export
logistic_activation = function(s) 1/(1 + exp(-s))

#' Compute errors from output to the last hidden layer
#'
#' @param y The outputs
#' @param p The predictions
#' @return The error term (a.k.a delta) from the output to the last hidden layer
#' @note For Internal Use
#' @export
get_error_output = function(y, p) y - p

#' Compute errors from a layer to the previous layer
#'
#' @param delta The error from the next layer
#' @param grad_s The gradient of the activation function for linear predictors at the current layer
#' @param The weights associated with the current layer
#' @return The error term to be back-propagated
#' @note For Internal use
#' @export
get_error_hidden = function(delta, grad_s, w) {
  if (nrow(w) > 2) grad_s * tcrossprod(delta, w[-1,])
  else grad_s * (delta %*% w[-1,])
}

#' Compute the gradient of the logistic activation function
#'
#' @param s The linear predictors
#' @return The gradient of logistic activation evaluated at s
#' @note For Internal Use
#' @export
grad_logistic = function(s) logistic_activation(s)*(1-logistic_activation(s))

#' Compute the negative cross-entropy for multi-class classification
#'
#' @param y The outputs
#' @param p The predictions
#' @return The negative cross-entropy
#' @note For Internal Use
#' @export
cross_entropy = function(p, y) -mean(rowSums(y*log(p)))

#' Compute the gradient of the weight for a given layer
#'
#' @param delta The errors passed from the next layer
#' @param x The inputs or the current hidden units
#' @return The gradient of the weights for gradient descent update
#' @note For Internal Use
#' @export
grad_w = function(delta, x) -crossprod(x, delta)/nrow(x)

#' Compute crucial quantities evaluated from one forward-Backward pass through the neural network
#'
#' @param x The inputs
#' @param y The outputs
#' @param w The list of weights: 1st element are connection of weights from input to 1st hidden layer,
#' and the last element are connection weights from the last hidden layer to the outputs
#' @return A list containing the following elements:
#' p: the output probabilities
#' delta: a list of errors backpropagated throught the layers
#' z: the hidden units values
#' @export
forward_backward_pass = function(x, y, w) {

  s_list = vector("list", length(w) - 1)
  z_list = vector("list", length(w))

  z_list[[1]] = x

  # compute the linear predictors and hidden units over the layers

  for (i in 2:length(z_list)) {
    s_list[[i-1]] = get_s(z_list[[i-1]], w[[i-1]])
    z_list[[i]] = cbind(rep(1, nrow(x)), logistic_activation(s_list[[i-1]]))
  }

  # compute the output units

  t = get_s(z_list[[length(z_list)]], w[[length(w)]])
  p = soft_max(t)

  # delta_list stores the delta from
  # output -> last hidden layer -> 2nd last hidden layer etc.

  delta_list = vector("list", length(w))

  # compute the errors from the output-hidden layer

  delta_list[[length(w)]] = get_error_output(y, p)

  # compute the errors from the hidden-input layer propagating backwards

  for (i in (length(w) - 1):1) {
    grad_s = grad_logistic(s_list[[i]])
    delta_list[[i]] = get_error_hidden(delta_list[[i+1]], grad_s, w[[i+1]])
  }

  return(ls = list(p=p, delta = delta_list, z = z_list))

}

#' Fit a neural network using back-propagation
#'
#' @param x_train The training inputs
#' @param y_train The training outputs
#' @param x_test The test inputs
#' @param y_test The test outputs
#' @param num_hidden A vector with length equal the number of hidden layers, and
#' values equal the number of hidden units in the corresponding layer. The default c(2, 2) will fit
#' a neural network with 2 hidden layers with 2 hidden units in each layer.
#' @param iter The number of iterations of gradient descent
#' @param step_size The step size for gradient descent
#' @param lambda The weight decay parameter
#' @param momentum The momentum for the momentum term in gradient descent
#' @param ini_w A list of initial weights. If not provided the function will initialize the weights
#' automatically by simulating from a Gaussian distribution with small variance.
#' @param sparse If the input matrix is sparse, setting sparse to TRUE can speed up the code.
#' @param verbose Will display fitting progress when set to TRUE
#' @return A list containing the following elements:
#'
#' cost_train The training cost by iteration
#'
#' cost_test: The test cost by iteration
#'
#' w: The list of weights at the final iteration
#' @examples
#'set.seed(8)
#'logistic = function(alpha, beta, x) 1/(1 + exp(-(alpha + beta*x)))
#'x_train = matrix(rnorm(300), 100, 3)
#'y_train = factor(rbinom(100, 1, prob = logistic(alpha = 0, beta = 1, x_train[,1])) +
#'                   rbinom(100, 1, prob = logistic(alpha = 0, beta = 1, x_train[,2])))
#'x_test = matrix(rnorm(3000), 1000, 3)
#'y_test = factor(rbinom(1000, 1, prob = logistic(alpha = 0, beta = 1, x_test[,1])) +
#'                  rbinom(1000, 1, prob = logistic(alpha = 0, beta = 1, x_test[,2])))
#'fit = netzuko(x_train, y_train, x_test, y_test, num_hidden = c(3, 3), step_size = 0.01, iter = 200)
#'plot(fit$cost_train, type = "l")
#'lines(fit$cost_test, col = 2)
#'fit_2 = netzuko(x_train, y_train, iter = 200)
#'plot(fit_2$cost_train, type = "l")
#' @export
netzuko = function(x_train, y_train, x_test = NULL, y_test = NULL, num_hidden = c(2, 2),
                          iter = 300, step_size = 0.01, lambda = 1e-5, momentum = 0.9,
                          ini_w = NULL, sparse = FALSE, verbose = F) {

  if ((!is.null(x_test) & is.null(y_test)) | (is.null(x_test) & !is.null(y_test))) {
    stop("x_test and y_test must either be both provided or both NULL")
  }

  num_p = ncol(x_train)
  x_train = cbind(rep(1, nrow(x_train)), x_train)
  y_train = model.matrix(~ y_train - 1)
  cost_train = rep(NA, iter)
  cost_test = NULL

  if (!is.null(x_test) & !is.null(y_test)) {
    x_test = cbind(rep(1, nrow(x_test)), x_test)
    y_test = model.matrix(~ y_test - 1)
    cost_test = rep(NA, iter)
  }

  if (sparse) {
    require(Matrix)
    x_train = Matrix(x_train)
    y_train = Matrix(y_train)
    if (!is.null(x_test) & !is.null(y_test)) {
      x_test = Matrix(x_test)
      y_test = Matrix(y_test)
    }
  }

  if (is.null(ini_w)) {
    num_hidden = c(num_p, num_hidden, ncol(y_train))
    w = vector("list", length(num_hidden) - 1)
    for (i in 1:(length(num_hidden) - 1)) {
      w[[i]] = matrix(rnorm((num_hidden[i] + 1)*num_hidden[i+1], sd = 0.1), num_hidden[i] + 1, num_hidden[i+1])
    }
  }

  fb_train = forward_backward_pass(x_train, y_train, w)
  penalty = lambda/2*sum(sapply(w, function(x) sum(x[-1,]^2)))
  cost_train[1] = cross_entropy(fb_train$p, y_train) + penalty

  if (!is.null(x_test) & !is.null(y_test)) {
    fb_test = forward_backward_pass(x_test, y_test, w)
    cost_test[1] = cross_entropy(fb_test$p, y_test)
  }

  if (verbose) {
    if (!is.null(x_test) & !is.null(y_test)) message("iter = 1, training cost = ", round(cost_train[1], 6),
                                                     " test cost = ", round(cost_test[1], 6))
    else message("iter = 1, training cost = ", round(cost_train[1], 6))
  }

  g_w = vector("list", length(w))

  for (j in 1:length(w)) g_w[[j]] = matrix(0, num_hidden[j] + 1, num_hidden[j+1])

  for (i in 2:iter) {

    for (j in 1:length(w)) {
      pen_w = lambda * w[[j]]
      pen_w[1, ] = 0

      g_w[[j]] = momentum*g_w[[j]] - step_size * (grad_w(fb_train$delta[[j]], fb_train$z[[j]]) + pen_w)

      w[[j]] = w[[j]] + g_w[[j]]
    }

    fb_train = forward_backward_pass(x_train, y_train, w)
    penalty = lambda/2*sum(sapply(w, function(x) sum(x[-1,]^2)))
    cost_train[i] = cross_entropy(fb_train$p, y_train) + penalty

    if (!is.null(x_test) & !is.null(y_test)) {
      fb_test = forward_backward_pass(x_test, y_test, w)
      cost_test[i] = cross_entropy(fb_test$p, y_test)
    }

    if (verbose) {
      if (!is.null(x_test) & !is.null(y_test)) message("iter = ", i, " training cost = ", round(cost_train[i], 6),
                                                       " test cost = ", round(cost_test[i], 6))
      else message("iter = ", i, " training cost = ", round(cost_train[i], 6))
    }
  }

  return(ls = list(cost_train = cost_train, cost_test = cost_test, w = w))

}
