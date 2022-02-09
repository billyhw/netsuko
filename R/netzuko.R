#' MNIST Database Training Images
#'
#' Pixel intensities and class labels of 5000 training images from the
#' MNIST database
#'
#' @format A list with 2 elements
#' \describe{
#'   \item{x_train}{A matrix of 784 pixel intensites of 5000 images, normalized to range [0,1]}
#'   \item{y_train}{The digit of each image, from 0 to 9}
#' }
#'
#' @note The data was downloaded from the source below and the links
#' "train-labels-idx1-ubyte" and "train-images-idx3-ubyte". A script from
#' https://github.com/rstudio/tfestimators/blob/master/vignettes/examples/mnist.R
#' was used to decode the original file format to pixel intensities.
#' Only a subset of 5000 training images are included in this file.
#' @source \url{http://yann.lecun.com/exdb/mnist/}
"mnist"

#' Compute the linear predictors to be activated via the activation function
#'
#' @param z The inputs or hidden units
#' @param w The weights
#' @return The linear predictors
#' @note For Internal Use
get_s = function(x, w) x %*% w

#' Compute the soft max activation for output predictive probabilities
#' @param a The linear predictors from the last hidden layer
#'
#' @return The output probabilities
#' @note For Internal Use
soft_max = function(a) exp(a)/rowSums(exp(a))

#' Compute logistic activation given linear predictors
#'
#' @param s The linear predictors
#' @return The unit activations
#' @note For Internal Use
logistic_activation = function(s) 1/(1 + exp(-s))

#' Compute tanh activation given linear predictors
#'
#' @param s The linear predictors
#' @return The unit activations
#' @note For Internal Use
tanh_activation = function(s) tanh(s)

#' Compute ReLU activation given linear predictors
#'
#' @param s The linear predictors
#' @return The unit activations
#' @note For Internal Use
relu_activation = function(s) s*(s > 0)

#' Compute errors from output to the last hidden layer
#'
#' @param y The outputs
#' @param p The predictions
#' @param batch_norm If batch_norm is applied
#' @param x The original hidden units
#' @param gamma The gamma parameter from batch-norm
#' @param mean_x The column means of x
#' @param sd_x The column standard deviation of x (+ epsilon)
#' @return The error term (a.k.a delta) from the output to the last hidden layer
#' @note For Internal Use
get_error_output = function(y, p, batch_norm = F, batch_norm_obj = NULL) {
  delta = y - p
  if (!batch_norm) return(delta)
  else {
    sum_1 = colMeans(delta)
    sum_2 = t(batch_norm_obj$x_centered)*rowMeans(t(delta*batch_norm_obj$x_centered)/(batch_norm_obj$sd_x^2))
    t((t(delta) - sum_1 - sum_2)*batch_norm_obj$gamma/batch_norm_obj$sd_x)
  }
}

#' Compute errors from a layer to the previous layer
#'
#' @param delta The error from the next layer
#' @param grad_s The gradient of the activation function for linear predictors at the current layer
#' @param The weights associated with the current layer
#' @param x The original hidden units
#' @param gamma The gamma parameter from batch-norm
#' @param mean_x The column means of x
#' @param sd_x The column standard deviation of x (+ epsilon)
#' @return The error term to be back-propagated
#' @note For Internal use
get_error_hidden = function(delta, grad_s, w, batch_norm = F, batch_norm_obj = NULL) {
  if (!batch_norm) {
    if (nrow(w) > 2) grad_s * tcrossprod(delta, w[-1,])
    else grad_s * (delta %*% w[-1,])
  }
  else {
    if (!is.null(nrow(w))) delta_beta = grad_s * tcrossprod(delta, w)
    else delta_beta = grad_s * (delta %*% w)
    sum_1 = colMeans(delta_beta)
    sum_2 = t(batch_norm_obj$x_centered)*rowMeans(t(delta_beta*batch_norm_obj$x_centered)/(batch_norm_obj$sd_x^2))
    delta = t((t(delta_beta) - sum_1 - sum_2)*batch_norm_obj$gamma/batch_norm_obj$sd_x)
    return(ls = list(delta = delta, delta_beta = delta_beta))
  }
}

#' Compute the gradient of the logistic activation function
#'
#' @param s The linear predictors
#' @return The gradient of logistic activation evaluated at s
#' @note For Internal Use
grad_logistic = function(s) logistic_activation(s)*(1-logistic_activation(s))

#' Compute the gradient of the tanh activation function
#'
#' @param s The linear predictors
#' @return The gradient of logistic activation evaluated at s
#' @note For Internal Use
grad_tanh = function(s) 1-tanh_activation(s)^2

#' Compute the gradient of the relu activation function
#'
#' @param s The linear predictors
#' @return The gradient of logistic activation evaluated at s
#' @note For Internal Use. We set gradient at 0 to be 0.
grad_relu = function(s) s > 0

#' Compute the negative cross-entropy for multi-class classification
#'
#' @param y The outputs
#' @param p The predictions
#' @return The negative cross-entropy
#' @note For Internal Use
cross_entropy = function(p, y) -mean(rowSums(y*log(p)))

#' Compute the KL-divergence for logistic output
#'
#' @param y The outputs
#' @param p The predictions
#' @return The KL divergence
#' @note For Internal Use
kl_divergence = function(p, y) -mean(rowSums(y*log(p) + (1-y)*log(1-p)))

#' Compute 1/2 least square error for regression
#'
#' @param y The outputs
#' @param p The predictions
#' @return The least square error (halved)
#' @note For Internal Use. Note the least square error is halved
#' to simplify the gradients
least_square = function(p, y) mean(rowSums((y - p)^2))/2

#' Compute the gradient of the weight for a given layer
#'
#' @param delta The errors passed from the next layer
#' @param x The inputs or the current hidden units
#' @return The gradient of the weights for gradient descent update
#' @note For Internal Use
grad_w = function(delta, x) -crossprod(x, delta)/nrow(x)

#' Scale a matrix
#' @param x The matrix to be scaled
#' @param mean_x The means to subtract. Will be computed if NULL
#' @param sd_x The standard deviation to divide. Will be computed if NULL
#' @param intercept Indicates if the first column of x is an intercept (all 1's)
#' @return A list containing the scaled matrix, and the mean
#' and standard deviations used by scaling
#' @note For Internal Use
scale_matrix = function(x, mean_x = NULL, sd_x = NULL, intercept = F) {
  if (is.null(mean_x)) mean_x = colMeans(x)
  if (is.null(sd_x)) sd_x = sqrt(rowSums((t(x) - mean_x)^2)/(nrow(x)-1))
  x_scaled = t((t(x) - mean_x)/sd_x)
  if (intercept) x_scaled[,1] = rep(1, nrow(x))
  return(ls = list(x = x_scaled, mean_x = mean_x, sd_x = sd_x))
}

#' Batch Normalization for re-scaling a set of units
#' @param x The matrix of units to be batch-normalized
#' @param gamma A vector of re-scaling parameters
#' @param beta A vector of re-shifting constants
#' @param epsilon The additive constant for numerical stability
#' @return The batch-normalized matrix of re-scaled units
#' @note For Internal Use. The input matrix should have no intercept.
batch_normalization = function(x, gamma, beta, mean_x = NULL, var_x = NULL, epsilon = 1e-8) {
  if (is.null(mean_x)) mean_x = colMeans(x)
  x_centered = t(x) - mean_x
  if (is.null(var_x)) var_x = rowMeans(x_centered^2)
  sd_x = sqrt(var_x + epsilon)
  x_scaled = x_centered/sd_x
  x_rescaled = t(x_scaled * gamma + beta)
  return(ls = list(x_rescaled = x_rescaled, x_centered = t(x_centered),
                   x_scaled = t(x_scaled), mean_x = mean_x, var_x = var_x, sd_x = sd_x, gamma = gamma))
}

#' Initialize weights
#'
#' @param x_train The training inputs
#' @param y_train The training outputs
#' @param num_hidden Vector with number of hidden units for each hidden layer
#' @param method The initialization method
#' @param batch_norm Whether batch-normalization is applied
#' @return Initial weights
#' @note For Internal Use
initialize_weights = function(x_train, y_train, num_hidden, method, batch_norm = F) {
  w = vector("list", length(num_hidden) - 1)
  if (batch_norm) gamma = beta = vector("list", length(num_hidden) - 1)

  for (i in 1:(length(num_hidden) - 1)) {
    if (method == "gaussian") {
      w[[i]] = matrix(rnorm((num_hidden[i] + 1)*num_hidden[i+1], sd = 0.1),
                      num_hidden[i] + 1, num_hidden[i+1])
    }
    else if (method == "uniform") {
      w[[i]] = matrix(runif((num_hidden[i] + 1)*num_hidden[i+1],
                            min = -1/sqrt(num_hidden[i]), max = 1/sqrt(num_hidden[i])),
                      num_hidden[i] + 1, num_hidden[i+1])
    }
    else if (method == "normalized") {
      w[[i]] = matrix(runif((num_hidden[i] + 1)*num_hidden[i+1],
                            min = -sqrt(6/(num_hidden[i] + num_hidden[i+1])),
                            max = sqrt(6/(num_hidden[i] + num_hidden[i+1]))),
                      num_hidden[i] + 1, num_hidden[i+1])
    }
    if (batch_norm) w[[i]] = w[[i]][-1,]
    else w[[i]][1,] = 0
  }

  if (batch_norm) {
    for (i in 1:(length(num_hidden) - 1)) {
      if (method == "gaussian") {
        gamma[[i]] = rnorm(num_hidden[i+1], sd = 0.1)
        beta[[i]] = rnorm(num_hidden[i+1], sd = 0.1)
      }
      else if (method == "uniform") {
        gamma[[i]] = runif(num_hidden[i+1], min = -1/sqrt(num_hidden[i]), max = 1/sqrt(num_hidden[i]))
        beta[[i]] = runif(num_hidden[i+1], min = -1/sqrt(num_hidden[i]), max = 1/sqrt(num_hidden[i]))
      }
      else if (method == "normalized") {
        gamma[[i]] = rep(1, num_hidden[i+1])
        # gamma[[i]] = runif(num_hidden[i+1],
        #                       min = -sqrt(6/(num_hidden[i] + num_hidden[i+1])),
        #                       max = sqrt(6/(num_hidden[i] + num_hidden[i+1])))
        # beta[[i]] = runif(num_hidden[i+1],
        #                    min = -sqrt(6/(num_hidden[i] + num_hidden[i+1])),
        #                    max = sqrt(6/(num_hidden[i] + num_hidden[i+1])))
        beta[[i]] = rep(0, num_hidden[i+1])
      }
    }
  }
  if (!batch_norm) return(w)
  else return(ls = list(w = w, gamma = gamma, beta = beta))
}

#' Initialize weights by Stacking Autoencoders
#'
#' @param x_train The training inputs
#' @param y_train The training outputs
#' @param num_hidden Vector with number of hidden units for each hidden layer
#' @param method The initialization method
#' @return Initial weights
#' @note For Internal Use
pretrain = function(x_train, y_train, num_hidden,
                    iter = 300, step_size = 0.01,
                    lambda = 1e-5, momentum = 0.9,
                    ini_method = c("normalized", "uniform", "gaussian"),
                    sparse = FALSE, verbose = F) {
  w = initialize_weights(x_train, y_train, num_hidden, ini_method)
  pred = as.matrix(x_train)
  for (i in 1:(length(w) - 1)) {
    if (verbose) message("Pretraining layer ", i)
    tmp = netzuko(pred, pred, num_hidden = ncol(w[[i]]),
                  iter = iter, step_size = step_size, lambda = lambda, momentum = momentum,
                  output = "logistic", activation = "logistic", verbose = verbose)
    pred = as.matrix(predict(tmp, as.matrix(pred), type = "hidden")[[2]][,-1])
    w[[i]] = tmp$w[[1]]
  }
  w
}

#' Compute crucial quantities evaluated from one forward-Backward pass through the neural network
#'
#' @param x The inputs
#' @param y The outputs
#' @param w The list of weights: 1st element are connection of weights from input to 1st hidden layer,
#' and the last element are connection weights from the last hidden layer to the outputs
#' @param output_type The output type: either "numeric" (regression) or "categorical" (prediction).
#' @param dropout Boolean to indicate whether dropout is used
#' @param retain_rate The proportion of units to retain during dropout
#' @param forward_only If TRUE the function will only evaluate the forward pass
#' @param batch_norm If TRUE, batch-normalization will be applied on the linear predictors
#' @param gamma The scale parameters for batchnorm
#' @param beta The location parameters for batchnorm
#' @return A list containing the following elements:
#' p: the output probabilities
#' delta: a list of errors backpropagated throught the layers
#' z: the hidden units values
forward_backward_pass = function(x, y, w, activation, output_type,
                                 dropout = FALSE, retain_rate = NULL,
                                 batch_norm = F, gamma = NULL, beta = NULL,
                                 forward_only = FALSE, mean_x = NULL, var_x = NULL) {

  if (activation == "logistic") {
    activation_func = logistic_activation
    grad_func = grad_logistic
  }

  if (activation == "tanh") {
    activation_func = tanh_activation
    grad_func = grad_tanh
  }

  if (activation == "relu") {
    activation_func = relu_activation
    grad_func = grad_relu
  }

  s_list = vector("list", length(w) - 1)
  z_list = vector("list", length(w))
  b_list = NULL
  if (batch_norm) b_list = vector("list", length(w))

  z_list[[1]] = x

  # compute the linear predictors and hidden units over the layers

  for (i in 2:length(z_list)) {
    if (dropout) {
      r = rbinom(ncol(z_list[[i-1]]), 1, retain_rate)
      if (!batch_norm) r[1] = 1
      z_dropout = z_list[[i-1]]
      z_dropout[, which(r == 0)] = 0
      if (batch_norm) {
        s = get_s(z_dropout, w[[i-1]])
        b_list[[i-1]] = batch_normalization(s, gamma[[i-1]], beta[[i-1]],
                                            mean_x = mean_x[[i-1]], var_x = var_x[[i-1]])
        s_list[[i-1]] = b_list[[i-1]]$x_rescaled
      }
      else s_list[[i-1]] = get_s(z_dropout, w[[i-1]])
    }
    else {
      if (batch_norm) {
        s = get_s(z_list[[i-1]], w[[i-1]])
        b_list[[i-1]] = batch_normalization(s, gamma[[i-1]], beta[[i-1]],
                                            mean_x = mean_x[[i-1]], var_x = var_x[[i-1]])
        s_list[[i-1]] = b_list[[i-1]]$x_rescaled
      }
      else s_list[[i-1]] = get_s(z_list[[i-1]], w[[i-1]])
    }
    if (batch_norm) z_list[[i]] = activation_func(s_list[[i-1]])
    else z_list[[i]] = cbind(rep(1, nrow(x)), activation_func(s_list[[i-1]]))
  }

  # compute the output units

  if (batch_norm) {
    s = get_s(z_list[[length(z_list)]], w[[length(w)]])
    b_list[[length(b_list)]] = batch_normalization(s, gamma[[length(w)]], beta[[length(w)]],
                                                   mean_x = mean_x[[length(w)]], var_x = var_x[[length(w)]])
    s = b_list[[length(b_list)]]$x_rescaled
  }
  else s = get_s(z_list[[length(z_list)]], w[[length(w)]])

  if (output_type == "categorical") p = soft_max(s)
  else if (output_type == "logistic") p = logistic_activation(s)
  else if (output_type == "numeric") p = s

  if (forward_only) return(ls = list(p=p, delta = NULL, z = z_list, s = s_list))

  # delta_list stores the delta from
  # output -> last hidden layer -> 2nd last hidden layer etc.

  delta_list = vector("list", length(w))
  delta_beta_list = NULL
  if (batch_norm) delta_beta_list = vector("list", length(w))

  # compute the errors from the output-hidden layer

  if (batch_norm) {
    delta_list[[length(w)]] = get_error_output(y, p, batch_norm = batch_norm,
                                               batch_norm_obj = b_list[[length(b_list)]])
    delta_beta_list[[length(w)]] = get_error_output(y, p, batch_norm = F)
  }
  else delta_list[[length(w)]] = get_error_output(y, p)

  # compute the errors from the hidden-input layer propagating backwards

  for (i in (length(w) - 1):1) {
    grad_s = grad_func(s_list[[i]])
    if (batch_norm) {
      tmp = get_error_hidden(delta_list[[i+1]], grad_s, w[[i+1]], batch_norm = batch_norm,
                             batch_norm_obj = b_list[[i]])
      delta_list[[i]] = tmp$delta
      delta_beta_list[[i]] = tmp$delta_beta
    }
    else delta_list[[i]] = get_error_hidden(delta_list[[i+1]], grad_s, w[[i+1]])
  }

  return(ls = list(p=p, delta = delta_list, z = z_list, s = s_list, b = b_list, delta_beta = delta_beta_list))

}

#' Make Predictions on a test set
#'
#' @param nn_fit A fitted neural network object from netzuko
#' @param newdata The test inputs
#' @param type The prediction type. When type = "prob" (default) the output is a matrix of class
#' probabilities. When type = "class", the output is the class with the highest predictive probability
#' @return A matrix of output probabilities
#' @note This function is essentially the forward pass of a neural network.
#' @examples
#'set.seed(8)
#'logistic = function(alpha, beta, x) 1/(1 + exp(-(alpha + beta*x)))
#'x_train = matrix(rnorm(300), 100, 3)
#'y_train = factor(rbinom(100, 1, prob = logistic(alpha = 0, beta = 1, x_train[,1])) +
#'                   rbinom(100, 1, prob = logistic(alpha = 0, beta = 1, x_train[,2])))
#'x_test = matrix(rnorm(3000), 1000, 3)
#'y_test = factor(rbinom(1000, 1, prob = logistic(alpha = 0, beta = 1, x_test[,1])) +
#'                  rbinom(1000, 1, prob = logistic(alpha = 0, beta = 1, x_test[,2])))
#'fit = netzuko(x_train, y_train, x_test, y_test, num_hidden = c(3, 3), step_size = 0.01, iter = 100)
#'pred = predict(fit, x_test)
#'fit$cost_test[100]
#'-mean(rowSums(model.matrix(~ y_test - 1)*log(pred))) # negative cross entropy
#'y_train = factor(rbinom(100, 1, prob = logistic(alpha = 0, beta = 1, x_train[,1])))
#'y_test = factor(rbinom(1000, 1, prob = logistic(alpha = 0, beta = 1, x_test[,1])))
#'fit_2 = netzuko(x_train[,1], y_train, x_test[,1], y_test, iter = 100, num_hidden = 2)
#'pred_2 = predict(fit_2, x_test[,1])
#'fit_2$cost_test[100]
#'-mean(rowSums(model.matrix(~ y_test - 1)*log(pred_2))) # negative cross entropy
#'x_train = matrix(rnorm(300), 100, 3)
#'y_train = x_train[,1]^2
#'x_test = matrix(rnorm(3000), 1000, 3)
#'y_test = x_test[,1]^2
#'fit_3 = netzuko(x_train, y_train, x_test, y_test, step_size = 0.003, iter = 100)
#'pred_3 = predict(fit_3, x_test)
#'fit_3$cost_test[100]
#'mean((y_test - pred_3)^2)/2 # halved mean square error
#' \dontrun{
#' fit_4 = netzuko(mnist$x_train[1:1000,], mnist$y_train[1:1000],
#' num_hidden = 100, step_size = 0.01, iter = 100, sparse = T)
#' pred_4 = predict(fit_4, mnist$x_train[1001:2000,], type = "class")
#' mean(pred_4 == mnist$y_train[1001:2000])
#' }
#' @export
predict.netzuko = function(nn_fit, newdata, type = c("prob", "class", "hidden")) {

  # if (is.vector(newdata) | is.null(dim(newdata))) newdata = matrix(newdata, ncol = 1)

  type = match.arg(type)
  #newdata = cbind(rep(1, nrow(newdata)), newdata)
  newdata = model.matrix(~ newdata)
  if (!is.null(nn_fit$mean_x) & !is.null(nn_fit$sd_x)) {
    newdata = scale_matrix(newdata, mean_x = nn_fit$mean_x, sd_x = nn_fit$sd_x, intercept = T)$x
  }

  activation = nn_fit$activation
  w = nn_fit$w
  if (nn_fit$dropout) {
    retain_rate = nn_fit$retain_rate
    w = lapply(w, function(x) {
      x[-1,] = retain_rate*x[-1,]
      x
    })
  }

  if (activation == "logistic") activation_func = logistic_activation
  if (activation == "tanh") activation_func = tanh_activation
  if (activation == "relu") activation_func = relu_activation

  s_list = vector("list", length(w) - 1)
  z_list = vector("list", length(w))

  z_list[[1]] = newdata

  for (i in 2:length(z_list)) {
    s_list[[i-1]] = get_s(z_list[[i-1]], w[[i-1]])
    z_list[[i]] = cbind(rep(1, nrow(newdata)), activation_func(s_list[[i-1]]))
  }

  if (type == "hidden") return(z_list)

  s = get_s(z_list[[length(z_list)]], w[[length(w)]])
  if (nn_fit$output_type == "categorical") p = soft_max(s)
  if (nn_fit$output_type == "logistic") p = logistic_activation(s)
  else if (nn_fit$output_type == "numeric") {
    if (is.null(nn_fit$mean_y) | is.null(nn_fit$sd_y)) return(s)
    else return(t(t(s)*nn_fit$sd_y + nn_fit$mean_y))
  }

  if (type == "prob" | nn_fit$output_type == "logistic") return(p)
  else if (type == "class") {
    max_ind = apply(p, 1, which.max)
    return(factor(nn_fit$y_levels[max_ind], levels = nn_fit$y_levels))
  }
}

#' Fit a neural network using back-propagation
#'
#' @param x_train The training inputs
#' @param y_train The training outputs
#' @param x_test The test inputs
#' @param y_test The test outputs
#' @param output_type The output type: either "numeric" (regression) or "categorical" (prediction).
#' If NULL the function will try to guess the output type based on y_train
#' @param num_hidden A vector with length equal the number of hidden layers, and
#' values equal the number of hidden units in the corresponding layer. The default c(2, 2) will fit
#' a neural network with 2 hidden layers with 2 hidden units in each layer.
#' @param iter The number of iterations of gradient descent
#' @param activation The hidden unit activation function (Tanh, ReLU, or Logistic)
#' @param step_size The step size for gradient descent
#' @param batch_size The batch size for stochastic gradient descent.
#' If NULL, run (non-stochastic) gradient descent
#' @param lambda The weight decay parameter
#' @param momentum The momentum for the momentum term in gradient descent
#' @param dropout If dropout should be used
#' @param retain_rate If dropout is used, the retain rate for the input and hidden units
#' @param adam If ADAM should be used for weight updates
#' @param beta_1 A parameter for ADAM
#' @param beta_2 A parameter for ADAM
#' @param epsilon A parameter for ADAM
#' @param ini_w A list of initial weights. If not provided the function will initialize the weights
#' automatically by simulating from a Gaussian distribution with small variance.
#' @param ini_method The initialization method
#' @param batch_norm If TRUE, batch-normalization will be applied
#' @param ini_gamma The gamma parameters for batch-normalization
#' @param ini_beta The beta parameters for batch-normalization
#' @param sparse If the input matrix is sparse, setting sparse to TRUE can speed up the code.
#' @param verbose Will display fitting progress when set to TRUE
#' @param keep_grad Save the gradients at each iteration? (For research purpose)
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
#'fit_3 = netzuko(x_train, y_train, x_test, y_test, iter = 200, activation = "logistic")
#'plot(fit_3$cost_train, type = "l")
#'lines(fit$cost_test, col = 2)
#'y_train = factor(rbinom(100, 1, prob = logistic(alpha = 0, beta = 1, x_train[,1])))
#'y_test = factor(rbinom(1000, 1, prob = logistic(alpha = 0, beta = 1, x_test[,1])))
#'fit_4 = netzuko(x_train[,1], y_train, x_test[,1], y_test, iter = 200, num_hidden = 2)
#'plot(fit_4$cost_train, type = "l", ylim = range(c(fit_4$cost_train, fit_4$cost_test)))
#'lines(fit_4$cost_test, col = 2)
#'x_train = matrix(rnorm(300), 100, 3)
#'y_train = x_train[,1]^2
#'x_test = matrix(rnorm(3000), 1000, 3)
#'y_test = x_test[,1]^2
#'fit_5 = netzuko(x_train, y_train, x_test, y_test, step_size = 0.003, iter = 200)
#'plot(fit_5$cost_train, type = "l")
#'lines(fit_5$cost_test, col = 2)
#'y_train = cbind(y_train, x_train[,2]^2)
#'y_test = cbind(y_test, x_test[,2]^2)
#'fit_6 = netzuko(x_train, y_train, x_test, y_test, step_size = 0.003, iter = 200)
#'plot(fit_6$cost_train, type = "l")
#'lines(fit_6$cost_test, col = 2)
#'pred_6 = predict(fit_6, x_test)
#'fit_6$cost_test[200]
#'mean(rowSums((y_test - pred_6)^2))/2
#'fit_7 = netzuko(x_train, y_train, x_test, y_test, step_size = 0.01, iter = 500, scale = T)
#'plot(fit_7$cost_train, type = "l")
#'lines(fit_7$cost_test, col = 2)
#'pred_7 = predict(fit_7, x_test)
#'fit_7$cost_test[500]
#'tmp = scale_matrix(y_train, intercept = F)
#'mean(rowSums((scale_matrix(y_test, tmp$mean_x, tmp$sd_x, intercept = F)$x - scale_matrix(pred_7, tmp$mean_x, tmp$sd_x, intercept = F)$x)^2))/2
#' @export
#' @import Matrix
netzuko = function(x_train, y_train, x_test = NULL, y_test = NULL, output_type = NULL, num_hidden = c(2, 2),
                   iter = 300, activation = c("relu", "tanh", "logistic"), step_size = 0.01, batch_size = 128,
                   lambda = 1e-5, momentum = 0.9, dropout = FALSE, retain_rate = 0.5,
                   adam = FALSE, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8,
                   ini_w = NULL, ini_method = c("normalized", "uniform", "gaussian", "pretrain"),
                   batch_norm = F, ini_gamma = NULL, ini_beta = NULL, epsilon_batch_norm = 1e-8,
                   scale = FALSE, sparse = FALSE, verbose = F, keep_grad = F) {

  # if (is.vector(x_train) | is.null(dim(x_train))) x_train = matrix(x_train, ncol = 1)

  if (is.null(output_type)) {
    if (is.numeric(y_train)) {
      output_type = "numeric"
      message("output_type not specified, set to numeric")
    }
    else if (is.factor(y_train)) {
      output_type = "categorical"
      message("output_type not specified, set to categorical")
    }
    else stop("output_type must be one of numeric, categorical, or logistic")
  }

  if (output_type == "categorical") cost_func = cross_entropy
  else if (output_type == "logistic") cost_func = kl_divergence
  else if (output_type == "numeric") {
    cost_func = least_square
    # if (is.vector(y_train) | is.null(dim(y_train))) y_train = matrix(y_train, ncol = 1)
  }
  else stop("output_type must be one of numeric or categorical")

  if (output_type == "categorical") y_levels = levels(y_train)
  else y_levels = NULL

  activation = match.arg(activation)
  ini_method = match.arg(ini_method)

  if ((!is.null(x_test) & is.null(y_test)) | (is.null(x_test) & !is.null(y_test))) {
    stop("x_test and y_test must either be both provided or both NULL")
  }

  # num_p = ncol(x_train)
  #x_train = cbind(rep(1, nrow(x_train)), x_train)
  if (batch_norm) x_train = model.matrix(~ x_train - 1)
  else x_train = model.matrix(~ x_train)
  y_train = model.matrix(~ y_train - 1)
  cost_train = rep(NA, iter)
  cost_test = NULL

  if (!is.null(x_test) & !is.null(y_test)) {
    # if (is.vector(x_test) | is.null(dim(x_test))) x_test = matrix(x_test, ncol = 1)
    if (batch_norm) x_test = model.matrix(~ x_test - 1)
    else x_test = model.matrix(~ x_test)
    y_test = model.matrix(~ y_test - 1)
    cost_test = rep(NA, iter)
  }

  mean_x = NULL
  sd_x = NULL
  mean_y = NULL
  sd_y = NULL

  if (scale) {
    x_train = scale_matrix(x_train, intercept = T)
    mean_x = x_train$mean_x
    sd_x = x_train$sd_x
    x_train = x_train$x
    if (output_type == "numeric") {
      y_train = scale_matrix(y_train, intercept = F)
      mean_y = y_train$mean_x
      sd_y = y_train$sd_x
      y_train = y_train$x
    }
    if (any(sd_x[-1] == 0) | any(sd_y == 0)) {
      stop("Training data contains inputs/outputs with 0 standard deviation.
           Please check your data or set scale = F")
    }
    if (!is.null(x_test) & !is.null(y_test)) {
      x_test = scale_matrix(x_test, mean_x, sd_x, intercept = T)$x
      if (output_type == "numeric") y_test = scale_matrix(y_test, mean_y, sd_y, intercept = F)$x
    }
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

  # if (is.null(ini_w)) {
  #   num_hidden = c(ncol(x_train)-1, num_hidden, ncol(y_train))
  #   w = vector("list", length(num_hidden) - 1)
  #   for (i in 1:(length(num_hidden) - 1)) {
  #     w[[i]] = matrix(rnorm((num_hidden[i] + 1)*num_hidden[i+1], sd = 0.1), num_hidden[i] + 1, num_hidden[i+1])
  #   }
  # }

  if (batch_norm) num_hidden = c(ncol(x_train), num_hidden, ncol(y_train))
  else num_hidden = c(ncol(x_train)-1, num_hidden, ncol(y_train))

  if (is.null(ini_w)) {
    if (ini_method == "pretrain") {
      if (activation != "logistic") {
      stop("pretraining currently only supports logistic activation,
           please set activation to logistic and restart")
      }
      else if (min(x_train) < 0 | max(x_train > 1)) {
        stop("pretraining currently only supports inputs with range between (0, 1),
             please rescale inputs if appropriate")
      }
      else {
        w = pretrain(x_train[,-1], y_train, num_hidden,
                   iter = iter, step_size = step_size,
                   lambda = lambda, momentum = momentum,
                   ini_method = "normalized",
                   sparse = sparse, verbose = verbose)
      }
    }
    else w = initialize_weights(x_train, y_train, num_hidden, method = ini_method, batch_norm = batch_norm)
    if (!batch_norm) ini_w = w
    else {
      ini_w = w$w
      ini_gamma = w$gamma
      ini_beta = w$beta
      w = ini_w
      gamma = ini_gamma
      beta = ini_beta
    }
  }
  else {
    w = ini_w
    gamma = ini_gamma
    beta = ini_beta
  }

  if (is.null(batch_size)) batch_size = nrow(x_train)
  if (batch_size > nrow(x_train)) stop("batch_size must be NULL (for non-stochastic gradient descent)
                                       or less than or equal to the number of training samples")

  ind_mat = matrix(1:nrow(x_train), nrow = batch_size)

  #ind = sample(1:nrow(x_train), batch_size)
  ind = ind_mat[, 1]

  if (dropout) {
    if (batch_norm) {
      fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type,
                                       dropout = T, retain_rate = retain_rate,
                                       batch_norm = T, gamma = gamma, beta = beta)
    }
    else fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type,
                                                dropout = T, retain_rate = retain_rate)
  }
  else {
    if (batch_norm) {
      fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type,
                                       batch_norm = T, gamma = gamma, beta = beta)
      }
    else fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type)
  }

  batch_mean = NULL
  batch_var = NULL
  if (batch_norm) {
    batch_mean = lapply(fb_train$b, function(x) 0.1*x$mean_x)
    batch_var = lapply(fb_train$b, function(x) 0.1*x$var_x)
  }

  if (batch_norm) penalty = lambda/2*sum(sapply(w, function(x) sum(x^2)))
  else penalty = lambda/2*sum(sapply(w, function(x) sum(x[-1,]^2)))
  cost_train[1] = cost_func(fb_train$p, y_train[ind,]) + penalty

  if (!is.null(x_test) & !is.null(y_test)) {
    if (batch_norm) fb_test = forward_backward_pass(x_test, y_test, w, activation, output_type, batch_norm = batch_norm,
                                                    forward_only = T, gamma = gamma, beta = beta,
                                                    mean_x = batch_mean, var_x = batch_var)
    else fb_test = forward_backward_pass(x_test, y_test, w, activation, output_type, forward_only = T)
    cost_test[1] = cost_func(fb_test$p, y_test)
  }

  if (verbose) {
    if (!is.null(x_test) & !is.null(y_test)) message("iter = 1, training cost = ", round(cost_train[1], 6),
                                                     " test cost = ", round(cost_test[1], 6))
    else message("iter = 1, training cost = ", round(cost_train[1], 6))
  }

  g_hist = NULL
  beta_g_hist = NULL
  gamma_g_hist = NULL
  if (keep_grad) {
    g_hist = vector("list", iter)
    beta_g_hist = vector("list", iter)
    gamma_g_hist = vector("list", iter)
  }
  # z_hist = vector("list", num_iter)
  g_w = vector("list", length(w))
  grad = vector("list", length(w))
  g_beta = vector("list", length(w))
  grad_beta = vector("list", length(w))
  g_gamma = vector("list", length(w))
  grad_gamma = vector("list", length(w))

  for (j in 1:length(w)) {
    if (batch_norm) {
      g_w[[j]] = matrix(0, num_hidden[j], num_hidden[j+1])
      g_beta[[j]] = rep(0, num_hidden[j+1])
      g_gamma[[j]] = rep(0, num_hidden[j+1])
    }
    else g_w[[j]] = matrix(0, num_hidden[j] + 1, num_hidden[j+1])
  }
  if (keep_grad) {
    g_hist[[1]] = NA
    beta_g_hist[[1]] = NA
    gamma_g_hist[[1]] = NA
  }
  # z_hist[[1]] = fb_test$z

  if (adam) {
    # m = v = m_hat = v_hat = vector("list", length(w))
    m = v = vector("list", length(w))
    if (batch_norm) m_gamma = v_gamma = m_beta = v_beta = vector("list", length(w))

    for (j in 1:length(w)) {
      if (batch_norm) {
        m[[j]] = matrix(0, num_hidden[j], num_hidden[j+1])
        v[[j]]  = matrix(0, num_hidden[j], num_hidden[j+1])
        m_gamma[[j]] = rep(0, num_hidden[j+1])
        v_gamma[[j]]  = rep(0, num_hidden[j+1])
        m_beta[[j]] = rep(0, num_hidden[j+1])
        v_beta[[j]]  = rep(0, num_hidden[j+1])
      }
      else {
        m[[j]] = matrix(0, num_hidden[j] + 1, num_hidden[j+1])
        v[[j]]  = matrix(0, num_hidden[j] + 1, num_hidden[j+1])
      }
      # m_hat[[j]] = matrix(0, num_hidden[j] + 1, num_hidden[j+1])
      # v_hat[[j]] = matrix(0, num_hidden[j] + 1, num_hidden[j+1])
    }
  }

  for (i in 2:iter) {

    for (j in 1:length(w)) {
      pen_w = lambda * w[[j]]
      if (!batch_norm) pen_w[1, ] = 0

      grad[[j]] = grad_w(fb_train$delta[[j]], fb_train$z[[j]]) + pen_w
      if (adam) {
        m[[j]] = beta_1 * m[[j]] + (1 - beta_1) * grad[[j]]
        v[[j]] = beta_2 * v[[j]] + (1 - beta_2) * grad[[j]]^2
        # m_hat[[j]] = m[[j]]/(1 - beta_1^(i-1))
        # v_hat[[j]] = v[[j]]/(1 - beta_2^(i-1))
        # g_w[[j]] = -step_size*m_hat[[j]]/(sqrt(v_hat[[j]]) + epsilon)
        step_size_hat = step_size * sqrt(1 - beta_2^(i-1))/(1 - beta_1^(i-1))
        g_w[[j]] = -step_size_hat*m[[j]]/(sqrt(v[[j]]) + epsilon)
      }
      else g_w[[j]] = momentum*g_w[[j]] - step_size * grad[[j]]

      w[[j]] = w[[j]] + g_w[[j]]

      if (batch_norm) {
        grad_beta[[j]] = -colMeans(fb_train$delta_beta[[j]])
        grad_gamma[[j]] = -colMeans(fb_train$delta_beta[[j]]*fb_train$b[[j]]$x_scaled)
        if (adam) {
          m_beta[[j]] = beta_1 * m_beta[[j]] + (1 - beta_1) * grad_beta[[j]]
          v_beta[[j]] = beta_2 * v_beta[[j]] + (1 - beta_2) * grad_beta[[j]]^2
          # step_size_hat = step_size * sqrt(1 - beta_2^(i-1))/(1 - beta_1^(i-1))
          g_beta[[j]] = -step_size_hat*m_beta[[j]]/(sqrt(v_beta[[j]]) + epsilon)

          m_gamma[[j]] = beta_1 * m_gamma[[j]] + (1 - beta_1) * grad_gamma[[j]]
          v_gamma[[j]] = beta_2 * v_gamma[[j]] + (1 - beta_2) * grad_gamma[[j]]^2
          # step_size_hat = step_size * sqrt(1 - beta_2^(i-1))/(1 - beta_1^(i-1))
          g_gamma[[j]] = -step_size_hat*m_gamma[[j]]/(sqrt(v_gamma[[j]]) + epsilon)
        }
        else {
          g_beta[[j]] = momentum*g_beta[[j]] - step_size * grad_beta[[j]]
          g_gamma[[j]] = momentum*g_gamma[[j]] - step_size * grad_gamma[[j]]
        }
        beta[[j]] = beta[[j]] + g_beta[[j]]
        gamma[[j]] = gamma[[j]] + g_gamma[[j]]
      }
    }

    if (keep_grad) {
      g_hist[[i]] = grad
      beta_g_hist[[i]] = grad_beta
      gamma_g_hist[[i]] = grad_gamma
    }

    if (dropout) {
      if (batch_norm) {
        fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type,
                                         dropout = T, retain_rate = retain_rate,
                                         batch_norm = T, gamma = gamma, beta = beta)
      }
      else {
        fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type,
                                         dropout = T, retain_rate = retain_rate)
      }
    }
    else {
      if (batch_norm) {
        fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type,
                              batch_norm = T, gamma = gamma, beta = beta)
      }
      else fb_train = forward_backward_pass(x_train[ind,], y_train[ind,], w, activation, output_type)
    }

    if (batch_norm) {
      batch_mean_new = lapply(fb_train$b, function(x) x$mean_x)
      batch_var_new = lapply(fb_train$b, function(x) x$var_x)
      batch_mean = mapply(function(a, b) 0.9*a + 0.1*b, batch_mean, batch_mean_new)
      batch_var = mapply(function(a, b) 0.9*a + 0.1*b, batch_var, batch_var_new)
    }

    if (batch_norm) penalty = lambda/2*sum(sapply(w, function(x) sum(x^2)))
    else penalty = lambda/2*sum(sapply(w, function(x) sum(x[-1,]^2)))
    cost_train[i] = cost_func(fb_train$p, y_train[ind,]) + penalty

    if (!is.null(x_test) & !is.null(y_test)) {
      if (dropout) {
        if (batch_norm) {
          w_adjust = lapply(w, function(x) {
            x = retain_rate*x
            x
          })
          fb_test = forward_backward_pass(x_test, y_test, w_adjust, activation, output_type, batch_norm = batch_norm,
                                          forward_only = T, gamma = gamma, beta = beta,
                                          mean_x = batch_mean, var_x = batch_var)
        }
        else {
          w_adjust = lapply(w, function(x) {
            x[-1,] = retain_rate*x[-1,]
            x
          })
          fb_test = forward_backward_pass(x_test, y_test, w_adjust, activation, output_type, forward_only = T)
        }
      }
      else {
        if (batch_norm) fb_test = forward_backward_pass(x_test, y_test, w, activation, output_type, batch_norm = batch_norm,
                                                        forward_only = T, gamma = gamma, beta = beta,
                                                        mean_x = batch_mean, var_x = batch_var)
        else fb_test = forward_backward_pass(x_test, y_test, w, activation, output_type, forward_only = T)
      }
      cost_test[i] = cost_func(fb_test$p, y_test)
    }

    # z_hist[[i]] = fb_test$z

    if (verbose) {
      if (!is.null(x_test) & !is.null(y_test)) message("iter = ", i, " training cost = ", round(cost_train[i], 6),
                                                       " test cost = ", round(cost_test[i], 6))
      else message("iter = ", i, " training cost = ", round(cost_train[i], 6))
    }

    # ind = sample(1:nrow(x_train), batch_size)
    ind = ind_mat[, (i-1) %% ncol(ind_mat) + 1]
  }

  fit = list(cost_train = cost_train, cost_test = cost_test, w = w, ini_w = ini_w,
             activation = activation, y_levels = y_levels, output_type = output_type,
             g_hist = g_hist, beta_g_hist = beta_g_hist, gamma_g_hist = gamma_g_hist,
             mean_x = mean_x, mean_y = mean_y, sd_x = sd_x, sd_y = sd_y,
             dropout = dropout, retain_rate = retain_rate,
             ini_gamma = ini_gamma, ini_beta = ini_beta,
             batch_mean = batch_mean, batch_var = batch_var)

  class(fit) = "netzuko"
  return(fit)

}
