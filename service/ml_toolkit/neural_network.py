import warnings
import numpy as np
import scipy
from scipy import optimize
from sklearn.base import RegressorMixin
from sklearn.neural_network.multilayer_perceptron import MLPRegressor as BaseMLP
from sklearn.neural_network.multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.neural_network.multilayer_perceptron import _STOCHASTIC_SOLVERS
from sklearn.neural_network.multilayer_perceptron import _pack
from sklearn.utils  import check_random_state
from sklearn.utils import check_X_y, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import gen_batches
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from sklearn.base import is_classifier
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import DERIVATIVES, LOSS_FUNCTIONS, ACTIVATIONS
from sklearn.model_selection import train_test_split

try:
    from scipy.optimize import anneal
except Exception as e:
    print('Version <= 0.14.0 for using Simulated Anealing')

class MLPRegressor(BaseMLP):
    def __init__(self, bound=(-10,10), popsize=5, strategy='best1bin', recombination=None,
                 T=0.01, stepsize=0.01, *args, **kwargs):
        super(MLPRegressor, self).__init__(*args, **kwargs)
        self.lpath = dict(train=[], val=[])
        self.n_iter_no_change = kwargs.get('n_iter_no_change', 1)
        self._iter = 0

        # parameter for differents evolution
        self.bound = bound
        self.popsize = popsize
        self.straegy = strategy
        self.recombination = recombination

        # parameter for anneal
        self.T = T
        self.stepsize = stepsize

    def _calculator_activations(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        coefs_ = []
        intercepts_ = []
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            coefs_.append(np.reshape(packed_parameters[start:end], shape))

            start, end = self._intercept_indptr[i]
            intercepts_.append(packed_parameters[start:end])

        """Perform a forward pass on the network by computing the values
            of the neurons in the hidden layers and the output layer.

            Parameters
            ----------
            activations : list, length = n_layers - 1
                The ith element of the list holds the values of the ith layer.

            with_output_activation : bool, default True
                If True, the output passes through the output activation
                function, which is either the softmax function or the
                logistic function
        """
        # calculator predict validation set
        activations = list(self.val_activations)
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                    coefs_[i])
            activations[i + 1] += intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations[i + 1] = output_activation(activations[i + 1])
        p_val = list(activations[-1])


        # calculator predict trainset
        activations = list(self.train_activations)
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                    coefs_[i])
            activations[i + 1] += intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations[i + 1] = output_activation(activations[i + 1])
        p_train = list(activations[-1])

        return p_train, p_val

    def _callback(self, packed_parameters, *args, **kwargs):
        p_train, p_val = self._calculator_activations(packed_parameters)
        loss_train = LOSS_FUNCTIONS[self.loss](self.y_train, p_train)
        loss_val = LOSS_FUNCTIONS[self.loss](self.y_val, p_val)
        if self.verbose:
            print('Iter %3d loss train %.5f loss val %.5f' % \
                (self._iter, loss_train, loss_val))
        self._iter += 1
        self.lpath['train'].append(loss_train)
        self.lpath['val'].append(loss_val)

    def _fit_bfgs(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run BFGS
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        optimal_parameters, self.loss_, d, Bopt, func_calls, grad_calls, warnflag = \
            optimize.fmin_bfgs(x0=packed_coef_inter,
                               f=self._loss_func,
                               fprime=self._grad_func,
                               maxiter=self.max_iter,
                               disp=False,
                               gtol=self.tol,
                               args=(X, y, activations, deltas, coef_grads, intercept_grads),
                               full_output=True,
                               callback=self._callback)

        self._unpack(optimal_parameters)

    def _fit_lbfgs(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        optimal_parameters, self.loss_, d = optimize.fmin_l_bfgs_b(
            x0=packed_coef_inter,
            func=self._loss_grad_lbfgs,
            maxfun=self.max_iter,
            iprint=-1,
            pgtol=self.tol,
            args=(X, y, activations, deltas, coef_grads, intercept_grads),
            callback=self._callback)

        self._unpack(optimal_parameters)

    def _fit_evol(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run evolution
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        bounds = list()
        for i in range(len(packed_coef_inter)):
            bounds.append(self.bound)

        result = optimize.differential_evolution(
                    func=self._loss_func,
                    bounds=bounds,
                    maxiter=self.max_iter,
                    disp=False,
                    polish=True,
                    init='latinhypercube',
                    popsize=self.popsize,
                    strategy=self.straegy,
                    seed=self.random_state,
                    args=(X, y, activations, deltas, coef_grads, intercept_grads),
                    callback=self._callback)

        optimal_parameters = result.x
        self.loss_ = result.fun

        self._unpack(optimal_parameters)

    def _fit_cg(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                 intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run CG
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        optimal_parameters, self.loss_, func_calls, grad_calls, d = \
            optimize.fmin_cg(x0=packed_coef_inter,
                             f=self._loss_func,
                             fprime=self._grad_func,
                             maxiter=self.max_iter,
                             disp=False,
                             epsilon=self.epsilon,
                             gtol=self.tol,
                             args=(X, y, activations, deltas, coef_grads, intercept_grads),
                             callback=self._callback,
                             full_output=True)

        self._unpack(optimal_parameters)

    def _fit_ncg(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                 intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run Newton-CG
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        optimal_parameters, self.loss_, func_calls, grad_calls, h_calls, d = \
            optimize.fmin_ncg(x0=packed_coef_inter,
                              f=self._loss_func,
                              fprime=self._grad_func,
                              maxiter=200,
                            #   maxiter=self.max_iter,
                              disp=True,
                              args=(X, y, activations, deltas, coef_grads, intercept_grads),
                              callback=self._callback,
                              full_output=True)

        self._unpack(optimal_parameters)

    def _fit_basinhopping(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                 intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run Basinhopping
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'args': (X, y, activations, deltas, coef_grads, intercept_grads)
        }

        result = optimize.basinhopping(x0=packed_coef_inter,
                                       T = self.T,
                                       stepsize=self.stepsize,
                                       func=self._loss_func,
                                       niter=self.max_iter,
                                       callback=self._callback,
                                       minimizer_kwargs=minimizer_kwargs)

        optimal_parameters = result.x
        self.loss = result.fun

        self._unpack(optimal_parameters)

    def _fit_anneal(self, X, y, X_val, Y_val, activations, deltas, coef_grads,
                 intercept_grads, layer_units):

        if scipy.__version__ == '0.14.0':
            # Store meta information for the parameters
            self._coef_indptr = []
            self._intercept_indptr = []
            start = 0

            # Save sizes and indices of coefficients for faster unpacking
            for i in range(self.n_layers_ - 1):
                n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

                end = start + (n_fan_in * n_fan_out)
                self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
                start = end

            # Save sizes and indices of intercepts for faster unpacking
            for i in range(self.n_layers_ - 1):
                end = start + layer_units[i + 1]
                self._intercept_indptr.append((start, end))
                start = end

            # Run Simulated Annealing
            packed_coef_inter = _pack(self.coefs_,
                                    self.intercepts_)

            result = optimize.anneal(x0=packed_coef_inter,
                                     T0 = self.T,
                                     stepsize=self.stepsize,
                                     func=self._loss_func,
                                     maxiter=self.max_iter,
                                     args=(X, y, activations, deltas, coef_grads, intercept_grads))

            optimal_parameters = result.x
            self.loss = result.fun

            self._unpack(optimal_parameters)
        else:
            raise ImportError('Anneal method require scipy version <= 0.14.0')

    def _grad_func(self, packed_coef_inter, X, y, activations, deltas,
                      coef_grads, intercept_grads):
        loss, coef_grads, intercept_grads = \
            self._backprop(X, y, activations, deltas,
                           coef_grads, intercept_grads)
        grad = _pack(coef_grads, intercept_grads)
        return grad

    def _loss_func(self, packed_coef_inter, X, y, activations, deltas,
                      coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.
        Returned loss are packed in a single vector so it can be used
        in cg
        Parameters
        ----------
        packed_coef_inter : array-like
            A vector comprising the flattened coefficients and intercepts.
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        Returns
        -------
        loss : float
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads)
        self.n_iter_ += 1

        return loss

    def _fit_stochastic(self, X, y, X_val, y_val, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):

        if not incremental or not hasattr(self, '_optimizer'):
            params = self.coefs_ + self.intercepts_

            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1, self.beta_2,
                    self.epsilon)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental

        n_samples = X.shape[0]

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                X, y = shuffle(X, y, random_state=self._random_state)
                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    activations[0] = X[batch_slice]
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X[batch_slice], y[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(grads)

                # Get val path
                activations[0] = X_val
                activations = self._forward_pass(activations)
                loss_val = LOSS_FUNCTIONS[self.loss](y_val, activations[-1])
                self.lpath['val'].append(loss_val)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                self.lpath['train'].append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = ("Validation score did not improve more than "
                               "tol=%f for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))
                    else:
                        msg = ("Training loss did not improve more than tol=%f"
                               " for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))

                    is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        flag: True if calculator train error
        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # Add L2 regularization term to loss
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads

    def _fit(self, X, y, incremental=False):
        X, X_val, y, Y_val = train_test_split(X, y, test_size=0.1)
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)

        X, y = self._validate_input(X, y, incremental)
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if not hasattr(self, 'coefs_') or (not self.warm_start and not
                                           incremental):
            # First time training the model
            self._initialize(y, layer_units)

        # lbfgs does not support mini-batches
        if self.solver == 'lbfgs' or self.solver == 'cg':
            batch_size = n_samples
        elif self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        self.val_activations = list(activations)
        self.train_activations = list(activations)
        self.val_activations[0] = [X_val]
        self.train_activations[0] = [X]
        self.y_val = Y_val
        self.y_train = y

        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                      n_fan_out_ in zip(layer_units[:-1],
                                        layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, X_val, Y_val, activations, deltas, coef_grads,
                                 intercept_grads, layer_units, incremental)

        # Run the LBFGS solver
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, X_val, Y_val, activations, deltas, coef_grads,
                            intercept_grads, layer_units)

        # Run the BFGS solver
        elif self.solver == 'bfgs':
            self._fit_bfgs(X, y, X_val, Y_val, activations, deltas, coef_grads,
                            intercept_grads, layer_units)

        # Run the Conjugate Gradient
        elif self.solver == 'cg':
            self._fit_cg(X, y, X_val, Y_val, activations, deltas, coef_grads,
                          intercept_grads, layer_units)

        # Run the Newton-CG
        elif self.solver == 'ncg':
            self._fit_ncg(X, y, X_val, Y_val, activations, deltas, coef_grads,
                          intercept_grads, layer_units)

        # Run the Evolution
        elif self.solver == 'evol':
            self._fit_evol(X, y, X_val, Y_val, activations, deltas, coef_grads,
                          intercept_grads, layer_units)

        # Run the Basinhopping
        elif self.solver == 'basinhopping':
            self._fit_basinhopping(X, y, X_val, Y_val, activations, deltas, coef_grads,
                          intercept_grads, layer_units)

        # Run the Simulated Annealing
        elif self.solver == 'anneal':
            self._fit_anneal(X, y, X_val, Y_val, activations, deltas, coef_grads,
                          intercept_grads, layer_units)

        # Delete dataset is member class
        del self.train_activations
        del self.val_activations
        del self.y_train
        del self.y_val

        return self

    def _validate_hyperparameters(self):
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False, got %s." %
                             self.shuffle)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.alpha)
        if (self.learning_rate in ["constant", "invscaling", "adaptive"] and
                self.learning_rate_init <= 0.0):
            raise ValueError("learning_rate_init must be > 0, got %s." %
                             self.learning_rate)
        if self.momentum > 1 or self.momentum < 0:
            raise ValueError("momentum must be >= 0 and <= 1, got %s" %
                             self.momentum)
        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError("nesterovs_momentum must be either True or False,"
                             " got %s." % self.nesterovs_momentum)
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False,"
                             " got %s." % self.early_stopping)
        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)
        if self.beta_1 < 0 or self.beta_1 >= 1:
            raise ValueError("beta_1 must be >= 0 and < 1, got %s" %
                             self.beta_1)
        if self.beta_2 < 0 or self.beta_2 >= 1:
            raise ValueError("beta_2 must be >= 0 and < 1, got %s" %
                             self.beta_2)
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0, got %s." % self.epsilon)

        # raise ValueError if not registered
        supported_activations = ('identity', 'logistic', 'tanh', 'relu')
        if self.activation not in supported_activations:
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s." % (self.activation,
                                                      supported_activations))
        if self.learning_rate not in ["constant", "invscaling", "adaptive"]:
            raise ValueError("learning rate %s is not supported. " %
                             self.learning_rate)
        supported_solvers = _STOCHASTIC_SOLVERS + ["lbfgs", "bfgs", "cg", "ncg", "evol", "anneal", "basinhopping"]
        if self.solver not in supported_solvers:
            raise ValueError("The solver %s is not supported. "
                             " Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))
        supported_strategy = ['best1bin', 'best1exp','rand1exp','randtobest1exp',
                              'best2exp','rand2exp','randtobest1bin','best2bin',
                              'rand2bin','rand1bin']
        if self.straegy not in supported_strategy:
            raise ValueError("The strategy %s is not supported. "
                             " Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from matplotlib import pyplot as  plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    boston = load_boston()
    x_train, x_test, y_train, y_test = \
        train_test_split(boston.data, boston.target, test_size=0.2)
    steps = [
            #  ('stand', StandardScaler()),
             ('normal', MinMaxScaler()),
             ('estimator', MLPRegressor(hidden_layer_sizes=(100),
                                        solver='ncg',
                                        activation='relu',
                                        verbose=True,
                                        max_iter=200))]
    pipe = Pipeline(steps)
    pipe.fit(x_train, y_train)
    train = pipe.named_steps['estimator'].lpath['train']
    val = pipe.named_steps['estimator'].lpath['val']
    plt.plot(list(range(len(val))), val, label='Val')
    plt.plot(list(range(len(train))), train, label='Train')
    plt.legend()
    plt.show()


