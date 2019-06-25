import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.regularizers import l1_l2
from .Classifier import *
from statistics import mean

class NeuralNetworkClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=None,
                 val_size=0.2, feature_cols=None, label_col=-1, feature_degree=1,
                 include=None, preprocess=True):

        super().__init__(train_set, val_set, data_file, header, val_size,
                         feature_cols, label_col, feature_degree, True,
                         include, preprocess, True, True)

        self.structure()

    def structure(self, hidden_layer_sizes=[10,20,10], activation='elu'):

        self.model = Sequential()

        self.model.add(Dense(activation=activation,
                             input_dim=self.num_features,
                             units=hidden_layer_sizes[0],
                             kernel_initializer=glorot_uniform(),
                             kernel_regularizer=l1_l2(0.0),
                             use_bias=False))

        for layer in hidden_layer_sizes[1:]:
            self.model.add(Dense(activation=activation,
                                 units=layer,
                                 kernel_initializer=glorot_uniform(),
                                 kernel_regularizer=l1_l2(0.0),
                                 use_bias=False))

        self.model.add(Dense(activation='softmax',
                             units=self.num_labels,
                             kernel_initializer=glorot_uniform(),
                             kernel_regularizer=l1_l2(0.0),
                             use_bias=False))

    def fit(self, algorithm='backprop', batch_size=None, num_epochs=10000,
            optimizer='nadam', learning_rate=0.001, warm_up=False, decay=1e-6,
            population = 50, sigma = 0.01, boosting_ops = 0, verbose=False):

            if algorithm == 'backprop':
                self.train_backprop(batch_size, num_epochs, optimizer,
                                    learning_rate, warm_up, decay, verbose)
            elif algorithm == 'evolution':
                self.train_evolution(batch_size, num_epochs, population, sigma,
                                     learning_rate, boosting_ops, optimizer,
                                     decay, verbose)
            else:
                raise RuntimeError('Invalid algorithm')

            return self.his

    def train_backprop(self, batch_size, num_epochs, optimizer, learning_rate,
                       warm_up, decay, verbose):

        if verbose:
            verbose = 1
        else:
            verbose = 0

        if learning_rate is not None:
            if optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay)
            elif optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(lr=learning_rate, decay=decay)
            elif optimizer == 'adagrad':
                optimizer = keras.optimizers.Adagrad(lr=learning_rate, decay=decay)
            elif optimizer == 'adadelta':
                optimizer = keras.optimizers.Adadelta(lr=learning_rate, decay=decay)
            elif optimizer == 'adam':
                optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay)
            elif optimizer == 'adamax':
                optimizer = keras.optimizers.Adamax(lr=learning_rate, decay=decay)
            elif optimizer == 'nadam':
                optimizer = keras.optimizers.Nadam(lr=learning_rate)

        if batch_size is None:
            batch_size = self.num_samples

        if verbose:
            print('\nTraining Neural Network with back propagation...\n')

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=32, verbose=0, mode='auto')

        X, y = self.X_train, to_categorical(self.y_train, self.num_labels)
        X_val, y_val = self.X_val, to_categorical(self.y_val, self.num_labels)

        if (warm_up):
            if verbose:
                print('\nWarming up...\n')
            self.model.fit(X, y,
                          batch_size=1,
                          epochs=1,
                          validation_data=(X_val, y_val),
                          verbose=verbose)

        if verbose:
            print('\n Training with early stopping and batch size = %d...' % batch_size)
        self.his = self.model.fit(X, y,
                                  batch_size=batch_size,
                                  epochs=num_epochs,
                                  validation_data=(X_val, y_val),
                                  callbacks=[es],
                                  verbose=verbose).history

        train_accuracy = self.his['acc'][-1]
        train_loss = self.his['loss'][-1]
        if verbose:
            print('\n--- Training result ---')
            print('Accuracy: ', train_accuracy * 100, '  Loss: ', train_loss)

        self.evaluate_test(verbose=verbose)

    def train_evolution(self, batch_size, num_epochs, population, sigma,
                        learning_rate, boosting_ops, optimizer, decay, verbose):

        if batch_size is None:
            batch_size = self.num_samples

        if verbose:
            print('\nTraining Neural Network with evolution strategy...\n')

        self.his = {'acc':[], 'loss':[], 'val_acc':[], 'val_loss':[]}

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        num_w = 0
        w = []
        for layer in self.model.layers:
            cur_w = layer.get_weights()[0]
            shape = cur_w.shape
            num_w += shape[0] * shape[1]
            w = np.concatenate((w, np.ravel(cur_w)))

        X, y = self.X_train, to_categorical(self.y_train, self.num_labels)
        X_val, y_val = self.X_val, to_categorical(self.y_val, self.num_labels)

        r = 17
        for i in range(num_epochs):
            N = np.random.randn(population, num_w)
            R = np.zeros(population)
            for j in range(population):
                w_try = w + sigma * N[j]
                self.__set_weights(w_try)
                [loss, acc] = self.__result(X, y)
                R[j] = -loss
            A = (R - np.mean(R)) / np.std(R)
            old_w = w
            w += learning_rate / (population*sigma) * np.dot(N.T, A)
            learning_rate *= (1. / (1. + decay * i))

            self.__set_weights(w)
            [loss, acc] = self.__result(X, y)
            [val_loss, val_acc] = self.__result(X_val, y_val)

            self.his['loss'].append(loss)
            self.his['acc'].append(acc)
            self.his['val_loss'].append(val_loss)
            self.his['val_acc'].append(val_acc)
            if verbose:
                print('epoch %d/%d. loss: %f, accuracy: %f, val_loss: %f, val_accuracy: %f'
                      % (i+1, num_epochs, loss, acc*100, val_loss, val_acc*100))
            r = acc

            if val_loss > mean(self.his['val_loss'][-32:]):
                w = old_w
                self.__set_weights(w)
                break

        if boosting_ops > 0:
            if verbose:
                print('\nBoosting accuracy...\n')

            for i in range(boosting_ops):
                N = np.random.randn(population, num_w)
                new_w = w.copy()
                his_tmp = None
                for j in range(population):
                    w_try = w + sigma * N[j]
                    self.__set_weights(w)
                    [loss, acc] = self.__result(X, y)
                    if acc > r:
                        r = acc
                        new_w = w_try
                w = new_w
                self.__set_weights(w)
                [loss, acc] = self.__result(X, y)
                if verbose:
                    print('round %d/%d. loss: %f, accuracy: %f' % (i+1, 100, loss, acc*100))

        if verbose:
            print('\n--- Training result ---')
            print('Accuracy: ', acc * 100, '  Loss: ', loss)

        self.evaluate_test(verbose=verbose)

    def __fold(self, w):
        weights = []
        for layer in self.model.layers:
            shape = layer.get_weights()[0].shape
            elements = shape[0] * shape[1]
            weights.append(np.reshape(w[:elements], shape))
            w = w[elements:]
        return weights

    def __set_weights(self, w):
        weights = self.__fold(w)
        for w, layer in zip(weights, self.model.layers):
            layer.set_weights([w])

    def __result(self, X, y):
        return self.model.evaluate(X, y, batch_size=self.num_samples, verbose=0)

    def evaluate_helper(self, X, y, radius, verbose):

        y_matrix = to_categorical(y, self.num_labels)

        [loss, accuracy] = self.model.evaluate(X, y_matrix,
                                               batch_size=len(X),
                                               verbose=verbose)

        if verbose:
            print('Accuracy: ', accuracy * 100, '  Loss: ', loss)

        pred = self.model.predict(X).argmax(axis=1)

        pred = smoothen(pred, radius)
        accuracy = np.count_nonzero(y == pred) / len(y)
        if verbose and radius > 0:
            print('Accuracy after smoothening with radius =', radius, ': ', accuracy * 100)

        self.cm = confusion_matrix(y, pred, labels=self.labels)

        return accuracy, loss

    def evaluate_test(self, radius=0, verbose=False):
        if len(self.X_val) > 0:
            if verbose:
                print('\nEvaluating on test set...')

            self.score, _ = self.evaluate_helper(self.X_val, self.y_val,
                                                 radius, verbose)

    def judge(self, X=None, y=None, data=None, data_file=None, header=None,
              radius=0, verbose=False, threshold=0.8):

        if X is not None and y is not None:
            data = np.append(X, np.array(y).reshape(-1,1), axis=1)
        elif data is not None:
            pass
        elif data_file is not None:
            if verbose:
                print('\Judging on ', data_file, '...', sep='')
            data = pd.read_csv(data_file, header=header).values
        else:
            raise RuntimeError('Missing data')

        data = filt_data(data[:, self.cols], self.include)
        X = data[:, :-1]
        y = data[:, -1]
        if self.preprocess:
            X = add_features(X)

        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)
        y = self.le.transform(y)

        prob = self.model.predict(X, verbose=0)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = judge(pred, confidence, threshold=threshold)

        pred = smoothen(pred, radius)

        cm = confusion_matrix(y, pred, labels=self.labels+[-9999])

        return cm

    def probability(self, X=None, data_file=None, header=None):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        return self.model.predict(X, verbose=0)

    def predict(self, X=None, data_file=None, header=None, radius=0,
                threshold=0.0):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        prob = self.model.predict(X, verbose=0)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, confidence, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        return pred

    def get_result(self, X=None, data_file=None, header=None, radius=0,
                   threshold=0.0):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        prob = self.model.predict(X, verbose=0)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, confidence, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        cum_prob = cumulate_prob(prob)

        return dict(target=pred.tolist(), prob=cum_prob.tolist())

    def plot(self, name='loss and accuracy per epoch'):
        print(name)
        plt.plot(self.his['loss'], label='train loss')
        plt.plot(self.his['acc'], label='train accuracy')
        plt.plot(self.his['val_loss'], label='evaluate loss')
        plt.plot(self.his['val_acc'], label='evaluate accuracy')
        plt.legend()
        plt.title(name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def save(self, model_dir=None):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        if model_dir is None:
            model_dir = 'nn_model_' + str(int(round(self.score*10000,1)))

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # save weights
        self.model.save_weights(os.path.join(model_dir, 'weights'))

        # save save architecture
        self.model.save(os.path.join(model_dir, 'architecture'))

        self.model = None
        joblib.dump(self, os.path.join(model_dir, 'model'))

    @classmethod
    def load(NeuralNetworkClassifier, model_dir):
        classifier = joblib.load(os.path.join(model_dir, 'model'))
        classifier.model = load_model(os.path.join(model_dir, 'architecture'))
        classifier.model.load_weights(os.path.join(model_dir, 'weights'))
        classifier.model.compile(optimizer='adamax', loss='categorical_crossentropy',
                                 metrics=['accuracy'])

        return classifier
