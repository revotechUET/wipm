from ml_toolkit.Classifier import *
from ml_toolkit.DecisionTreeClassifier import DecisionTreeClassifier
from ml_toolkit.KNearestNeighborsClassifier import KNearestNeighborsClassifier
from ml_toolkit.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml_toolkit.NeuralNetworkClassifier import NeuralNetworkClassifier
from ml_toolkit.RandomForestClassifier import RandomForestClassifier
from ml_toolkit.S_SOMClassifier import S_SOMClassifier
from ml_toolkit.clf_helper import *
confusion_matrix = cm_with_percentage

CLASSIFIER = {
    'NeuralNetClassifier': NeuralNetworkClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'KNN': KNearestNeighborsClassifier,
    'LogisticRegression': LogisticRegressionClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'S_SOM': S_SOMClassifier,
    
    'neuralnetclassifier': NeuralNetworkClassifier,
    'decisiontreeclassifier': DecisionTreeClassifier,
    'knn': KNearestNeighborsClassifier,
    'logisticregression': LogisticRegressionClassifier,
    'randomforestclassifier': RandomForestClassifier,
    's_som': S_SOMClassifier,
}

class HIMFAClassifier(object):

    def __init__(self, train_set):
        self.train_set, self.val_set = split_data(train_set, 0.2, True, True)
        self.his = {'acc': None, 'loss': None, 'val_acc': None, 'val_loss': None}

        self.cm = None
        self.cm_groups = None

        self.labels_origin = np.unique(self.train_set[:,-1].astype('int')).tolist()
        self.num_labels = len(self.labels_origin)

    def fit(self, groups_model, facies_models, params=None, verbose=False):

        self.groups = []
        self.facies_models = []

        for model in facies_models:
            type = model['type']
            group = model['group']
            self.facies_models.append(CLASSIFIER[type](train_set=self.train_set, include=group))
            self.groups.append(group)

        self.groups_model = CLASSIFIER[groups_model](train_set=group_data(self.train_set, self.groups))
        self.num_groups = len(self.groups)
        self.labels_groups = list(range(self.num_groups))

        if not params:
            for model in [self.groups_model] + self.facies_models:
                model.fit()
        else:
            for model, param in zip([self.groups_model] + self.facies_models, params):
                model.fit(**param)

        pred = self.predict(X=self.train_set[:,:-1])
        accuracy = np.count_nonzero(self.train_set[:, -1] == pred) / len(pred)

        if verbose:
            print('\n--- Training result ---')
            print('Accuracy:', accuracy * 100)

        self.evaluate_test()

    def evaluate_test(self, radius=0, verbose=False):

        if verbose:
            print('\nEvaluating on test set...')
        X = self.val_set[:,:-1]
        y_facies = self.val_set[:,-1]
        y_groups = group_data(self.val_set[:,-1:], self.groups)[:,0]

        pred = self.groups_model.predict(X=X, radius=radius)

        pred = smoothen(pred, radius)

        pred_groups = np.array(pred)
        y_facies_models = []
        for model, group in zip(self.facies_models, range(len(self.facies_models))):
            data = X[pred_groups == group]
            if data.size:
                y_facies_models.append(model.predict(X=data))
            else:
                y_facies_models.append([])

        count = [0] * len(self.facies_models)
        for i in range(len(pred_groups)):
            if pred[i] is not None:
                group = pred[i]
                index = count[group]
                y_of_a_group = y_facies_models[group]
                pred[i] = y_of_a_group[index]
                count[group] += 1

        pred = smoothen(pred, radius)

        self.cm_groups = confusion_matrix(y_groups, pred_groups,
                                          labels=self.labels_groups)
        self.cm = confusion_matrix(y_facies, pred, labels=self.labels_origin)

    def predict(self, X, level=1, radius=0, threshold=0.0):

        pred = self.groups_model.predict(X=X, radius=radius, threshold=threshold)

        pred = smoothen(pred, radius)

        if level == 0:
            return pred

        pred_groups = np.array(pred)
        y_facies_models = []
        for model, group in zip(self.facies_models, range(len(self.facies_models))):
            data = X[pred_groups == group]
            if data.size:
                y_facies_models.append(model.predict(X=data))
            else:
                y_facies_models.append([])

        count = [0] * len(self.facies_models)
        for i in range(len(pred_groups)):
            if pred[i] is not None:
                group = pred[i]
                index = count[group]
                y_of_a_group = y_facies_models[group]
                pred[i] = y_of_a_group[index]
                count[group] += 1

        pred = smoothen(pred, radius)

        return pred

    def evaluate(self, X, y, level=1, radius=0, verbose=True):

        pred = self.predict(X, level, radius=radius, threshold=0.0)

        if level == 0:
            y = group_data(y, self.groups)

        accuracy = np.count_nonzero(y == pred) / len(y)

        if verbose:
            print('\n--- Training result ---')
            print('Accuracy:', accuracy * 100)

        pred = smoothen(pred, radius)
        accuracy = np.count_nonzero(y == pred) / len(y)
        if verbose and radius > 0:
            print('Accuracy after smoothening with radius =', radius, ': ', accuracy * 100)

        self.cm = confusion_matrix(y, pred, labels=self.labels_origin)

        return {'acc': accuracy, 'loss': None}

    def get_result(self, X, radius=0, threshold=0.0):
        pred = self.groups_model.predict(X=X, radius=radius, threshold=threshold)

        pred = smoothen(pred, radius)

        pred_groups = np.array(pred)
        y_facies_models = []
        for model, group in zip(self.facies_models, range(len(self.facies_models))):
            data = X[pred_groups == group]
            if data.size:
                y_facies_models.append(model.predict(X=data))
            else:
                y_facies_models.append([])

        count = [0] * len(self.facies_models)
        for i in range(len(pred_groups)):
            if pred[i] is not None:
                group = pred[i]
                index = count[group]
                y_of_a_group = y_facies_models[group]
                pred[i] = y_of_a_group[index]
                count[group] += 1

        pred = smoothen(pred, radius)

        return dict(target=pred.tolist(), target_groups=pred_groups.tolist())

    def get_cm_data_url(self, id):
        if self.cm is None or self.cm_groups is None:
            return None

        draw_confusion_matrix_himfa(self.cm, self.labels_origin + [''],
                                    self.cm_groups, self.labels_groups + [''])
        img = id + '.png'
        plt.savefig(img)
        data_url = image_to_data_url(img)
        os.remove(img)

        return data_url

    def save(self, model_dir=None):
        self.train_set = None

        if model_dir is None:
            model_dir = 'himfa_model'

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.groups_model.save(os.path.join(model_dir, 'groups_model'))
        self.groups_model = None

        for i in range(len(self.facies_models)):
            self.facies_models[i].save(os.path.join(model_dir, 'facies_model' + str(i)))
            self.facies_models[i] = None

        joblib.dump(self, os.path.join(model_dir, 'model'))

    @classmethod
    def load(HIMFAClassifier, model_dir):

        def __load(file_name):
            if os.path.isdir(file_name):
                return NeuralNetworkClassifier.load(file_name)
            return Classifier.load(file_name)

        classifier = __load(os.path.join(model_dir, 'model'))
        classifier.groups_model = __load(os.path.join(model_dir, 'groups_model'))
        for i in range(len(classifier.facies_models)):
            classifier.facies_models[i] = __load(os.path.join(model_dir, 'facies_model' + str(i)))

        return classifier
