import pandas as pd
from sklearn.metrics import confusion_matrix


class Node:

    def __init__(self):
        self.left = None
        self.right = None
        self.term = False
        self.label = None
        self.feature = None
        self.value = None

    def set_split(self, feature, value):
        # this function saves the node splitting feature and its value
        self.feature = feature
        self.value = value

    def set_term(self, label):
        # if the node is a leaf, this function saves its label
        self.term = True
        self.label = label


class DecisionTree:

    def __init__(self, min_samples=1, numerical=None):
        self.root = Node()
        self.min_samples = min_samples
        self.numerical = numerical

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._run_splitting(self.root, X, y)

    def predict(self, X: pd.DataFrame) -> list:
        predictions = []
        for i, row in X.iterrows():
            # print(f'Prediction for sample # {i}')
            prediction = self._run_prediction(self.root, row)
            predictions.append(prediction)
        return predictions

    def _weighted_gini_impurity(self, class_labels_1: list, class_labels_2: list) -> float:
        samples_num_1 = len(class_labels_1)
        samples_num_2 = len(class_labels_2)
        samples_num = samples_num_1 + samples_num_2

        gini_1 = self._gini_impurity(class_labels_1)
        gini_2 = self._gini_impurity(class_labels_2)

        return (samples_num_1 / samples_num) * gini_1 + (samples_num_2 / samples_num) * gini_2

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        min_weighted_gini_impurity = 1
        feature = None
        feature_value = None
        split_1, split_2 = None, None
        for col_name, values in X.iteritems():
            unique_values = values.unique()
            for value in unique_values:
                if self.numerical and col_name in self.numerical:
                    idx_1 = X.index[X[col_name] <= value].tolist()
                    idx_2 = X.index[X[col_name] > value].tolist()
                else:
                    idx_1 = X.index[X[col_name] == value].tolist()
                    idx_2 = X.index[X[col_name] != value].tolist()
                wgi = self._weighted_gini_impurity(y.iloc[idx_1].tolist(), y.iloc[idx_2].tolist())
                if wgi < min_weighted_gini_impurity:
                    min_weighted_gini_impurity = wgi
                    feature = col_name
                    feature_value = value
                    split_1, split_2 = idx_1, idx_2
        return min_weighted_gini_impurity, feature, feature_value, split_1, split_2

    def _is_node_leaf(self, X: pd.DataFrame, y: pd.Series) -> bool:
        if X.shape[0] <= self.min_samples:
            return True
        if self._gini_impurity(y.to_list()) == 0:
            return True
        if all(map(lambda x: len(set(x)) == 1, X.values.T)):
            return True
        return False

    def _run_splitting(self, node: Node, X: pd.DataFrame, y: pd.Series) -> None:
        if self._is_node_leaf(X, y):
            node.set_term(y.value_counts().idxmax())
            return

        mwgi, feature, feature_value, split_1, split_2 = self._find_best_split(X, y)
        node.set_split(feature, feature_value)
        # print(f'Made split: {feature} is {feature_value}')

        left_node, right_node = Node(), Node()
        node.left = left_node
        node.right = right_node

        left_X = X.iloc[split_1].reset_index(drop=True)
        right_X = X.iloc[split_2].reset_index(drop=True)
        left_y = y.iloc[split_1].reset_index(drop=True)
        right_y = y.iloc[split_2].reset_index(drop=True)

        self._run_splitting(left_node, left_X, left_y)
        self._run_splitting(right_node, right_X, right_y)

    def _run_prediction(self, node: Node, x: pd.Series) -> int:
        if node.term:
            # print(f'  Predicted label: {node.label}')
            return node.label
        # self._print_log_prediction_rule(node.feature, node.value)
        if self.numerical and node.feature in self.numerical:
            if x[node.feature] <= node.value:
                return self._run_prediction(node.left, x)
            else:
                return self._run_prediction(node.right, x)
        if x[node.feature] == node.value:
            return self._run_prediction(node.left, x)
        else:
            return self._run_prediction(node.right, x)

    @staticmethod
    def _gini_impurity(class_labels: list) -> float:
        samples_num = len(class_labels)
        labels = set(class_labels)
        gini = 1
        for class_label in labels:
            gini -= (class_labels.count(class_label) / samples_num) ** 2
        return gini

    @staticmethod
    def _print_log_prediction_rule(feature, value) -> None:
        print(f'  Considering decision rule on feature {feature} with value {value}')


if __name__ == "__main__":
    train_file_path, pred_file_path = input().split()
    df = pd.read_csv(train_file_path, index_col=0, encoding='latin-1')
    df_test = pd.read_csv(pred_file_path, index_col=0, encoding='latin-1')
    X_train = df.iloc[:, :-1]
    y_train = df['Survived']
    X_test = df_test.iloc[:, :-1]
    y_test = df_test['Survived']
    decision_tree = DecisionTree(min_samples=74, numerical=['Age', 'Fare'])
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    print(round(conf_mat[1, 1], 3), round(conf_mat[0, 0], 3))
