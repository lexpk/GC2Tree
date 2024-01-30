import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class GC2Tree:
    """Class representing the GC2 node classifier.
    """
    def __init__(
        self,
        depth=2,
        tree_depth=3,
        strategy=None
    ):
        """Initialize a GC2NodeClassifier object. Encode node features
        such that graph.nodes[vertex] yields a dictionary with the featurename
        as key and the feature value as value. The label should also be part
        of this dictionary.

        Args:
            data: A list of tuples containing a graph and a vertex.
            target: The key of the label.

        Returns:
            A GC2NodeClassifier object.
        """
        self.depth = depth
        self.tree_depth = tree_depth
        self.n_splits = 2 ** tree_depth - 1
        self.tree = DecisionTreeRegressor(
            max_depth=tree_depth
        )
        self.strategy = strategy

    def fit(self, adj, X, y, test_mask=None):
        self.init_data(adj, X, y, test_mask)

        if self.strategy is None:
            self.strategy = \
                np.flip(np.diag(np.ones(self.depth+1)), axis=1) + \
                np.flip(np.diag(np.ones(self.depth), k=1), axis=1)

        for i in range(self.depth+1):
            self.iterate(self.strategy[i], final=i == self.depth)

    def init_data(self, adj, X, y, test_mask=None):
        self.adj = adj
        self.directed = (adj != adj.T).max() != 0
        if self.directed:
            self.out_degree = np.array(adj.sum(axis=1)).reshape(-1, 1)
            self.in_degree = np.array(adj.sum(axis=0)).reshape(-1, 1)
        else:
            self.degree = np.array(adj.sum(axis=1)).reshape(-1, 1)

        self.n_nodes = adj.shape[0]
        self.n_features = X.shape[1]
        self.labels = np.unique(y)
        self.one_hot = np.array([y == label for label in self.labels]).T
        self.n_labels = self.one_hot.shape[1]
        self.n_computed_labels = self.n_labels * (self.depth + 1)

        if test_mask is None:
            test_mask = np.zeros(self.n_nodes, dtype=bool)
        self.train_samples = np.where(~test_mask)[0]
        self.test_samples = np.where(test_mask)[0]
        self.scaler = StandardScaler().fit(self.one_hot[self.train_samples, :])
        self.scaled_labels = self.scaler.transform(self.one_hot)
        self.train_labels = np.array(self.scaled_labels)
        self.train_labels[self.test_samples, :] = 0

        if self.directed:
            self.n_accumulated_features = 2 * self.n_splits * \
                self.n_computed_labels
            self.X = np.zeros((
                self.n_nodes,
                # features, number of in-neighbors with feautre,
                # number of out-neighbors with feature
                3 * self.n_features +
                # features that are split on, not initialized
                self.n_splits +
                # accumulated features at leaves, not initialized
                self.n_accumulated_features +
                # number of in- and out-neighbors with each accumulated
                # feature, not initialized
                2 * self.n_accumulated_features +
                # in-degree, out-degree
                2
            ))
            self.X[:, :self.n_features] = X
            self.X[:, self.n_features:2 * self.X] = \
                self.adj.dot(X)
            self.X[:, 2 * self.n_features:] = \
                self.adj.T.dot(X)
            self.X[:, -2:-1] = self.out_degree
            self.X[:, -1:] = self.in_degree

            self.y = np.zeros([
                self.n_nodes,
                (2**(self.depth+1) - 1) * self.n_labels
            ])

            def init_labels_directed(index, iterations, data):
                if iterations > self.depth:
                    return
                self.y[:, index*self.n_labels:(index+1)*self.n_labels] = data
                init_labels_directed(2*index+1, iterations+1,
                                     np.divide(
                                         self.adj.T.dot(data),
                                         self.in_degree,
                                         out=np.zeros_like(
                                            data,
                                            dtype=float),
                                         where=self.in_degree != 0
                                     ))
                init_labels_directed(2*index+2, iterations+1,
                                     np.divide(
                                         self.adj.dot(data),
                                         self.out_degree,
                                         out=np.zeros_like(self.adj.dot(data)),
                                         where=self.out_degree != 0
                                     ))

            init_labels_directed(0, 0, self.train_labels)
        else:
            self.n_accumulated_features = 2 * self.n_splits * \
                self.n_computed_labels
            self.X = np.zeros((
                self.n_nodes,
                # features, number of neighbors with feature
                2 * self.n_features +
                # features that are split on, not initialized
                self.n_splits +
                # accumulated features at leaves, not initialized
                self.n_accumulated_features +
                # number of neighbors with each accumulated feature,
                # not initialized
                self.n_accumulated_features +
                # degree
                1
            ))
            self.X[:, :self.n_features] = X
            self.X[:, self.n_features:2*self.n_features] = \
                self.adj.dot(X)
            self.X[:, -1:] = self.degree

            self.y = np.zeros([
                self.n_nodes,
                (self.depth+1) * self.n_labels
            ])

            def init_labels_undirected(index, iteration, data):
                scaled_data = StandardScaler().fit_transform(data)
                self.y[:, index*self.n_labels:(index+1)*self.n_labels] = \
                    scaled_data
                if iteration < self.depth:
                    init_labels_undirected(
                        index+1,
                        iteration+1,
                        np.divide(
                            self.adj.dot(scaled_data > 0),
                            self.degree,
                            out=np.zeros_like(self.adj.dot(scaled_data)),
                            where=self.degree != 0
                        )
                    )

            init_labels_undirected(0, 0, self.train_labels)

    def iterate(self, depth_weights=None, final=False):
        """Iterate the GC2 node classifier.

        Args:
            depth_weights: A list of weights for each depth.

        Returns:
            A GC2NodeClassifier object.
        """
        if depth_weights is None:
            depth_weights = np.ones(self.depth + 1)
        assert len(depth_weights) == self.depth + 1, \
            "weights for strategy must be of length depth + 1"
        weights = (depth_weights *
                   np.ones((self.n_labels, self.depth + 1))).T.flatten()
        self.tree.fit(self.X[self.train_samples, :],
                      self.y[self.train_samples, :] * weights)

        if final:
            return

        splits = [f for f in self.tree.tree_.feature if f != -2]
        leaves = self.tree.apply(self.X)

        if self.directed:
            # Don't overwrite features and number of neighbors with feature
            offset = 3*self.n_features
            # First set of computed features are features used in the tree
            self.X[:, offset:offset+len(splits)] = \
                self.X[:, splits]
            offset += self.n_splits
            # Now compute for each feature the accumulations at the leaves
            for i in range(self.n_computed_labels):
                sorted_leaves = sorted(
                    set(leaves),
                    key=lambda leaf: self.tree.tree_.value[leaf][i]
                )
                self.X[:, offset] = leaves == sorted_leaves[0]
                for j, leaf in enumerate(sorted_leaves[1:-1]):
                    self.X[:, offset + j] = \
                        self.X[:, offset + j - 1] + (leaves == leaf)
                offset += len(sorted_leaves) - 1
                self.X[:, offset:offset+len(sorted_leaves)-1] = \
                    1 - self.X[:, offset-len(sorted_leaves)+1:offset]
                offset += self.n_splits
            # Now compute for each accumulated feature the number of
            # in- and out-neighbors with that feature
            self.X[:, offset:offset+self.n_accumulated_features] = \
                self.adj.dot(
                    self.X[:, offset-self.n_accumulated_features:offset])
            offset += self.n_accumulated_features
            self.X[:, offset:offset+self.n_accumulated_features] = \
                self.adj.T.dot(
                    self.X[:, offset-2*self.n_accumulated_features:
                           offset-self.n_accumulated_features])
            offset += self.n_accumulated_features
        else:
            # Don't overwrite features and number of neighbors with feature
            offset = 2*self.n_features
            # First set of computed features are features used in the tree
            self.X[:, offset:offset+len(splits)] = \
                self.X[:, splits]
            offset += self.n_splits
            # Now compute for each feature the accumulations at the leaves
            for i in range(self.n_computed_labels):
                sorted_leaves = sorted(
                    set(leaves),
                    key=lambda leaf: self.tree.tree_.value[leaf][i]
                )
                self.X[:, offset] = leaves == sorted_leaves[0]
                for j, leaf in enumerate(sorted_leaves[1:-1]):
                    self.X[:, offset + j + 1] = \
                        self.X[:, offset + j] + (leaves == leaf)
                offset += self.n_splits
                self.X[:, offset:offset+len(sorted_leaves)-1] = \
                    1 - self.X[:, offset-len(sorted_leaves)+1:offset]
                offset += self.n_splits
            # Now compute for each accumulated feature the number of
            # neighbors with that feature
            self.X[:, offset:offset+self.n_accumulated_features] = \
                self.adj.dot(
                    self.X[:, offset-self.n_accumulated_features:offset])
            offset += self.n_accumulated_features

    def predict(self):
        """Predict the labels of the test set.

        Args:
            test_mask: A boolean array indicating which nodes are in the test
                set.

        Returns:
            The predicted labels.
        """
        return self.labels[
            np.argmax(self.tree.predict(self.X)[:, :self.n_labels], axis=1)
        ]

    def score(self, y_true, test_mask=None):
        """Compute the accuracy of the predictions.

        Args:
            y_true: The true labels.

        Returns:
            The accuracy of the predictions.
        """
        assert test_mask is not None, \
            "test_mask must be specified to evaluate the model"
        return np.mean((self.predict() == y_true)[test_mask])

    def training_accuracy(self, y_true, test_mask=None):
        """Compute the accuracy of the predictions.

        Args:
            y_true: The true labels.

        Returns:
            The accuracy of the predictions.
        """
        if test_mask is None:
            test_mask = np.zeros(self.n_nodes, dtype=bool)
        return np.mean((self.predict() == y_true)[~test_mask])
