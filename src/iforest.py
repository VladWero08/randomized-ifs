import typing as t
import numpy as np
from . import InternalNode, ExternalNode


class ITree:
    def __init__(self, height: int, height_limit: int):
        self.height = height
        self.height_limit = height_limit
        self.root = None

    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        """
        Fits a isolation tree for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.
        """
        if self.height >= self.height_limit or X.shape[0] <= 1:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # randomly choose the split variable
        split_attribute = np.random.randint(0, X.shape[1])

        # randomly choose the split value
        split_value_min, split_value_max = np.min(X[:, split_attribute]), np.max(X[:, split_attribute]) 
        split_value = np.random.uniform(split_value_min, split_value_max)

        # compute the split of the data
        X_left = X[X[:, split_attribute] < split_value] 
        X_right = X[X[:, split_attribute] >= split_value]

        # compute the left and right side of the tree
        node_left = ITree(self.height + 1, self.height_limit).fit(X_left)
        node_right = ITree(self.height + 1, self.height_limit).fit(X_right)

        self.root = InternalNode(node_left, node_right, split_attribute, split_value)
        return self.root


class IForest:
    def __init__(self, tree_size: int = 100, sub_sample_size: int = 256):
        self.tree_size = tree_size
        self.sub_sample_size = sub_sample_size
        
        self.height_limit: int = np.ceil(np.log2(self.sub_sample_size))
        self.normalization: float = self.c(sub_sample_size)
        self.itrees: t.List[ITree] = []

    def c(self, size: int) -> float:
        """
        Sets the normalization constant based on the number of sub-samples.

        Parameters:
        -----------
        size: int
            The number of instances for each the unsuccessful 
            search for a BST is computed on. 
        """        
        if size >= 2:
            H = np.log(size) + 0.5772156649
            return 2 * H * (size - 1) - 2 * (size - 1) / size

        return 0.0

    def fit(self, X: np.ndarray):
        """
        Fits an ensemble of isolation trees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.itrees = []

        for _ in range(self.tree_size):
            indexes = np.random.randint(0, X.shape[0], size=self.sub_sample_size)
            X_sub = X[indexes]

            # build the isolation tree for the selected sub samples
            itree = ITree(height=0, height_limit=self.height_limit)
            itree.fit(X_sub)

            self.itrees.append(itree)

    def path_length(self, x: np.ndarray, itree: ITree) -> float:
        """
        Computes the path length for a given sample
        in the given isolation tree.
        """
        node = itree.root
        path = 0

        while not isinstance(node, ExternalNode):
            path += 1

            if x[node.split_attribute] < node.split_value:
                node = node.left
            else:
                node = node.right

        return path + self.c(node.size)

    def avg_path_length(self, x: np.ndarray) -> float:
        """
        Computes the average path length for a given sample
        among all trained isolation trees.
        """
        path_lengths = np.array([self.path_length(x, itree) for itree in self.itrees])
        return np.mean(path_lengths)    
    
    def score(self, x: np.ndarray) -> float:
        """
        Computes the score for the given sample.
        """
        return np.power(2, -1 * self.avg_path_length(x) / self.normalization)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the score for all samples in a dataset.
        """
        scores = np.array([self.score(x) for x in X])
        return scores
