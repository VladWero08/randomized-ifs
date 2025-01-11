import typing as t
import numpy as np

from multiprocessing import Pool

class ExternalNode:
    def __init__(self, size: int, data):
        self.size = size 
        self.data = data


class InternalNode:
    pass


class InternalNode:
    def __init__(
        self, 
        left: ExternalNode | InternalNode, 
        right: ExternalNode | InternalNode,
        split_attribute: int,
        split_value: int | float
    ):
        self.left = left
        self.right = right
        self.split_attribute = split_attribute
        self.split_value = split_value


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
    def __init__(self, n_trees: int = 100, sub_sample_size: int = 256, n_processes: int = 8):
        self.n_trees = n_trees
        self.sub_sample_size = sub_sample_size
        self.n_processes = n_processes
        
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

    def fit_itree(self, _) -> t.Optional[ITree]:
        """
        Fits a single ITree, assuming the data set 
        was already defined in the IForest object.
        """
        if self.X is None:
            return
        
        indexes = np.random.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        itree = ITree(height=0, height_limit=self.height_limit)
        itree.fit(X_sub)

        return itree

    def fit(self, X: np.ndarray):
        """
        Fits an ensemble of isolation trees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.X = X

        # use a pool to compute the trees in parallel
        with Pool(processes=self.n_processes) as pool:
            # assign the scitree training to the pool
            itrees = pool.map(self.fit_itree, range(self.n_trees))
        
        self.itrees = itrees 

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
        Computes the average path length for a single sample
        among all trained isolation trees.
        """
        path_lengths = np.array([self.path_length(x, itree) for itree in self.itrees])
        return np.mean(path_lengths)    
    
    def score(self, x: np.ndarray) -> float:
        """
        Computes the score for a single sample.
        """
        return np.power(2, -1 * self.avg_path_length(x) / self.normalization)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the score for all dataset with multiple samples.
        """
        scores = np.array([self.score(x) for x in X])
        return scores
