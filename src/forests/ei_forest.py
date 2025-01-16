import typing as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import multiprocessing


class ExternalNode:
    def __init__(self, size: int, data):
        self.size = size 
        self.data = data


class InternalNode:
    def __init__(
        self, 
        left: "ExternalNode | InternalNode", 
        right: "ExternalNode | InternalNode",
        split_slope: np.ndarray,
        split_intercept: np.ndarray,
    ):
        self.left = left
        self.right = right
        self.split_slope = split_slope
        self.split_intercept = split_intercept


class EITree:
    def __init__(
        self, 
        height: int, 
        height_limit: int,
        extension_level: int = 0,    
    ):
        self.height = height
        self.height_limit = height_limit
        self.extension_level = extension_level
        self.root: InternalNode | ExternalNode = None

    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        """
        Fits a extended isolation tree for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.
        """
        if self.height >= self.height_limit or X.shape[0] <= 1:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # randomly choose the split slope
        # idxs = np.random.choice(range(X.shape[1]), X.shape[1] - self.extension_level, replace=False)
        split_slope = np.random.normal(loc=0.0, scale=1.0, size=X.shape[1])
        # split_slope[idxs] = 0
        # randomly choose the split intercept
        split_intercept = np.random.uniform(X.min(axis=0), X.max(axis=0))
        # compute the branching criteria
        split_criteria = np.dot(X - split_intercept, split_slope)

        # compute the split of the data
        X_left = X[split_criteria < 0] 
        X_right = X[split_criteria >= 0]

        # compute the left and right side of the tree
        node_left = EITree(self.height + 1, self.height_limit).fit(X_left)
        node_right = EITree(self.height + 1, self.height_limit).fit(X_right)

        self.root = InternalNode(node_left, node_right, split_slope, split_intercept)
        return self.root


class EIForest:
    def __init__(
        self, 
        n_trees: int = 100, 
        sub_sample_size: int = 256, 
        height_limit: t.Optional[int] = None,
        n_processes: int = 8
    ):
        self.n_trees = n_trees
        self.sub_sample_size = sub_sample_size
        self.n_processes: int = n_processes if n_processes else multiprocessing.cpu_count()
        self.height_limit: int = height_limit if height_limit else np.ceil(np.log2(self.sub_sample_size))
        
        self.expected_depth: float = self.c(sub_sample_size)
        self.ei_trees: t.List[EITree] = []

    def c(self, size: int) -> float:
        """
        Sets the expected depth of a iTree 
        based on the number of sub-samples.

        Parameters:
        -----------
        size: int
            The number of instances for each the unsuccessful 
            search for a BST is computed on. 
        """        
        if size > 2:
            H = np.log(size - 1) + 0.5772156649
            return 2 * H - 2 * (size - 1) / size

        if size == 2:
            return 1

        return 0.0

    def fit_ei_tree(self, _) -> t.Optional[EITree]:
        """
        Fits a single EITree, assuming the data set 
        was already defined in the EIForest object.
        """
        if self.X is None:
            return
        
        indexes = np.random.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        ei_tree = EITree(height=0, height_limit=self.height_limit)
        ei_tree.fit(X_sub)

        return ei_tree

    def fit(self, X: np.ndarray) -> None:
        """
        Fits an ensemble of isolation trees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.X = X

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            ei_trees = pool.map(self.fit_ei_tree, range(self.n_trees))
        
        self.ei_trees = ei_trees 

    def path_length(self, x: np.ndarray, ei_tree: EITree) -> float:
        """
        Computes the path length for a given sample in the given ITree.

        The path value starts from 0. After each movement
        from a node to another, the path is incremented.

        After reaching an external node (leaf), the path
        calculated until that point is returned plus an adjustment
        for the subtree that was not continued beyond the height limit.
        """
        node = ei_tree.root
        path = 0

        while not isinstance(node, ExternalNode):
            y = np.dot(x - node.split_intercept, node.split_slope)

            if y < 0:
                node = node.left
            else:
                node = node.right

            path += 1

        return path + self.c(node.size)

    def avg_path_length(self, x: np.ndarray) -> float:
        """
        Computes the average path length for a single sample
        among all trained isolation trees.
        """
        path_lengths = np.array([self.path_length(x, ei_tree) for ei_tree in self.ei_trees])
        return np.mean(path_lengths)    
    
    def score(self, x: np.ndarray) -> float:
        """
        Computes the score for a single sample.
        """
        return np.power(2, -1 * self.avg_path_length(x) / self.expected_depth)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the score for all dataset with multiple samples.
        """
        scores = np.array([self.score(x) for x in X])
        return scores

    def decision_area(self) -> t.Optional[matplotlib.figure.Figure]:
        """
        Computes the decision area for the fitted data,
        only if the data is 2-dimensional.
        """
        if self.X is None:
            return None
        
        if self.X.shape[1] != 2:
            return None

        xx, yy = np.meshgrid(
            np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 100),
            np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 100),
        )
        points = np.c_[xx.ravel(), yy.ravel()]
        scores = self.scores(points)
        scores = scores.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, scores, levels=15, cmap="coolwarm")
        ax.set_title("EIF")

        return fig
