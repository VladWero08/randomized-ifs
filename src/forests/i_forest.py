import typing as t
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.figure

class ExternalNode:
    def __init__(self, size: int):
        self.size = size 


class InternalNode:
    def __init__(
        self, 
        left: "ExternalNode | InternalNode", 
        right: "ExternalNode | InternalNode",
        split_attribute: int,
        split_value: int | float
    ):
        self.left = left
        self.right = right
        self.split_attribute = split_attribute
        self.split_value = split_value


class ITree:
    def __init__(self, height: int, height_limit: int):
        self.height: int = height
        self.height_limit: int = height_limit
        self.root: InternalNode | ExternalNode = None

    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        """
        Fits a isolation tree for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.
        """
        if self.height >= self.height_limit or X.shape[0] <= 1:
            self.root = ExternalNode(size=X.shape[0])
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
    def __init__(
        self, 
        n_trees: int = 100, 
        sub_sample_size: int = 256, 
        contamination: float = 0.1,
        height_limit: t.Optional[int] = None,
        n_processes: int = 8
    ):
        # initialize parameters passed from the constructor
        self.n_trees: int = n_trees
        self.sub_sample_size: int = sub_sample_size
        self.contamination: float = contamination
        self.height_limit: int = height_limit if height_limit else np.ceil(np.log2(self.sub_sample_size))
        self.n_processes: int = n_processes if n_processes else multiprocessing.cpu_count()        

        # initialize parameters used for fitting
        self.expected_depth: float = self.c(sub_sample_size)
        self.itrees: t.List[ITree] = []
        self.decision_scores: t.List[float] = [] 
        self.threshold: t.Optional[float] = None
        self.labels: t.List[int] = []

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
            itrees = pool.map(self.fit_itree, range(self.n_trees))
        
        self.itrees = itrees 
        
        # compute the scores for the training data
        self.decision_scores = self.scores(X)
        # compute the threshold and labels for the training data
        self.threshold = np.quantile(self.decision_scores, 1 - self.contamination)
        self.labels = (self.decision_scores > self.threshold).astype(int)

    def predict(self, X: np.ndarray) -> None:
        """
        Predicts the outlier labels for the given data, based
        on the threshold defined when the forest was trained.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be predicted.  
        """
        # compute the scores for the data
        X_scores = self.scores(X)

        # compute the outlier labels
        X_labels = (X_scores > self.threshold).astype(int)

        return X_labels


    def path_length(self, x: np.ndarray, itree: ITree) -> float:
        """
        Computes the path length for a given sample in the given ITree.

        The path value starts from 0. After each movement
        from a node to another, the path is incremented.

        After reaching an external node (leaf), the path
        calculated until that point is returned plus an adjustment
        for the subtree that was not continued beyond the height limit.
        """
        node = itree.root
        path = 0

        while not isinstance(node, ExternalNode):
            y = x[node.split_attribute]

            if y < node.split_value:
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
        path_lengths = np.array([self.path_length(x, itree) for itree in self.itrees])
        return np.mean(path_lengths)    
    
    def score(self, x: np.ndarray) -> float:
        """
        Computes the score for a single sample.
        """
        return np.power(2, -self.avg_path_length(x) / self.expected_depth)

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
        ax.set_title("IF")

        return fig
