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
        split_value: float,
        split_coeff: np.ndarray,
        split_attr: np.ndarray,
        split_attr_stds: np.ndarray,
        limit: float,
    ):
        self.left = left
        self.right = right
        self.split_value = split_value
        self.split_coeff = split_coeff 
        self.split_attr = split_attr
        self.split_attr_stds = split_attr_stds
        self.limit = limit 


class SCITree:
    def __init__(
        self, 
        n_attributes: int = 2, 
        n_hyperplanes: int = 5
    ) -> None:
        self.n_attributes = n_attributes
        self.n_hyperplanes = n_hyperplanes
        self.root: InternalNode = None

    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if X.shape[0] <= 2:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # randomly select coefficients and attributes for the hyperplanes
        coeffs = np.random.uniform(-1, 1, size=(self.n_hyperplanes, self.n_attributes))
        attrs = np.random.choice(range(0, X.shape[1]), size=(self.n_hyperplanes, self.n_attributes))

        # compute the projections onto the hyperplanes, and select the best one  
        Ys = np.array([self.hyperplane_projection(X, coeffs[i], attrs[i]) for i in range(self.n_hyperplanes)])        
        Y_best_idx, Y_best_split_value = self.hyperplane_select(Ys)
        Y = Ys[Y_best_idx]

        # compute the standard deviation for the attributes
        # that correspond to the best hyperplane
        attrs_stds = np.array([np.std(X[:, attrs[Y_best_idx][i]]) for i in range(self.n_attributes)])
        
        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]
        limit = max(Y) - min(Y)

        # compute the left and right side of the tree
        node_left = SCITree().fit(X_left)
        node_right = SCITree().fit(X_right)

        self.root = InternalNode(node_left, node_right, Y_best_split_value, coeffs[Y_best_idx], attrs[Y_best_idx], attrs_stds, limit)
        return self.root

    def hyperplane_projection(self, X: np.ndarray, coefficients: np.ndarray, attributes: np.ndarray) -> np.ndarray | float:
        """
        Projects the given dataset using the coefficients 
        and the attributes given.
        """
        projection = np.zeros(X.shape[0])
    
        for i in range(self.n_attributes):
            projection = projection + coefficients[i] * X[:, attributes[i]] / np.std(X[:, attributes[i]])

        return projection
    
    def hyperplane_projection_from_node(self, x: np.ndarray, node: InternalNode) -> float:
        """
        Projects a single sample using the coefficients and
        the attributes from the given node. Used mainly for inference.

        Parameters:
        -----------
        x: np.ndarray
            Single sample.
        node: InternalNode
            Current node where the sample is positioned in the SCITree.
        """
        projection = 0

        for i in range(self.n_attributes):
            projection = projection + node.split_coeff[i] * x[node.split_attr[i]] / node.split_attr_stds[i]

        return projection

    def hyperplane_select(self, Ys: np.ndarray) -> t.Tuple[int, float]:
        """
        Computes the SDGain for every hyperplane projection, searching
        for the best split value for each one. The scope is to search
        for the hyperplane and split value that lead to the best SDGain. 
        
        Parameters:
        -----------
        Ys: np.ndarray
            Projection of the data (X) onto the hyperplanes (coefficients and random attributes).

        Returns:
        --------
        (index, best_split_value): (int, float)
            The index corresponding to the hyperplane found and its best split value.
        """
        best_std_gain = 0
        best_split_value = None
        index = None

        for i, Y in enumerate(Ys):
            for split_value in Y:
                Y_left = Y[Y < split_value]
                Y_right = Y[Y >= split_value]
                std_gain = self.std_gain(Y, Y_left, Y_right)

                if std_gain > best_std_gain:
                    best_std_gain = std_gain
                    best_split_value = split_value
                    index = i

        return index, best_split_value

    def std_gain(self, Y: np.ndarray, Y_left: np.ndarray, Y_right: np.ndarray) -> float:
        """
        Computes the SDgain for the given projections and splits.
        """
        return (np.std(Y) - (np.std(Y_left) + np.std(Y_right)) / 2) / np.std(Y)


class SCIForest:
    def __init__(self, n_trees: int = 100, sub_sample_size: int = 256, n_processes: int = 8):
        self.n_trees = n_trees
        self.sub_sample_size = sub_sample_size
        self.n_processes = n_processes

        self.normalization: float = self.c(sub_sample_size)
        self.scitrees: t.List[SCITree] = []

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

    def fit_scitree(self, _) -> t.Optional[SCITree]:
        """
        Fits a single SCITree, assuming the data set 
        was already defined in the SCIForest object.
        """
        if self.X is None:
            return
        
        indexes = np.random.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        scitree = SCITree()
        scitree.fit(X_sub)

        return scitree

    def fit(self, X: np.ndarray):
        """
        Fits an ensemble of SCItrees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.X = X

        # use a pool to compute the trees in parallel
        with Pool(processes=self.n_processes) as pool:
            # assign the scitree training to the pool
            scitrees = pool.map(self.fit_scitree, range(self.n_trees))
        
        self.scitrees = scitrees 
    
    def path_length(self, x: np.ndarray, scitree: SCITree) -> float:
        """
        Computes the path length for a given sample
        in the SCisolation tree.
        """
        node = scitree.root
        path = 0

        while not isinstance(node, ExternalNode):
            # project the sample onto the hyperplane
            y = scitree.hyperplane_projection_from_node(x, node)

            if y < node.split_value:
                # check if the projection is inside 
                # the acceptable range (split - limit, split)
                path = path + (1 if y >= node.split_value - node.limit else 0)                
                node = node.left
            else:
                # check if the projection is inside 
                # the acceptable range (split, split + limit)
                path = path + (1 if y < node.split_value + node.limit else 0)
                node = node.right

        return path + self.c(node.size)

    def avg_path_length(self, x: np.ndarray) -> float:
        """
        Computes the average path length for a single sample
        among all trained SCisolation trees.
        """
        path_lengths = np.array([self.path_length(x, itree) for itree in self.scitrees])
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
