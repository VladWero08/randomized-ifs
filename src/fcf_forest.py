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
        split_value: np.ndarray,
        split_coeff: np.ndarray,
        split_attr: np.ndarray,
        split_attr_means: np.ndarray,
        split_attr_stds: np.ndarray,
    ):
        self.left = left
        self.right = right
        self.split_value = split_value
        self.split_coeff = split_coeff
        self.split_attr = split_attr
        self.split_attr_means = split_attr_means
        self.split_attr_stds = split_attr_stds


class FCFTree:
    def __init__(
        self,
        height: int, 
        height_limit: int,
        n_attributes: int = 2,
        n_hyperplanes: int = 5,
    ) -> None:
        self.height = height
        self.height_limit = height_limit
        self.n_attributes = n_attributes
        self.n_hyperplanes = n_hyperplanes
    
    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if self.height >= self.height_limit or X.shape[0] == 1:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # randomly select coefficients and attributes for the hyperplanes
        coeffs = np.random.normal(loc=0, scale=1, size=(self.n_hyperplanes, self.n_attributes))
        attrs = np.array([
            np.random.choice(range(0, X.shape[1]), size=(self.n_attributes), replace=False) 
            for _ in range(self.n_hyperplanes)
        ])

        # compute the projections onto the hyperplanes, and select the best one
        Ys = np.array([self.hyperplane_projection(X, coeffs[i], attrs[i]) for i in range(self.n_hyperplanes)])        
        Y_best_idx, Y_best_split_value = self.hyperplane_select(Ys)
        Y = Ys[Y_best_idx]

        # compute the expectance and standard deviation 
        # for the attributes that correspond to the best hyperplane
        attrs_stds = np.array([np.std(X[:, attrs[Y_best_idx][i]]) for i in range(self.n_attributes)])
        attrs_means = np.array([np.mean(X[:, attrs[Y_best_idx][i]]) for i in range(self.n_attributes)])

        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]

        # compute the left and right side of the tree
        node_left = FCFTree(self.height + 1, self.height_limit).fit(X_left)
        node_right = FCFTree(self.height + 1, self.height_limit).fit(X_right)

        self.root = InternalNode(
            node_left, 
            node_right, 
            Y_best_split_value, 
            coeffs[Y_best_idx], 
            attrs[Y_best_idx], 
            attrs_means, 
            attrs_stds
        )
        return self.root

    def hyperplane_projection(
        self, 
        X: np.ndarray, 
        coefficients: np.ndarray, 
        attributes: np.ndarray
    ) -> np.ndarray | float:
        """
        Projects the given dataset using the coefficients 
        and the attributes given.
        """
        projection = np.zeros(X.shape[0])
    
        for i in range(self.n_attributes):
            projection = projection + coefficients[i] * (X[:, attributes[i]] - np.mean(X[:, attributes[i]])) / np.std(X[:, attributes[i]])

        return projection

    def hyperplane_projection_from_node(
        self, 
        x: np.ndarray, 
        node: InternalNode
    ) -> float:
        """
        Projects a single sample using the coefficients and
        the attributes from the given node. Used mainly for inference.

        - standardize the sample using the means and stds of the training data
        - multiply by the coefficient corresponding to the chosen attribute

        Parameters:
        -----------
        x: np.ndarray
            Single sample.
        node: InternalNode
            Current node where the sample is positioned in the SCITree.
        """
        projection = 0

        for i in range(self.n_attributes):
            projection = projection + node.split_coeff[i] * (x[node.split_attr[i]] - node.split_attr_means[i]) / node.split_attr_stds[i]

        return projection
    
    def hyperplane_select(self, Ys: np.ndarray) -> t.Tuple[int, float]:
        """
        Computes the pooled gain obtained from each hyperplane, searching
        for the best split value for each of them. The scope is to find
        the hyperplane and split value that lead to the best pooled gain. 
        
        Parameters:
        -----------
        Ys: np.ndarray
            Projection of the data onto the hyperplanes.

        Returns:
        --------
        (index, best_split_value): (int, float)
            The index corresponding to the hyperplane found and its best split value.
        """
        best_pool_gain = 0
        best_split_value = None
        index = None

        for i, Y in enumerate(Ys):
            for split_value in Y:
                Y_left = Y[Y < split_value]
                Y_right = Y[Y >= split_value]
                pool_gain = self.pool_gain(Y, Y_left, Y_right)

                if pool_gain > best_pool_gain:
                    best_pool_gain = pool_gain
                    best_split_value = split_value
                    index = i

        return index, best_split_value

    def pool_gain(self, Y: np.ndarray, Y_left: np.ndarray, Y_right: np.ndarray) -> float:
        """
        Computes the averaged gain for the given projections and splits.
        """
        n_samples_left = Y_left.shape[0]
        n_samples_right = Y_right.shape[0]

        return (np.std(Y) - (n_samples_left * np.std(Y_left) + n_samples_right * np.std(Y_right)) / (n_samples_left + n_samples_right)) / np.std(Y)
    

class FCFForest:
    def __init__(
        self,
        n_trees: int = 100,
        sub_sample_size: int = 256,
        height_limit: t.Optional[int] = None,
        n_processes: int = 8
    ):
        self.n_trees = n_trees
        self.sub_sample_size = sub_sample_size
        self.n_processes = n_processes

        self.height_limit: int = height_limit if height_limit else np.ceil(np.log2(self.sub_sample_size))
        self.expected_depth: float = self.c(sub_sample_size)
        self.fcf_trees: t.List[FCFTree] = []

    def c(self, size: int) -> float:
        """
        Sets the expected depth of a FCF Tree 
        based on the number of sub-samples.

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
    
    def fit_fcf_tree(self, _) -> t.Optional[FCFTree]:
        """
        Fits a single FCF tree, assuming the data set 
        was already defined in the FCF forest object.
        """
        if self.X is None:
            return
        
        indexes = np.random.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        fcf_tree = FCFTree(height=0, height_limit=self.height_limit)
        fcf_tree.fit(X_sub)

        return fcf_tree

    def fit(self, X: np.ndarray):
        """
        Fits an ensemble of FCF trees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.X = X

        # use a pool to compute the trees in parallel
        with Pool(processes=self.n_processes) as pool:
            # assign the scitree training to the pool
            fcf_trees = pool.map(self.fit_fcf_tree, range(self.n_trees))
        
        self.fcf_trees = fcf_trees 


    def path_length(self, x: np.ndarray, fcf_tree: FCFTree) -> float:
        """
        Computes the path length for a given sample in the given FCF tree.

        The path value starts from 0. After each movement
        from a node to another, the path is incremented.

        After reaching an external node (leaf), the path
        calculated until that point is returned plus an adjustment
        for the subtree that was not continued beyond the height limit.
        """
        node = fcf_tree.root
        path = 0

        while not isinstance(node, ExternalNode):
            y = fcf_tree.hyperplane_projection_from_node(x, node)

            if y < node.split_value:
                node = node.left
            else:
                node = node.right
            
            path += 1

        return path + self.c(node.size)

    def avg_path_length(self, x: np.ndarray) -> float:
        """
        Computes the average path length for a single sample
        among all trained FCF trees.
        """
        path_lengths = np.array([self.path_length(x, sci_tree) for sci_tree in self.fcf_trees])
        return np.mean(path_lengths)

    def score(self, x: np.ndarray) -> float:
        """
        Computes the score for a single sample.        
        """
        return np.power(2, -1 * self.avg_path_length(x) / self.expected_depth)

    def scores(self, X: np.ndarray) -> float:
        """
        Computes the score for a dataset with multiple samples.
        """
        return np.array([self.score(x) for x in X])