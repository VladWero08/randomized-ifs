import typing as t
import numpy as np

from multiprocessing import Pool


class ExternalNode:
    def __init__(self, size: int, data):
        self.size = size 
        self.data = data


class InternalNode:
    def __init__(
        self, 
        left: "ExternalNode | InternalNode", 
        right: "ExternalNode | InternalNode",
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
        height: int,
        height_limit: int,
        n_attributes: int = 2, 
        n_hyperplanes: int = 5
    ) -> None:
        self.height = height
        self.height_limit = height_limit
        self.n_attributes = n_attributes
        self.n_hyperplanes = n_hyperplanes
        self.root: InternalNode | ExternalNode = None

    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # randomly select coefficients and attributes for the hyperplanes
        coeffs = np.random.uniform(-1, 1, size=(self.n_hyperplanes, self.n_attributes))
        attrs = np.array([
            np.random.choice(range(0, X.shape[1]), size=(self.n_attributes), replace=False) 
            for _ in range(self.n_hyperplanes)
        ])

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
        node_left = SCITree(self.height + 1, self.height_limit).fit(X_left)
        node_right = SCITree(self.height + 1, self.height_limit).fit(X_right)

        self.root = InternalNode(
            node_left, 
            node_right, 
            Y_best_split_value, 
            coeffs[Y_best_idx], 
            attrs[Y_best_idx], 
            attrs_stds, 
            limit
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
            projection = projection + coefficients[i] * X[:, attributes[i]] / np.std(X[:, attributes[i]])

        return projection
    
    def hyperplane_projection_from_node(
        self, 
        x: np.ndarray, 
        node: InternalNode
    ) -> float:
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
        Computes the average gain obtained from each hyperplane, searching
        for the best split value for each of them. The scope is to find
        the hyperplane and split value that lead to the best average gain. 
        
        Parameters:
        -----------
        Ys: np.ndarray
            Projection of the data onto the hyperplanes.

        Returns:
        --------
        (index, best_split_value): (int, float)
            The index corresponding to the hyperplane found and its best split value.
        """
        best_sd_gain = 0
        best_split_value = None
        index = None

        for i, Y in enumerate(Ys):
            Y_std = np.std(Y)

            for split_value in Y:
                Y_left = Y[Y < split_value]
                Y_right = Y[Y >= split_value]
                sd_gain = self.avg_gain(Y_left, Y_right, Y_std)

                if sd_gain > best_sd_gain:
                    best_sd_gain = sd_gain
                    best_split_value = split_value
                    index = i

        return index, best_split_value

    def avg_gain(self, Y_left: np.ndarray, Y_right: np.ndarray, Y_std: float) -> float:
        """
        Computes the averaged gain for the given projections and splits.

        Parameters:
        -----------
        Y_left: np.ndarray
            Left values from the original projections, smaller then the split value.
        Y_right: np.ndarray
            Right values from the original projections, higher then the split value.
        Y_std: float
            Original standard deviation of the projections.
        """
        Y_left_std = 0
        Y_right_std = 0

        # only compute the standard deviation if the split
        # resulted in non-empty arrays
        if len(Y_left) > 0:
            Y_left_std = np.std(Y_left)

        if len(Y_right) > 0:
            Y_right_std = np.std(Y_right)

        return (Y_std - (Y_left_std + Y_right_std) / 2) / Y_std


class SCIForest:
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
        self.sci_trees: t.List[SCITree] = []

    def c(self, size: int) -> float:
        """
        Sets the expected depth of a SCITree 
        based on the number of sub-samples.

        Parameters:
        -----------
        size: int
            The number of instances for each the unsuccessful 
            search for a BST is computed on. 
        """        
        if size > 2:
            H = np.log(size) + 0.5772156649
            return 2 * H * (size - 1) - 2 * (size - 1) / size

        if size == 2:
            return 1

        return 0.0

    def fit_sci_tree(self, _) -> t.Optional[SCITree]:
        """
        Fits a single SCITree, assuming the data set 
        was already defined in the SCI forest object.
        """
        if self.X is None:
            return
        
        indexes = np.random.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        sci_tree = SCITree(height=0, height_limit=self.height_limit)
        sci_tree.fit(X_sub)

        return sci_tree

    def fit(self, X: np.ndarray) -> None:
        """
        Fits an ensemble of SCITrees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.X = X

        # use a pool to compute the trees in parallel
        with Pool(processes=self.n_processes) as pool:
            # assign the scitree training to the pool
            sci_trees = pool.map(self.fit_sci_tree, range(self.n_trees))
        
        self.sci_trees = sci_trees 
    
    def path_length(self, x: np.ndarray, sci_tree: SCITree) -> float:
        """
        Computes the path length for a given sample in the given SCITree.

        The path value starts from 0. After each movement
        from a node to another, the path is incremented
        only if the point lies inside the acceptance range:

        `[split value - limit, split value + limit]`

        After reaching an external node (leaf), the path
        calculated until that point is returned plus an adjustment
        for the subtree that was not continued beyond the height limit.
        """
        node = sci_tree.root
        path = 0

        while not isinstance(node, ExternalNode):
            # project the sample onto the hyperplane
            y = sci_tree.hyperplane_projection_from_node(x, node)

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
        among all trained SCITrees.
        """
        path_lengths = np.array([self.path_length(x, sci_tree) for sci_tree in self.sci_trees])
        return np.mean(path_lengths)

    def score(self, x: np.ndarray) -> float:
        """
        Computes the score for a single sample.
        """
        return np.power(2, -1 * self.avg_path_length(x) / self.expected_depth)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the score for a dataset with multiple samples.
        """
        scores = np.array([self.score(x) for x in X])
        return scores

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time 
    from pyod.utils.data import generate_data_clusters

    contamination = 0.1
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train=1000, n_test=200, n_clusters=2, n_features=2, contamination=contamination)
    
    sciforest = SCIForest(n_trees=100)
    start = time.time()
    sciforest.fit(X_train)
    end = time.time()
    print(f"Time: {end - start}s")

    X_train_scores = sciforest.scores(X_train)
    X_test_scores = sciforest.scores(X_test)
    
    threshold = np.quantile(X_train_scores, 1 - contamination)
    
    X_train_preds = np.array([int(label) for label in (X_train_scores > threshold)])
    X_test_preds = np.array([int(label) for label in (X_test_scores > threshold)])

    print(f"Train ACC: {np.mean(X_train_preds == y_train)}")
    print(f"Test ACC: {np.mean(X_test_preds == y_test)}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color="blue", label="normal")
    axs[0].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color="red", label="anomaly")
    axs[0].set_title("Ground truth for train")
    axs[0].legend()

    axs[1].scatter(X_train[X_train_preds == 0][:, 0], X_train[X_train_preds == 0][:, 1], color="blue", label="normal")
    axs[1].scatter(X_train[X_train_preds == 1][:, 0], X_train[X_train_preds == 1][:, 1], color="red", label="anomaly")
    axs[1].set_title("Predictions for train")
    axs[1].legend()

    plt.show()