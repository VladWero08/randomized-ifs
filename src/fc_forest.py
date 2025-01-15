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


class FCTree:
    def __init__(
        self,
        height: int, 
        height_limit: int,
        criterion: float = "height",
        min_gain: float = 0.5,
        n_attributes: int = 2,
        n_hyperplanes: int = 5,
    ) -> None:
        if criterion != "height" and criterion != "gain":
            raise "Criterion needs to be either 'height' or 'gain'!"
        
        if not (0 < min_gain < 1):
            raise "Min gain needs to be inside the interval (0, 1)!"

        self.height = height
        self.height_limit = height_limit
        self.criterion = criterion
        self.min_gain = min_gain
        self.n_attributes = n_attributes
        self.n_hyperplanes = n_hyperplanes
    
    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if self.criterion == "height":
            return self.fit_with_height(X)
        else:
            return self.fit_with_gain(X)

    def fit_with_height(self, X: np.ndarray) -> InternalNode | ExternalNode:
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
        Y_best_idx, Y_best_split_value, _ = self.hyperplane_select(Ys)
        Y = Ys[Y_best_idx]

        # compute the expectance and standard deviation 
        # for the attributes that correspond to the best hyperplane
        attrs_stds = np.array([np.std(X[:, attrs[Y_best_idx][i]]) for i in range(self.n_attributes)])
        attrs_means = np.array([np.mean(X[:, attrs[Y_best_idx][i]]) for i in range(self.n_attributes)])

        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]

        # compute the left and right side of the tree
        node_left = FCTree(self.height + 1, self.height_limit).fit(X_left)
        node_right = FCTree(self.height + 1, self.height_limit).fit(X_right)

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
    
    def fit_with_gain(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if X.shape[0] == 1:
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
        Y_best_idx, Y_best_split_value, Y_best_gain = self.hyperplane_select(Ys)

        # if the gain obtained is smaller then the minimum gain allowed,
        # stop and mark the current node as external node
        if Y_best_gain < self.min_gain:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root

        # compute the expectance and standard deviation 
        # for the attributes that correspond to the best hyperplane
        Y = Ys[Y_best_idx]
        attrs_stds = np.std(X[:, attrs[Y_best_idx]], axis=0)
        attrs_means = np.mean(X[:, attrs[Y_best_idx]], axis=0)

        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]

        # compute the left and right side of the tree
        node_left = FCTree(self.height + 1, self.height_limit).fit(X_left)
        node_right = FCTree(self.height + 1, self.height_limit).fit(X_right)

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
        return np.sum(
            coefficients * (X[:, attributes] - np.mean(X[:, attributes], axis=0)) / np.std(X[:, attributes], axis=0),
            axis=1
        )

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
        return np.sum(node.split_coeff * ((x[node.split_attr] - node.split_attr_means) / node.split_attr_stds))
    
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
            Y_std = np.std(Y)

            for split_value in Y:
                Y_left = Y[Y < split_value]
                Y_right = Y[Y >= split_value]
                pool_gain = self.pool_gain(Y_left, Y_right, Y_std)

                if pool_gain > best_pool_gain:
                    best_pool_gain = pool_gain
                    best_split_value = split_value
                    index = i

        return index, best_split_value, best_pool_gain

    def pool_gain(self, Y_left: np.ndarray, Y_right: np.ndarray, Y_std: float) -> float:
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
        n_samples_left = Y_left.shape[0]
        n_samples_right = Y_right.shape[0]

        # only compute the standard deviation if the split
        # resulted in non-empty arrays
        if len(Y_left) > 0:
            Y_left_std = np.std(Y_left)

        if len(Y_right) > 0:
            Y_right_std = np.std(Y_right)

        return (Y_std - (n_samples_left * Y_left_std + n_samples_right * Y_right_std) / (n_samples_left + n_samples_right)) / Y_std
    

class FCForest:
    def __init__(
        self,
        n_trees: int = 100,
        sub_sample_size: int = 256,
        criterion: str = "height",
        height_limit: t.Optional[int] = None,
        min_gain: float = 0.5,
        n_processes: int = 8
    ):
        if criterion != "height" and criterion != "gain":
            raise "Criterion needs to be either 'height' or 'gain'!"
        
        if not (0 < min_gain < 1):
            raise "Min gain needs to be inside the interval (0, 1)!"

        # initialize parameters passed in the constructor
        self.n_trees = n_trees
        self.sub_sample_size = sub_sample_size
        self.n_processes: int = n_processes if n_processes else multiprocessing.cpu_count()
        self.criterion: str = criterion
        self.height_limit: int = height_limit if height_limit else np.ceil(np.log2(self.sub_sample_size))
        self.min_gain: float = min_gain

        # initialize parameters used for fitting
        self.expected_depth: float = self.c(sub_sample_size)
        self.fcf_trees: t.List[FCTree] = []

    def c(self, size: int) -> float:
        """
        Sets the expected depth of a FCTree 
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
    
    def fit_fcf_tree(self, _) -> t.Optional[FCTree]:
        """
        Fits a single FCTree, assuming the data set 
        was already defined in the FCF forest object.
        """
        if self.X is None:
            return
        
        indexes = np.random.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        fcf_tree = FCTree(
            criterion=self.criterion, 
            height=0, 
            height_limit=self.height_limit,
            min_gain=self.min_gain,
        )
        fcf_tree.fit(X_sub)

        return fcf_tree

    def fit(self, X: np.ndarray):
        """
        Fits an ensemble of FCTrees for the given data.

        Parameters:
        -----------
        X: np.ndarray
            Data that needs to be fit.        
        """
        self.X = X

        # use a pool to compute the trees in parallel
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            # assign the scitree training to the pool
            fcf_trees = pool.map(self.fit_fcf_tree, range(self.n_trees))
        
        self.fcf_trees = fcf_trees 

    def path_length(self, x: np.ndarray, fcf_tree: FCTree) -> float:
        """
        Computes the path length for a given sample in the given FCTree.

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
        among all trained FCTrees.
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
        contour = ax.contourf(xx, yy, scores, levels=30, cmap="coolwarm")
        cbar = fig.colorbar(contour, ax=ax)
        ax.set_title("FCF")

        return fig
