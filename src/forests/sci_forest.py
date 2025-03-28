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
        rng: np.random.Generator,
        n_attributes: int = 2, 
        n_hyperplanes: int = 5
    ) -> None:
        self.height: int = height
        self.height_limit: int = height_limit
        self.rng: np.random.Generator = rng
        self.n_attributes: int = n_attributes
        self.n_hyperplanes: int = n_hyperplanes
        self.root: InternalNode | ExternalNode = None


    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if (
            self.height >= self.height_limit or 
            X.shape[0] <= 2 or
            all(np.std(X, axis=0) <= 1e-10)
        ):
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # select the attributes that have the minimum std allowed
        attrs_valid = np.where(np.std(X, axis=0) > 1e-10)[0]
        n_attributes_tree = min(len(attrs_valid), self.n_attributes)

        # randomly select coefficients and attributes for the hyperplanes
        coeffs = self.rng.uniform(-1, 1, size=(self.n_hyperplanes, n_attributes_tree))
        attrs = np.array([
            self.rng.choice(attrs_valid, size=(n_attributes_tree), replace=False) 
            for _ in range(self.n_hyperplanes)
        ])

        # compute the projections onto the hyperplanes, and select the best one  
        Ys = np.array([self.hyperplane_projection(X, coeffs[i], attrs[i]) for i in range(self.n_hyperplanes)])        
        Y_best_idx, Y_best_split_value = self.hyperplane_select(Ys)
        Y = Ys[Y_best_idx]

        # compute the standard deviation for the attributes
        # that correspond to the best hyperplane
        attrs_stds = np.array([np.std(X[:, attrs[Y_best_idx][i]]) for i in range(n_attributes_tree)])
        
        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]
        limit = max(Y) - min(Y)

        # compute the left and right side of the tree
        node_left = SCITree(self.height + 1, self.height_limit, self.rng).fit(X_left)
        node_right = SCITree(self.height + 1, self.height_limit, self.rng).fit(X_right)

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
        return np.sum(
            coefficients * (X[:, attributes] / np.std(X[:, attributes], axis=0)),
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

        Parameters:
        -----------
        x: np.ndarray
            Single sample.
        node: InternalNode
            Current node where the sample is positioned in the SCITree.
        """
        return np.sum(node.split_coeff * ((x[node.split_attr]) / node.split_attr_stds))

    def hyperplane_select(
        self, 
        Ys: np.ndarray,
    ) -> t.Tuple[int, float]:
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
        best_sd_gain = float("-inf")
        best_split_value = None
        index = None

        for i, Y in enumerate(Ys):
            # sort the possible split values
            Y_sorted = np.sort(Y)
            Y_std = np.std(Y)
            Y_size = len(Y)

            # precompute cumulative sums for left and right sides
            Y_cumulative_sum = np.cumsum(Y_sorted)
            # total sum of the projections
            total_sum = Y_cumulative_sum[-1]

            for j in range(1, Y_size): 
                Y_left_size = j
                Y_right_size = Y_size - j

                if Y_left_size == 0 or Y_right_size == 0:
                    continue
                
                # compute means based on the cumulative sums
                Y_left_mean = Y_cumulative_sum[j - 1] / Y_left_size
                Y_right_mean = (total_sum - Y_cumulative_sum[j - 1]) / Y_right_size

                # compute standard deviations
                Y_left_std = np.sqrt(np.sum((Y_sorted[:j] - Y_left_mean) ** 2) / Y_left_size)
                Y_right_std = np.sqrt(np.sum((Y_sorted[j:] - Y_right_mean) ** 2) / Y_right_size)

                # compute pooled gain
                avg_gain = self.avg_gain(Y_std, Y_left_std, Y_right_std)    

                if avg_gain > best_sd_gain:
                    best_sd_gain = avg_gain
                    best_split_value = Y_sorted[j]
                    index = i

        return index, best_split_value

    def avg_gain(self, Y_std: float, Y_left_std: float, Y_right_std: float) -> float:
        """
        Computes the averaged gain for the given projections and splits.

        Parameters:
        -----------
        Y_std: float
            Original standard deviation of the projections.

        Y_left_std: float
            Standard deviation of the left values from the original projections, 
            smaller then the split value.

        Y_right_std: float
            Standard deviation of the right values from the original projections, 
            higher then the split value.
        """
        return (Y_std - (Y_left_std + Y_right_std) / 2) / Y_std


class SCIForest:
    def __init__(
        self, 
        n_trees: int = 100, 
        sub_sample_size: int = 256, 
        contamination: t.Optional[float] = 0.1,
        height_limit: t.Optional[int] = None,
        n_processes: int = 8,
        seed: int = 1,
    ):
        # initialize parameters passed from the constructor
        self.n_trees: int = n_trees
        self.sub_sample_size: int = sub_sample_size
        self.contamination: float = contamination
        self.n_processes: int = n_processes if n_processes else multiprocessing.cpu_count()
        self.height_limit: int = height_limit if height_limit else np.ceil(np.log2(self.sub_sample_size))
        
        # initialize parameters used for fitting
        self.expected_depth: float = self.c(sub_sample_size)
        self.sci_trees: t.List[SCITree] = []
        self.decision_scores: t.List[float] = []
        self.threshold: t.Optional[float] = None
        self.labels: t.List[int] = []

        self.rng = np.random.default_rng(seed)

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
            H = np.log(size - 1) + 0.5772156649
            return 2 * H - 2 * (size - 1) / size

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
        
        indexes = self.rng.choice(range(0, self.X.shape[0]), size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        sci_tree = SCITree(
            height=0, 
            height_limit=self.height_limit, 
            rng=self.rng
        )
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

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            self.sci_trees = pool.map(self.fit_sci_tree, range(self.n_trees))
        
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
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            scores = pool.map(self.score, X)
        
        return np.array(scores)
    
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
            np.linspace(-10, 20, 150),
            np.linspace(-10, 20, 150),
        )
        points = np.c_[xx.ravel(), yy.ravel()]
        scores = self.scores(points)
        scores = scores.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, scores, levels=50, cmap="coolwarm")
        ax.set_title("SCIF")

        return fig
