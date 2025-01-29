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
        criterion: float = "normal",
        min_gain: float = 0.5,
        n_attributes: int = 2,
        n_hyperplanes: int = 1,
    ) -> None:
        if criterion != "normal" and criterion != "gain":
            raise Exception("Criterion needs to be either 'height' or 'gain'!")
        
        if not (0 < min_gain < 1):
            raise Exception("Min gain needs to be inside the interval (0, 1)!")

        self.height: int = height
        self.height_limit: int = height_limit
        self.criterion: str = criterion
        self.min_gain: float = min_gain
        self.n_attributes: int = n_attributes
        self.n_hyperplanes: int = n_hyperplanes
    
    def fit(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if self.criterion == "normal":
            return self.fit_normal(X)
        else:
            return self.fit_with_gain(X)

    def fit_normal(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if X.shape[0] == 1:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # extract the attributes that have more then 
        # one unique value in their column
        attrs_valid = np.apply_along_axis(lambda col: len(np.unique(col)) > 1, axis=0, arr=X)

        if not any(attrs_valid):
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # select the indexes of the valid attributes
        attrs_valid_idx = np.where(attrs_valid == True)[0]

        # randomly select coefficients and attributes for the hyperplane
        coeffs = np.random.normal(loc=0, scale=1, size=self.n_attributes)
        attrs = np.random.choice(attrs_valid_idx, size=self.n_attributes, replace=True) 

        # compute the projections onto the hyperplane, and select the best one
        Y = self.hyperplane_projection(X, coeffs, attrs)        
        Y_best_split_value, _ = self.split_value_select(Y)

        # compute the means and standard deviations 
        attrs_stds = np.std(X[:, attrs]) 
        attrs_means = np.mean(X[:, attrs]) 

        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]

        # compute the left and right side of the tree
        node_left = FCTree(self.height + 1, self.height_limit).fit(X_left)
        node_right = FCTree(self.height + 1, self.height_limit).fit(X_right)

        self.root = InternalNode(
            node_left, 
            node_right, 
            Y_best_split_value, 
            coeffs, 
            attrs, 
            attrs_means, 
            attrs_stds
        )
        return self.root
    
    def fit_with_gain(self, X: np.ndarray) -> InternalNode | ExternalNode:
        if X.shape[0] == 1:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # extract the attributes that have more then 
        # one unique value in their column
        attrs_valid = np.apply_along_axis(lambda col: len(np.unique(col)) > 1, axis=0, arr=X)

        if not any(attrs_valid):
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root
        
        # select the indexes of the valid attributes
        attrs_valid_idx = np.where(attrs_valid == True)[0]

        # randomly select coefficients and attributes for the hyperplane
        coeffs = np.random.normal(loc=0, scale=1, size=self.n_attributes)
        attrs = np.random.choice(attrs_valid_idx, size=self.n_attributes, replace=True) 

        # compute the projections onto the hyperplane, and select the best one
        Y = self.hyperplane_projection(X, coeffs, attrs)        
        Y_best_split_value, Y_best_gain = self.split_value_select(Y)

        # if the gain obtained is smaller then the minimum gain allowed,
        # stop and mark the current node as external node
        if Y_best_gain < self.min_gain:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root

        # compute the means and standard deviations 
        attrs_stds = np.std(X[:, attrs]) 
        attrs_means = np.mean(X[:, attrs]) 

        X_left = X[Y < Y_best_split_value]
        X_right = X[Y >= Y_best_split_value]

        # compute the left and right side of the tree
        node_left = FCTree(self.height + 1, self.height_limit).fit(X_left)
        node_right = FCTree(self.height + 1, self.height_limit).fit(X_right)

        self.root = InternalNode(
            node_left, 
            node_right, 
            Y_best_split_value, 
            coeffs, 
            attrs, 
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
            Current node where the sample is positioned in the FCTree.
        """
        return np.sum(node.split_coeff * ((x[node.split_attr] - node.split_attr_means) / node.split_attr_stds))
    
    def split_value_select(self, Y: np.ndarray) -> t.Tuple[int, float]:
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

        # sort the possible split values
        Y_sorted = np.sort(Y)   
        Y_std = np.std(Y)
        Y_size = len(Y)

        # precompute cumulative sums for left and right sides
        Y_cumulative_sum = np.cumsum(Y_sorted)
        # total sum of the projections
        total_sum = Y_cumulative_sum[-1]

        for i in range(1, Y_size): 
            Y_left_size = i
            Y_right_size = Y_size - i

            if Y_left_size == 0 or Y_right_size == 0:
                continue
            
            # compute means based on the cumulative sums
            Y_left_mean = Y_cumulative_sum[i - 1] / Y_left_size
            Y_right_mean = (total_sum - Y_cumulative_sum[i - 1]) / Y_right_size

            # compute standard deviations
            Y_left_std = np.sqrt(np.sum((Y_sorted[:i] - Y_left_mean) ** 2) / Y_left_size)
            Y_right_std = np.sqrt(np.sum((Y_sorted[i:] - Y_right_mean) ** 2) / Y_right_size)

            # compute pooled gain
            pool_gain = self.pool_gain(Y_size, Y_std, Y_left_size, Y_left_std, Y_right_size, Y_right_std)

            if pool_gain > best_pool_gain:
                best_pool_gain = pool_gain
                best_split_value = Y_sorted[i]

        return best_split_value, best_pool_gain

    def pool_gain(
        self, 
        Y_size: float,
        Y_std: float, 
        Y_left_size: float,
        Y_left_std: float, 
        Y_right_size: float,
        Y_right_std: float, 
    
    ) -> float:
        """
        Computes the averaged gain for the given projections and splits.

        Parameters:
        -----------
        Y_size: float
            Original size of the projections.

        Y_std: float
            Original standard deviation of the projections.

        Y_left_size: float
            Original size of the projections smaller then the split value.

        Y_left_std: float
            Standard deviation of the left values from the original projections, 
            smaller then the split value.

        Y_right_size: float
            Original size of the projections higher then the split value.
        
        Y_right_std: float
            Standard deviation of the right values from the original projections, 
            higher then the split value.
        """
        return (Y_std - (Y_left_size * Y_left_std + Y_right_size * Y_right_std) / Y_size) / Y_std
    

class FCForest:
    def __init__(
        self,
        n_trees: int = 100,
        sub_sample_size: int = 256,
        contamination: float = 0.1,
        criterion: str = "height",
        height_limit: t.Optional[int] = None,
        min_gain: float = 0.5,
        n_processes: int = 8
    ):
        if criterion != "height" and criterion != "gain":
            raise Exception("Criterion needs to be either 'height' or 'gain'!")
        
        if not (0 < min_gain < 1):
            raise Exception("Min gain needs to be inside the interval (0, 1)!")

        # initialize parameters passed from the constructor
        self.n_trees: int = n_trees
        self.sub_sample_size: int = sub_sample_size
        self.contamination: float = contamination
        self.n_processes: int = n_processes if n_processes else multiprocessing.cpu_count()
        self.criterion: str = criterion
        self.height_limit: int = height_limit if height_limit else np.ceil(np.log2(self.sub_sample_size))
        self.min_gain = min_gain

        # initialize parameters used for fitting
        self.expected_depth: float = self.c(sub_sample_size)
        self.fcf_trees: t.List[FCTree] = []
        self.decision_scores: t.List[float] = []
        self.threshold: t.Optional[float] = None
        self.labels: t.List[int] = []

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
            # H = np.log(size - 1) + 0.5772156649
            # return 2 * H - 2 * (size - 1) / size
            return np.log2(size)

        if size == 2:
            return 1

        return 0.0
    
    def fit_fc_tree(self, _) -> t.Optional[FCTree]:
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

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            fcf_trees = pool.map(self.fit_fc_tree, range(self.n_trees))
        
        self.fcf_trees = fcf_trees 

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
        path_lengths = np.array([self.path_length(x, fcf_tree) for fcf_tree in self.fcf_trees])
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
            np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 100),
            np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 100),
        )
        points = np.c_[xx.ravel(), yy.ravel()]
        scores = self.scores(points)
        scores = scores.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, scores, levels=15, cmap="coolwarm")
        ax.set_title("FCF")

        return fig
