import typing as t
import numpy as np
import rrcf
import multiprocessing

class RRCForest:
    def __init__(
        self,
        n_trees: int = 100,
        sub_sample_size: int = 256,
        height_limit: int = 8,
        contamination: float = 0.1,
        n_processes: int = 8,
        seed: int = 1,
    ):
        # initialize parameters passed from the constructor
        self.n_trees: int = n_trees
        self.sub_sample_size: int = sub_sample_size
        self.contamination: float = contamination
        self.n_processes: int = n_processes

        # initialize parameters used for fitting
        self.rrc_trees: t.List[rrcf.RCTree] = []
        self.decision_scores: t.List[float] = []
        self.threshold: t.Optional[float] = None
        self.labels: t.List[int] = []
        self.seed = seed

        self.rng = np.random.default_rng(seed)

    def fit_rrc_tree(self) -> t.Optional[rrcf.RCTree]:
        """Fits a single RCTree."""
        if self.X is None:
            return None
        
        indexes = self.rng.choice(self.X.shape[0], size=self.sub_sample_size, replace=False)
        X_sub = self.X[indexes]

        # create the tree
        rrc_tree = rrcf.RCTree()

        # insert the points with respect to their true index value
        for i in range(self.sub_sample_size):
            index = indexes[i]
            rrc_tree.insert_point(X_sub[i], index=index)

        return rrc_tree
    
    def fit(self, X: np.ndarray) -> None:
        """Fits an ensemble of RRC trees for the given data."""
        self.X = X

        self.rrc_trees = [self.fit_rrc_tree() for _ in range(self.n_trees)]

        # compute the scores for the training data
        self.decision_scores = self.scores()
        # compute the threshold and labels for the training data
        self.threshold = np.quantile(self.decision_scores, 1 - self.contamination)
        self.labels = (self.decision_scores > self.threshold).astype(int)

    def score(self, x: np.ndarray, index: int) -> np.ndarray:
        """Computes the score for a single samples."""
        rrc_forest_score = []

        for rrc_tree in self.rrc_trees:
            # check if the index is already part of the tree
            if index not in rrc_tree.leaves:
                rrc_tree.insert_point(x, index)
            
                # compute and add the score to the forest score
                rrc_tree_score = rrc_tree.codisp(index)
                rrc_forest_score.append(rrc_tree_score)

                rrc_tree.forget_point(index)
            else:
                # compute and add the score to the forest score
                rrc_tree_score = rrc_tree.codisp(index)
                rrc_forest_score.append(rrc_tree_score)


        return np.mean(np.array(rrc_forest_score))

    def scores(self) -> np.ndarray:
        """Computes the score for the dataset used for train."""
        return np.array([self.score(x, index) for index, x in enumerate(self.X)])