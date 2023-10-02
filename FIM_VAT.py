import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy import spatial
from matplotlib.cm import tab20 # type: ignore

class FIM_VAT:
    """
    Feature Importance Method for VAT
    """

    def __init__(self, X, distance_mat_norm=False):
        """
        Initialize the FIM_VAT object.

        Parameters:
        - X: DataFrame or ndarray
            Input data matrix.
        - distance_mat_norm(optional): bool, default=False
            Normalize the distance matrix.

        Attributes:
        - features: int
            Number of features.
        - R: ndarray
            Dissimilarity matrix computed using all features.
        - R_i: list
            List of dissimilarity matrices computed using individual features.
        """
        X = X.values if hasattr(X, 'values') else X
        self.features = X.shape[1]
        self._compute_distance_matrix(X, distance_mat_norm)

    def _compute_dissimilarity_matrix(self, X, is_norm):
        """
        Compute the dissimilarity matrix.

        Parameters:
        - X: ndarray
            Input data.
        - is_norm(optional): bool
            Normalize the distance matrix.

        Returns:
        - dist_mat: ndarray
            Dissimilarity matrix.
        """
        dist_mat = cdist(X, X, 'sqeuclidean')
        if is_norm:
            dist_mat = dist_mat / np.max(dist_mat)
        return dist_mat

    def _compute_distance_matrix(self, X, is_norm):
        """
        Compute the distance matrices using all features and single feature.

        Parameters:
        - X: ndarray
            Input data.
        - is_norm: bool
            Normalize the distance matrix.
        """
        self.R = self._compute_dissimilarity_matrix(X, is_norm)
        self.R_i = []
        for i in range(self.features):
            X_i = X[:, i].reshape((len(X), 1))
            R_i = self._compute_dissimilarity_matrix(X_i, is_norm)
            self.R_i.append(R_i)

    def spearman(self, d1, d2):
        """
        Compute the Spearman rank correlation between two distance matrices.

        Parameters:
        - d1: ndarray
            First distance matrix.
        - d2: ndarray
            Second distance matrix.

        Returns:
        - float
            Absolute value of the Spearman rank correlation.
        """
        x = spatial.distance.squareform(d1, force="tovector", checks=False)
        y = spatial.distance.squareform(d2, force="tovector", checks=False)
        return abs(scipy.stats.spearmanr(x, y)[0])

    def explainer(self, name):
        """
        compute feature importances.

        Parameters:
        - name: str
            Name of the dataset.
        """
        distances = np.zeros(self.features)
        for i in range(self.features):
            distances[i] = self.spearman(self.R, self.R_i[i])

        self._plot_scores(distances, name)

    def _plot_scores(self, scores, name):
        """
        Plot feature importances.

        Parameters:
        - scores: ndarray
            Feature importance scores.
        - name: str
            Name of the dataset for the title of the plot.
        """
        indices = np.argsort(scores)
        features = [i + 1 for i in range(self.features)]
        cmap = tab20
        num_features = len(features)
        color = [cmap(i / num_features) for i in range(num_features)]
        colors = [color[i] for i in indices]
        plt.figure(figsize=(10, 6), facecolor='w')
        plt.title(name, fontsize=24)
        bars = plt.barh(range(len(indices)), scores[indices], align='center')

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.yticks(range(len(indices)), [features[i]
                   for i in indices], fontsize=22)
        plt.xlabel('Feature Importances', fontsize=20)
        plt.ylabel('Features', fontsize=20)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        plt.show()
