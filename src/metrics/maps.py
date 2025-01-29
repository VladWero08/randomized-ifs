import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from forests.i_forest import IForest
from forests.ei_forest import EIForest
from forests.sci_forest import SCIForest
from forests.fc_forest import FCForest


def figure_to_img(fig: Figure) -> np.ndarray:
    io_buffer = io.BytesIO()
    
    # save the figure into the buffer
    fig.savefig(io_buffer, format="png", dpi=300)
    # close the figure immediately so it won't be shown
    plt.close(fig)

    io_buffer.seek(0)
    image = np.frombuffer(io_buffer.getvalue(), dtype=np.uint8)    
    io_buffer.close()

    image = cv2.imdecode(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def compare_methods(
    X_train: np.ndarray,
    save: bool = False,    
) -> None:
    """
    Given a dataset for training, run each forest algorithm
    and plot the decision areas for each one in a single figure.

    Each forest will be run on the defaul settings:
    - trees = 100
    - sample size = 256
    - criterion to stop = height
    
    Parameters:
    -----------
    X_train: np.ndarray
        Dataset to train the forests on.
    save: bool = False
        Whether to save each decision area as a PNG file.
    """
    if X_train.shape[1] != 2:
        raise Exception("Training data must be 2D to compare decision areas!")
    
    i_forest = IForest()
    i_forest.fit(X_train)
    i_forest_fig = i_forest.decision_area()
    i_forest_img = figure_to_img(i_forest_fig)
    print("IForest decision area computed...")

    ei_forest = EIForest()
    ei_forest.fit(X_train)
    ei_forest_fig = ei_forest.decision_area()
    ei_forest_img = figure_to_img(ei_forest_fig)
    print("EIForest decision area computed...")

    sci_forest = SCIForest()
    sci_forest.fit(X_train)
    sci_forest_fig = sci_forest.decision_area()
    sci_forest_img = figure_to_img(sci_forest_fig)
    print("SCIForest decision area computed...")

    fc_forest = FCForest()
    fc_forest.fit(X_train)
    fc_forest_fig = fc_forest.decision_area()
    fc_forest_img = figure_to_img(fc_forest_fig)
    print("FCForest decision area computed...")

    # plot the decision areas
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, img in enumerate([i_forest_img, ei_forest_img, sci_forest_img, fc_forest_img]):
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # save the plots
    if save:
        i_forest_fig.savefig("if_decision_area", format="png", dpi=300)
        ei_forest_fig.savefig("eif_decision_area", format="png", dpi=300)
        sci_forest_fig.savefig("sci_decision_area", format="png", dpi=300)
        fc_forest_fig.savefig("fc_decision_area", format="png", dpi=300)
