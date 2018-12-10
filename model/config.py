import os
from keras import losses

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))

# ---------------------------------------------------------------------------------
# ------------------------ EMBEDDING MODEL CONFIGURATION --------------------------
# ---------------------------------------------------------------------------------


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


embedding_cfg = DotDict({

    # Define which embedding to use
    'glove_embedding_len': 50,
    
    # Let's define some control variables, such as the max length of heads and desc
    # that we will use in our model
    
    # General input shapes
    'max_headline_len': 20,
    'max_article_len': 40,
    'min_headline_len': 5,
    'min_article_len': 10,
    
    # Split ratio
    'test_ratio': 0.2,

    'tot_epochs': 50,  # Number of epochs to train for.
    'epochs_per_chunk': 1,  # Number of epochs to train each chunk on
    'latent_dim': 512,  # Latent dimensionality of the encoding space.

    # Tensorboard log dir
    'tensorboard_log_dir': os.path.join(root_path, 'tensorboard/emb'),
    
    # Output layer config
    'dense_activation': 'linear',
    
    # Optimizer
    'optimizer': 'rmsprop',

    # Loss
    'loss': losses.cosine_proximity,

    'preprocess_data': True,

    'preprocess_folder': 'tokenized/emb/'
})


# -------------------------------------------------------------------------------------
# --------------------------------- END CONFIGURATION ---------------------------------
# -------------------------------------------------------------------------------------

