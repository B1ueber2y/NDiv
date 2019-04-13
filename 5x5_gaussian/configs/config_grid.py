import os

OUTPUT_DIR = 'output/Gaussian_grid'

cfg_dict = {
    "IMG_DIR": os.path.join(OUTPUT_DIR, 'img_grid'),
    "MODEL_DIR": os.path.join(OUTPUT_DIR, 'models_grid'),

    "epoch": 50000,
    "batch_size": 128,
    "alpha": 0.5,
    
    "dataset":{
        "name": "gaussian_grid",
        "n": 5,
        "n_data": 50,
        "sig": 0.10**2
    },

    "show_step": 200,
    "load_model":False,
    "use_gpu": True,

    "d_step": 3,
    "div_step": 3,
    "g_step": 3,

    "ndiv_loss_ratio": 1e4,
    
    "sampler":{
        "name": "gaussian",
        "dim": 25
    },
    
    "nn_config_G":{
        "name": "DeepMLP_G",
        "hidden_size": 128
    },
    
    "nn_config_D":{
        "name": "DeepMLP_D",
        "hidden_size": 128
    },
    
    "opt_config_G":{
        "name": "adam",
        "default": False,
        "lr": 1e-3,
        "betas": (0.5, 0.999)
    },
    
    "opt_config_D":{
        "name": "adam",
        "default": False,
        "lr": 1e-3,
        "betas": (0.5, 0.999)
    },
}


