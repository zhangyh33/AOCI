# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["lam_const"] = (2.0,2.0)
    hparams["T"] = (10,10)
    hparams["warmup_epoch"] = (100,100)
    hparams["warmup_type"] = ("sigmoid","sigmoid")
    hparams["KD_type"] = ("CIRL","CIRL")
    hparams["beta"] = (1.0,1.0)
    
    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["scheduler"] = ("const", "const")
    hparams["optimizer"] = ("adam", "adam")
    
    hparams["Fourier"] = (True, False)
    hparams["nonlinear_classifier"] = (False,False)
    
    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet
    hparams["mmd_gamma"] =[1,1]
    
    if algorithm in ['DANN', 'CDANN']:
        hparams["mlp_width"] = [256,256]
        hparams["mlp_dropout"] = [0.0,0.5]
        hparams["mlp_depth"] = [3,3]
        
        hparams["lambda"] = [1.0,1.0]
        hparams["weight_decay_d"] = [0.0,0.0]
        hparams["d_steps_per_g_step"] = [1,1]
        hparams["grad_penalty"] = [0.0,0.0]
        hparams["beta1"] = [0.5,0.5]    
        
        
    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        hparams["lr_g"] = [1e-3,1e-3]
        hparams["lr_d"] = [1e-3,1e-3]
        hparams["weight_decay_g"] = [0.0,0.0]
    elif algorithm in ['DANN', 'CDANN']:
        hparams["lr_g"] = [5e-5,5e-5]
        hparams["lr_d"] = [5e-5,5e-5]
        hparams["weight_decay_g"] = [0.0,0.01]
        
    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (1e-6, 10 ** random_state.uniform(-6, -2))

    if algorithm=="DNA":
        hparams["bottleneck_dim"] = (1024, random_state.choice([1024,2048]))
        hparams["dropout_rate"] = (0.5, random_state.choice([0.5,0.1]))
        hparams["dropout_type"] = ('Bernoulli', 'Bernoulli')
        hparams["lambda_v"] = (0.1, random_state.choice([0.01, 0.1, 1.0]))
    
    if algorithm == "RSC":
        hparams["rsc_f_drop_factor"] = (2/3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (2/3, random_state.uniform(0, 0.5))
    if algorithm == "SagNet":
        hparams["sag_w_adv"] = (0.1, 10**random_state.uniform(-2, 1))
    if algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10**random_state.uniform(-3, -1))
    if algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        hparams["mmd_gamma"] = (1, 10**random_state.uniform(-1, 1))

    if algorithm == "MLDG":
        hparams["mldg_beta"] = (1, 10**random_state.uniform(-1, 1))
        hparams["n_meta_test"] = (2, random_state.uniform(1, 2))

    if algorithm == "MTL":
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.]))
    if algorithm == "IRM":
        hparams["irm_lambda"] = (1e2, 10**random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (500, 10**random_state.uniform(0, 4))
    if algorithm == "VREx":
        hparams["vrex_lambda"] = (1e1, 10**random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (500, 10**random_state.uniform(0, 4))
        
        
        
    return hparams

    # To reproduce the results of DNA, we recommend the hyperparameters searched on the validation set:
    # PACS: (5e-5, 0.5, 1024, 0.1)
    # VLCS: (3e-5, 0.5, 1024, 0.1)
    # OfficeHome: (5e-5, 0.5, 1024, 0.1)
    # TerraIncognita: (5e-5, 0.5, 1024, 0.1)
    # DomainNet: (4e-5, 0.1, 2048, 0.1)
    # Each tuple represents (lr, dropout_rate, bottleneck_dim, lambda_v).

def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
