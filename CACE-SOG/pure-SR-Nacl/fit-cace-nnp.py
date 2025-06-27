#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../cace/')

import numpy as np
import torch
import torch.nn as nn
import logging
import ase

import datetime
save_folder = "../CACE-SOG/pure-SR-Nacl/loss_data/Ewald_"
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 4

xyz_path = '../CACE-SOG/pure-SR-Nacl/Train.xyz'
xyz = ase.io.read(xyz_path,':')
avge0 = cace.tools.compute_average_E0s(xyz)

LLLLL =0.1
val_frac = LLLLL

print("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path=xyz_path,
                                 valid_fraction=val_frac,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces'}, 
                                 atomic_energies=avge0 # avg
                                 )
batch_size = 2

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=4,
                              )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = cace.tools.init_device(use_device)
print(f"device: {device}")


print("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[11,17],
    n_atom_basis=3,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    #avg_num_neighbors=1,
    device=device,
    timeit=False
           )

cace_representation.to(device)
print(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)


forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

print("building CACE NNP")
cace_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)


q = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=1,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.EwaldPotential(dl=1,
                    sigma=1,
                    feature_key='q',
                    output_key='ewald_potential',
                    remove_self_interaction=False,
                   aggregation_mode='sum')

forces_lr = cace.modules.Forces(energy_key='ewald_potential',
                                    forces_key='ewald_forces')

cace_nnp_lr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, ep, forces_lr]
)

pot2 = {'CACE_energy': 'ewald_potential', 
        'CACE_forces': 'ewald_forces',
        'weight': 1
       }

pot1 = {'CACE_energy': 'CACE_energy', 
        'CACE_forces': 'CACE_forces',
       }

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1,pot2])
# if only using the sr part
#cace_nnp = cace.models.CombinePotential([cace_nnp_sr], [pot1])
cace_nnp.to(device)


print(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
print("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

boost = int((1./(1.-LLLLL))**0.5)

for i in range(5*boost):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False, #True,
        ema_start=10,
        warmup_steps=5,
        save_folder=save_folder,
        time_name=time_name,
    )

    print("training")
    task.fit(train_loader, valid_loader, epochs=50*boost, screen_nan=False)

task.save_model(save_folder+'pointcharge-model.pth')
cace_nnp.to(device)

print(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

task.update_loss([energy_loss, force_loss])
print("training")
task.fit(train_loader, valid_loader, epochs=100*boost, screen_nan=False)


task.save_model(save_folder+'pointcharge-model-2.pth')
cace_nnp.to(device)

print(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100*boost, screen_nan=False)

task.save_model(save_folder+'pointcharge-model-3.pth')

print(f"Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100*boost, screen_nan=False)

task.save_model(save_folder+'pointcharge-model-4.pth')

print(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")



