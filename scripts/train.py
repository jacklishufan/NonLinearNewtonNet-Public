#! /usr/bin/env python

import os
import argparse

import torch
from torch.optim import Adam
import yaml

import sys
sys.path.insert(0, r"C:\Users\Pumpkinfries\Desktop\Fa 22\CS 182\project\NewtonNet")

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.models import NewtonNet

from newtonnet.train import Trainer
from newtonnet.data import *

import wandb

# torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.DoubleTensor)

# argument parser description
parser = argparse.ArgumentParser(
    description=
    'This is a pacakge to train NewtonNet on a given data.'
)
parser.add_argument(
    '-c',
    "--config",
    type=str,
    required=True,
    help="The path to the Yaml configuration file.")

parser.add_argument(
    '-p',
    "--parser",
    type=str,
    required=False,
    default='ccsd',
    help="The name of dataset to select the appropriate parser. We provide data parsers for 'md17', 'ccsd', 'ani', 'hydroggen' and 'methane' data sets."\
         "For all other data sets do not specify.")

parser.add_argument(
    '-n',
    "--name",
    type=str,
    required=False,
    default='',
    help="The name of wandb log name")


parser.add_argument(
    '-k',
    "--keys",
    type=str,
    required=False,
    default='ccsd',
    help="List of keys to point to requested data. Options are 'original', 'CHNO', 'ccsd', 'dipole'.")

# define arguments
args = parser.parse_args()
config = args.config
parser = args.parser
data_keys = args.keys

# settings
settings_path = config
settings = yaml.safe_load(open(settings_path, "r"))

# device
if type(settings['general']['device']) is list:
    device = [torch.device(item) for item in settings['general']['device']]
else:
    device = [torch.device(settings['general']['device'])]

# data
# modify to use our md17 parser
if parser in ['ccsd']:
    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer = parse_train_test(settings,device[0])
    train_mode = 'energy/force'
    print('data set:', 'one of md17 data sets or a generic one.')
elif parser in ['ani']:
    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash = parse_ani_data(
        settings, 'cpu')
    train_mode = 'energy'
    print('data set:', 'ANI')
elif parser in ['ani_ccx']:
    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash = parse_ani_ccx_data(
        settings, data_keys,  'cpu')
    train_mode = 'energy'
    print('data set:', 'ANI_CCX')
elif parser in ['md17']:
    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash = parse_md17_data(
        settings, data_keys, 'cpu')
    train_mode = 'energy'
    print('data set:', 'MD17 for Test and Ani-cxx for Train')
elif parser in ['methane']:
    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash = parse_methane_data(
        settings, device[0])
    train_mode = 'energy/force'
    print('data set:', 'Methane Combustion')
elif parser in ['hydrogen']:
    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_train_data, n_val_data, n_test_data, normalizer, test_energy_hash = parse_methane_data(
        settings, device[0])
    train_mode = 'energy/force'
    print('data set:', 'Methane Combustion')

print('normalizer: ', normalizer)

# model
# activation function
activation = get_activation_by_string(settings['model']['activation'])
wandb.init(name=args.name or None)
model = NewtonNet(resolution=settings['model']['resolution'],
               n_features=settings['model']['n_features'],
               activation=activation,
               n_interactions=settings['model']['n_interactions'],
               dropout=settings['training']['dropout'],
               max_z=settings['model']['max_z'],
               cutoff=settings['data']['cutoff'],  ## data cutoff
               cutoff_network=settings['model']['cutoff_network'],
               normalizer=normalizer,
               normalize_atomic=settings['model']['normalize_atomic'],
               requires_dr=settings['model']['requires_dr'],
               device=device[0],
               create_graph=True,
               attention_heads=settings['model']['attention_heads'],
               shared_interactions=settings['model']['shared_interactions'],
               return_latent=settings['model']['return_latent'],
               double_update_latent=settings['model']['double_update_latent'],
               layer_norm=settings['model']['layer_norm'],
               nonlinear_attention=settings['model']['nonlinear_attention'],
               three_body=settings['model'].get('three_body',False)
               )

# laod pre-trained model
if settings['model']['pre_trained']:
    model_path = settings['model']['pre_trained']
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params,
                 lr=settings['training']['lr'],
                 weight_decay=settings['training']['weight_decay'])

# loss
w_energy = settings['model']['w_energy']
w_force = settings['model']['w_force']
w_f_mag = settings['model']['w_f_mag']
w_f_dir = settings['model']['w_f_dir']

def custom_loss(preds, batch_data, w_e=w_energy, w_f=w_force, w_fm=w_f_mag, w_fd=w_f_dir):
    # compute the mean squared error on the energies
    device = preds['E'].device
    diff_energy = preds['E'] - batch_data["E"].reshape(-1,1).to(device)
    assert diff_energy.shape[1] == 1
    err_sq_energy = torch.mean(diff_energy**2)
    err_sq = w_e * err_sq_energy

    # compute the mean squared error on the forces
    if w_f > 0:
        diff_forces = preds['F'] - batch_data["F"].to(device)
        err_sq_forces = torch.mean(diff_forces**2)
        err_sq = err_sq + w_f * err_sq_forces

    # compute the mean square error on the force magnitudes
    if w_fm > 0:
        diff_forces = torch.norm(preds['F'], p=2, dim=-1) - torch.norm(batch_data["F"].to(device), p=2, dim=-1)
        err_sq_mag_forces = torch.mean(diff_forces ** 2)
        err_sq = err_sq + w_fm * err_sq_mag_forces

    if w_fd > 0:
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        direction_diff = 1 - cos(preds['hs'][-1][1], batch_data["F"].to(device))
        # direction_diff = direction_diff * torch.norm(batch_data["F"], p=2, dim=-1)
        direction_loss = torch.mean(direction_diff)
        err_sq = err_sq + w_fd * direction_loss

    if settings['checkpoint']['verbose']:
        print('\n',
              ' '*8, 'energy loss: ', err_sq_energy.detach().cpu().numpy(), '\n')
        print(' '*8, 'force loss: ', err_sq_forces.detach().cpu().numpy(), '\n')

        if w_fm>0:
            print(' '*8, 'force mag loss: ', err_sq_mag_forces, '\n')

        if w_fd>0:
            print(' '*8, 'direction loss: ', direction_loss.detach().cpu().numpy())

    return err_sq


# training
trainer = Trainer(model=model,
                  loss_fn=custom_loss,
                  optimizer=optimizer,
                  requires_dr=settings['model']['requires_dr'],
                  device=device,
                  yml_path=args.config,
                  output_path=settings['general']['output'],
                  script_name=settings['general']['driver'],
                  lr_scheduler=settings['training']['lr_scheduler'],
                  energy_loss_w= w_energy,
                  force_loss_w=w_force,
                  loss_wf_decay=settings['model']['wf_decay'],
                  checkpoint_log=settings['checkpoint']['log'],
                  checkpoint_val=settings['checkpoint']['val'],
                  checkpoint_test=settings['checkpoint']['test'],
                  checkpoint_model=settings['checkpoint']['model'],
                  verbose=settings['checkpoint']['verbose'],
                  hooks=settings['hooks'],
                  mode=train_mode)

trainer.print_layers()

# tr_steps=1; val_steps=0; irc_steps=0; test_steps=0

trainer.train(train_generator=train_gen,
              epochs=settings['training']['epochs'],
              steps=tr_steps,
              val_generator=val_gen,
              val_steps=val_steps,
              irc_generator=None,
              irc_steps=None,
              test_generator=test_gen,
              test_steps=test_steps,
              clip_grad=1.0)

print('done!')
