# %%

import os
import sys
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
import numpy as np
import einops
from pathlib import Path
from jaxtyping import Float, Int
from typing import Optional, Union, Callable, List, Tuple
from tqdm.auto import tqdm
from dataclasses import dataclass
from functools import partial
from rich import print as rprint
import random
import matplotlib.pyplot as plt
# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, hist
from utils import (
    plot_features_in_2d,
    plot_features_in_Nd,
    plot_correlated_features,
    plot_feature_geometry,
    frac_active_line_plot,
)
import tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


# %%

def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    n_correlated: int = 0


class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, t.Tensor]] = None,
        importance: Optional[Union[float, t.Tensor]] = None,
        device = t.device("cuda" if t.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None:
            feature_probability = t.ones((cfg.n_instances, cfg.n_features))
        elif isinstance(feature_probability, float):
            feature_probability = t.full((cfg.n_instances, cfg.n_features), feature_probability)
        elif feature_probability.shape != (cfg.n_instances, cfg.n_features):
            feature_probability = feature_probability.expand(cfg.n_instances, cfg.n_features)
        self.feature_probability = feature_probability.to(device)

        if importance is None:
            importance = t.ones((cfg.n_instances, cfg.n_features))
        elif isinstance(importance, float):
            importance = t.full((cfg.n_instances, cfg.n_features), importance)
        elif importance.shape != (cfg.n_instances, cfg.n_features):
            importance = importance.expand(cfg.n_instances, cfg.n_features)
        self.importance = importance.to(device)

        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        return F.relu(out + self.b_final)


    def generate_correlated_features_pairs(self, batch_size, n_correlated_pairs) -> t.Tensor:
        feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_correlated_pairs), device=self.W.device)
        feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.W.device)
        feat_set_is_present = feat_set_seeds <= self.feature_probability[:, :n_correlated_pairs]
        feat_is_present = einops.repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
        return t.where(feat_is_present, feat, 0.0)

    def generate_correlated_features(self, batch_size, n_features) -> t.Tensor:
        feat = t.rand((batch_size, self.cfg.n_instances, n_features), device=self.W.device)
        corr_matrix = t.rand((n_features, n_features), device=self.W.device)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        corr_matrix.fill_diagonal_(1)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                correlation = corr_matrix[i, j]
                mask = t.rand((batch_size, self.cfg.n_instances), device=self.W.device) <= correlation
                feat[:, :, j] = t.where(mask, feat[:, :, i], feat[:, :, j])
        
        feat_is_present = t.rand((batch_size, self.cfg.n_instances, n_features), device=self.W.device) <= self.feature_probability[:, :n_features]
        return t.where(feat_is_present, feat, 0.0)

    def generate_anticorrelated_features(self, batch_size, n_anticorrelated_pairs) -> t.Tensor:
        feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs), device=self.W.device)
        feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
        first_feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
        feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, :n_anticorrelated_pairs]
        first_feat_is_present = first_feat_seeds <= 0.5
        first_feats = t.where(feat_set_is_present & first_feat_is_present, feat[:, :, :n_anticorrelated_pairs], 0.0)
        second_feats = t.where(feat_set_is_present & (~first_feat_is_present), feat[:, :, n_anticorrelated_pairs:], 0.0)
        return einops.rearrange(t.cat([first_feats, second_feats], dim=-1), "batch instances (pair features) -> batch instances (features pair)", pair=2)

    def generate_uncorrelated_features(self, batch_size, n_uncorrelated) -> t.Tensor:
        feat = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
        feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability[:, :n_uncorrelated]
        return t.where(feat_is_present, feat, 0.0)

    def generate_batch(self, batch_size):
        data = []
        n_features_accounted = 0

        if hasattr(self.cfg, 'n_correlated') and self.cfg.n_correlated > 0:
            correlated_features = self.generate_correlated_features(batch_size, self.cfg.n_correlated)
            data.append(correlated_features)
            n_features_accounted += self.cfg.n_correlated
        elif self.cfg.n_correlated_pairs > 0:
            correlated_pairs = self.generate_correlated_features_pairs(batch_size, self.cfg.n_correlated_pairs)
            data.append(correlated_pairs)
            n_features_accounted += 2 * self.cfg.n_correlated_pairs

        if self.cfg.n_anticorrelated_pairs > 0:
            anticorrelated_features = self.generate_anticorrelated_features(batch_size, self.cfg.n_anticorrelated_pairs)
            data.append(anticorrelated_features)
            n_features_accounted += 2 * self.cfg.n_anticorrelated_pairs

        n_uncorrelated = self.cfg.n_features - n_features_accounted
        if n_uncorrelated > 0:
            uncorrelated_features = self.generate_uncorrelated_features(batch_size, n_uncorrelated)
            data.append(uncorrelated_features)

        batch = t.cat(data, dim=-1)
        return batch


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        '''
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss


    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        n_enhanced_feats: int = 3
    ):
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)
        progress_bar = tqdm(range(steps))
        
        # Dictionary to store data for logging
        data_log = {"W": [], "colors": [], "titles": []}
        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Log data
            if step % log_freq == 0 or (step + 1 == steps):
                # Create colors based on feature probabilities
                colors = []
                for i in range(self.cfg.n_instances):
                    instance_colors = []
                    for j in range(self.cfg.n_features):
                        if j < n_enhanced_feats:
                            instance_colors.append('red')  # Enhanced features
                        else:
                            instance_colors.append('blue')  # Other features
                    colors.append(instance_colors)

                data_log["W"].append(self.W.detach().cpu().clone())
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}")
                progress_bar.set_postfix(loss=loss.item()/self.cfg.n_instances, lr=step_lr)

        return data_log

@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False
    weight_normalize_eps: float = 1e-8


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig):
        super(AutoEncoder, self).__init__()
        self.cfg = cfg

        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
        
        self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))

        self.to(device)

    def normalize_and_return_W_dec(self) -> Float[Tensor, "n_instances n_hidden_ae n_input_ae"]:
        '''
        If self.cfg.tied_weights = True, we return the normalized & transposed encoder weights.
        If self.cfg.tied_weights = False, we normalize the decoder weights in-place, and return them.

        Normalization should be over the `n_input_ae` dimension, i.e. each feature should have a noramlized decoder weight.
        '''
        if self.cfg.tied_weights:
            return self.W_enc.transpose(-1, -2) / (self.W_enc.transpose(-1, -2).norm(dim=1, keepdim=True) + self.cfg.weight_normalize_eps)
        else:
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=2, keepdim=True) + self.cfg.weight_normalize_eps)
            return self.W_dec

    def forward(self, h: Float[Tensor, "batch_size n_instances n_input_ae"]):

        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, self.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean(-1) # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @t.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size n_instances n_input_ae"],
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> Tuple[List[List[str]], str]:
        '''
        Resamples neurons that have been dead for 'dead_feature_window' steps, according to `frac_active`.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (n_input_ae,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron

        Returns colors and titles (useful for creating the animation: resampled neurons appear in red).
        '''
        l2_loss = self.forward(h)[1]

        # Create an object to store the dead neurons (this will be useful for plotting)
        dead_features_mask = t.empty((self.cfg.n_instances, self.cfg.n_hidden_ae), dtype=t.bool, device=self.W_enc.device)

        for instance in range(self.cfg.n_instances):

            # Find the dead neurons in this instance. If all neurons are alive, continue
            is_dead = (frac_active_in_window[:, instance].sum(0) < 1e-8)
            dead_features_mask[instance] = is_dead
            dead_features = t.nonzero(is_dead).squeeze(-1)
            alive_features = t.nonzero(~is_dead).squeeze(-1)
            n_dead = dead_features.numel()
            if n_dead == 0: 
                continue # If we have no dead features, then we don't need to resample

            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss[:, instance] # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue # If we have zero reconstruction loss, we don't need to resample

            # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
            distn = Categorical(probs = l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum())
            replacement_indices = distn.sample((n_dead,)) # shape [n_dead]

            # Index into the batch of hidden activations to get our replacement values
            replacement_values = (h - self.b_dec)[replacement_indices, instance] # shape [n_dead n_input_ae]
            replacement_values_normalized = replacement_values / (replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps)

            # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
            W_enc_norm_alive_mean = 1.0 if len(alive_features) == 0 else self.W_enc[instance, :, alive_features].norm(dim=0).mean().item()

            # Lastly, set the new weights & biases
            # For W_dec (the dictionary vectors), we just use the normalized replacement values
            self.W_dec.data[instance, dead_features, :] = replacement_values_normalized
            # For W_enc (the encoder vectors), we use the normalized replacement values scaled by (mean alive neuron norm * neuron resample scale)
            self.W_enc.data[instance, :, dead_features] = replacement_values_normalized.T * W_enc_norm_alive_mean * neuron_resample_scale
            # For b_enc (the encoder bias), we set it to zero
            self.b_enc.data[instance, dead_features] = 0.0

        # Return data for visualising the resampling process
        colors = [["red" if dead else "black" for dead in dead_feature_mask_inst] for dead_feature_mask_inst in dead_features_mask]
        title = f"resampling {dead_features_mask.sum()}/{dead_features_mask.numel()} neurons (shown in red)"
        return colors, title

    def optimize(
        self,
        model: Model,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        '''
        Optimizes the autoencoder using the given hyperparameters.

        This function should take a trained model as input.
        '''
        if neuron_resample_window is not None:
            assert (dead_neuron_window is not None) and (dead_neuron_window < neuron_resample_window)

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"W_enc": [], "W_dec": [], "colors": [], "titles": [], "frac_active": []}
        colors = None
        title = "no resampling yet"

        for step in progress_bar:

            # Resample dead neurons
            if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
                # Get the fraction of neurons active in the previous window
                frac_active_in_window = t.stack(frac_active_list[-neuron_resample_window:], dim=0)
                # Compute batch of hidden activations which we'll use in resampling
                batch = model.generate_batch(batch_size)
                h = einops.einsum(
                    batch, model.W,
                    "batch_size instances features, instances hidden features -> batch_size instances hidden"
                )
                # Resample
                colors, title = self.resample_neurons(h, frac_active_in_window, neuron_resample_scale)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                features = model.generate_batch(batch_size)
                h = einops.einsum(
                    features, model.W,
                    "... instances features, instances hidden features -> ... instances hidden"
                )

            # Optimize
            l1_loss, l2_loss, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate the mean sparsities over batch dim for each (instance, feature)
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(l1_loss=self.cfg.l1_coeff * l1_loss.mean(0).sum().item(), l2_loss=l2_loss.mean(0).sum().item(), lr=step_lr)
                data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                data_log["W_dec"].append(self.normalize_and_return_W_dec().detach().cpu().clone())
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}: {title}")
                data_log["frac_active"].append(frac_active.detach().cpu().clone())

        return data_log

# %%

def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class ExperimentConfig:
    def __init__(self, 
                 n_instances=8,
                 n_features=8,
                 n_hidden=2,
                 n_correlated=0,
                 n_anticorrelated_pairs=0,
                 pre_steps=20_000,
                 post_steps=20_000,
                 pre_log_freq=200,
                 post_log_freq=100,
                 pre_lr=1e-3,
                 post_lr=1e-4,
                 pre_feature_probability=0.005,
                 post_feature_probability_base=0.001,
                 post_feature_probability_enhanced=0.03,
                 n_enhanced_features=3,
                 ae_n_input=2,
                 ae_n_hidden=8,
                 ae_l1_coeff=0.25):
        
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_correlated = n_correlated
        self.n_anticorrelated_pairs = n_anticorrelated_pairs
        
        self.pre_steps = pre_steps
        self.post_steps = post_steps
        self.pre_log_freq = pre_log_freq
        self.post_log_freq = post_log_freq
        self.pre_lr = pre_lr
        self.post_lr = post_lr
        
        self.pre_feature_probability = pre_feature_probability
        self.post_feature_probability_base = post_feature_probability_base
        self.post_feature_probability_enhanced = post_feature_probability_enhanced
        self.n_enhanced_features = n_enhanced_features
        
        self.ae_n_input = ae_n_input
        self.ae_n_hidden = ae_n_hidden
        self.ae_l1_coeff = ae_l1_coeff

    def get_model_config(self):
        return Config(
            n_instances=self.n_instances,
            n_features=self.n_features,
            n_hidden=self.n_hidden,
            n_correlated=self.n_correlated,
            n_anticorrelated_pairs=self.n_anticorrelated_pairs,
        )

    def get_ae_config(self):
        return AutoEncoderConfig(
            n_instances=self.n_instances,
            n_input_ae=self.n_hidden,  
            n_hidden_ae=self.ae_n_hidden,
            l1_coeff=self.ae_l1_coeff,
        )

    def get_pre_feature_probability(self):
        return t.full((self.n_instances, self.n_features), self.pre_feature_probability)

    def get_post_feature_probability(self):
        prob = t.full((self.n_instances, self.n_features), self.post_feature_probability_base)
        prob[:, :self.n_enhanced_features] = self.post_feature_probability_enhanced
        return prob


def plot_sae(sae_data_log, folder, suffix, save='SAE.gif'):
    # We'll be plotting encoder & decoder on the first & second rows
    titles = [title + ", first row = encoder, second row = decoder" for title in sae_data_log["titles"]]

    # Stack encoder and decoder along the n_instances dimension
    data = t.concat([
        t.stack(sae_data_log["W_enc"], dim=0),
        t.stack(sae_data_log["W_dec"], dim=0).transpose(-1, -2)
    ], dim=1)

    save_path = os.path.join(folder, suffix, save)
    ensure_dir(save_path)

    plot_features_in_2d(
        data,
        colors=sae_data_log["colors"],
        title=titles,
        n_rows=2,
        save=save_path
    )

def plot_feature_dynamics(model_data_log, continued_model_data_log, exp_cfg, folder, suffix):
    all_W = model_data_log["W"] + continued_model_data_log["W"]
    initial_steps = len(model_data_log["W"]) * exp_cfg.pre_log_freq
    continued_steps = len(continued_model_data_log["W"]) * exp_cfg.post_log_freq

    steps = (list(range(0, initial_steps + 1, exp_cfg.pre_log_freq)) +
             list(range(initial_steps, initial_steps + continued_steps + 1, exp_cfg.post_log_freq)))

    # Ensure steps and W data have the same length
    min_length = min(len(steps), len(all_W))
    steps = steps[:min_length]
    all_W = all_W[:min_length]

    feature_norms = [t.norm(W, dim=1).mean(dim=0).cpu().numpy() for W in all_W]

    plt.figure(figsize=(12, 6))
    for i in range(exp_cfg.n_features):  
        plt.plot(steps, [norm[i] for norm in feature_norms], label=f'Feature {i+1}')

    plt.axvline(x=initial_steps, color='r', linestyle='--', label='Distribution Change')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Feature Norm')
    plt.title('Average Feature Dynamics Across Instances During Training')
    plt.tight_layout()
    save_path = os.path.join(folder, suffix, 'feature_dynamics.png')
    ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close()


def plot_sae_loss(model, autoencoder, continued_model_data_log, exp_cfg, folder, suffix, batch_size=1000):
    sae_losses = []
    initial_steps = exp_cfg.pre_steps
    steps = list(range(initial_steps, initial_steps + len(continued_model_data_log["W"]) * exp_cfg.post_log_freq, exp_cfg.post_log_freq))

    original_W = model.W.clone()

    for W in continued_model_data_log["W"]:
        with t.no_grad():
            model.W.data.copy_(W)
            features = model.generate_batch(batch_size)
            h = einops.einsum(
                features, model.W,
                "... instances features, instances hidden features -> ... instances hidden"
            )
            _, _, loss, _, _ = autoencoder.forward(h)
            sae_losses.append(loss.item())

    with t.no_grad():
        model.W.data.copy_(original_W)

    # Ensure steps and sae_losses have the same length
    min_length = min(len(steps), len(sae_losses))
    steps = steps[:min_length]
    sae_losses = sae_losses[:min_length]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, sae_losses)
    plt.xlabel('Fine-tuning Steps')
    plt.ylabel('SAE Loss')
    plt.title('SAE Loss During Model Fine-tuning')
    plt.tight_layout()
    save_path = os.path.join(folder, suffix, 'sae_loss_during_finetuning.png')
    ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close()

def plot_cosine_similarities(model_data_log, continued_model_data_log, exp_cfg, folder, suffix):
    all_W = model_data_log["W"] + continued_model_data_log["W"]
    initial_steps = len(model_data_log["W"]) * exp_cfg.pre_log_freq
    continued_steps = len(continued_model_data_log["W"]) * exp_cfg.post_log_freq

    steps = (list(range(0, initial_steps + 1, exp_cfg.pre_log_freq)) +
             list(range(initial_steps, initial_steps + continued_steps + 1, exp_cfg.post_log_freq)))

    min_length = min(len(steps), len(all_W))
    steps = steps[:min_length]
    all_W = all_W[:min_length]

    n_features = all_W[0].shape[2]
    cosine_sims = []

    for W in all_W:
        # Reshape W to [instances * hidden, features]
        W_reshaped = W.reshape(-1, n_features)

        # Normalize W
        W_normalized = W_reshaped / W_reshaped.norm(dim=0, keepdim=True)

        # Calculate cosine similarities
        sim_matrix = t.mm(W_normalized.t(), W_normalized)

        cosine_sims.append(sim_matrix)

    # Plot cosine similarity changes over time for each of the first three features
    for feature in range(min(exp_cfg.n_enhanced_features, n_features)):
        plt.figure(figsize=(12, 6))
        for j in range(n_features):
            if feature != j:
                similarities = [sim[feature, j].item() for sim in cosine_sims]
                plt.plot(steps, similarities, label=f'Feature {feature+1} vs Feature {j+1}')

        plt.axvline(x=initial_steps, color='r', linestyle='--', label='Distribution Change')
        plt.xlabel('Training Steps')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Cosine Similarities Between Feature {feature+1} and Other Features Over Time')
        plt.tight_layout()
        save_path = os.path.join(folder, suffix, f'cosine_similarities_feature_{feature+1}.png')
        ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def run_experiment(exp_cfg, seed, plot2d, folder="results", suffix="v0"):
    set_seed(seed)

    cfg = exp_cfg.get_model_config()

    # All same importance, not-super-low feature probabilities (all >10%)
    importance = 0.75 ** t.arange(1, cfg.n_features + 1)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (100 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg=cfg,
        device=device,
        importance=1.0,
        feature_probability=exp_cfg.get_pre_feature_probability(),
    )

    model_data_log = model.optimize(steps=exp_cfg.pre_steps, 
    log_freq=exp_cfg.pre_log_freq, 
    lr=exp_cfg.pre_lr,
    n_enhanced_feats = exp_cfg.n_enhanced_features
    )

    if plot2d:
        save_path = os.path.join(folder, suffix, 'pre_phase_video.gif')
        ensure_dir(save_path)
        plot_features_in_2d(
            t.stack(model_data_log["W"], dim=0),
            colors=model_data_log["colors"],
            title=model_data_log["titles"],
            save=save_path
        )

    ae_cfg = exp_cfg.get_ae_config()
    autoencoder = AutoEncoder(ae_cfg)

    sae_data_log = autoencoder.optimize(
        model=model,
        steps=exp_cfg.pre_steps,
        neuron_resample_window=2_500,
        dead_neuron_window=400,
        neuron_resample_scale=0.5,
        log_freq=exp_cfg.pre_log_freq,
    )

    if plot2d:
        plot_sae(sae_data_log, folder, suffix, save='pre_phase_SAE.gif')

    # Update the model's feature probabilities for post-phase
    model.feature_probability = exp_cfg.get_post_feature_probability().to(model.W.device)

    # Continue optimizing with updated frequencies
    continued_model_data_log = model.optimize(steps=exp_cfg.post_steps, 
    log_freq=exp_cfg.post_log_freq, 
    lr=exp_cfg.post_lr,
    n_enhanced_feats = exp_cfg.n_enhanced_features
    )

    # Adjust titles to reflect continued training
    continued_model_data_log["titles"] = [f"Continued Step {step}" for step in range(exp_cfg.pre_steps,
     exp_cfg.pre_steps + exp_cfg.post_steps + exp_cfg.post_log_freq, exp_cfg.post_log_freq)]

    if plot2d:
        save_path = os.path.join(folder, suffix, 'post_phase_video.gif')
        ensure_dir(save_path)
        plot_features_in_2d(
            t.stack(continued_model_data_log["W"], dim=0),
            colors=continued_model_data_log["colors"],
            title=continued_model_data_log["titles"],
            save=save_path
        )

    # Plot feature dynamics
    plot_feature_dynamics(model_data_log, continued_model_data_log, exp_cfg, folder, suffix)

    # Plot SAE loss during fine-tuning
    plot_sae_loss(model, autoencoder, continued_model_data_log, exp_cfg, folder, suffix)

    # Plot cosine similarities
    plot_cosine_similarities(model_data_log, continued_model_data_log, exp_cfg, folder, suffix)

    return model_data_log, continued_model_data_log, sae_data_log
    
# %%

if MAIN:
    exp_cfg = ExperimentConfig(
        n_features=8,  
        pre_steps=30_000,  
        post_steps=30_000,  
        pre_log_freq=200,  
        post_log_freq=200,  
        n_correlated=8
    )

    # Run the experiment
    model_data_log, continued_model_data_log, sae_data_log = run_experiment(exp_cfg, 100, 
    folder='plots', 
    suffix='default', 
    plot2d=True)

# %%

if MAIN:
    # Create an instance of ExperimentConfig with default or custom values
    exp_cfg_2 = ExperimentConfig(
        n_features=5,  
        pre_steps=20_000,  
        post_steps=20_000,  
        pre_log_freq=200,  
        post_log_freq=200,  
        n_correlated=5,
        ae_n_hidden=5,
    )

    # Run the experiment
    model_data_log_2, continued_model_data_log_2, sae_data_log_2 = run_experiment(exp_cfg_2, 100, 
    folder='plots',     
    suffix='low_dim', 
    plot2d=True)


# %%

if MAIN:
    # Create an instance of ExperimentConfig with default or custom values
    exp_cfg_3 = ExperimentConfig(
        n_features=40,  
        pre_steps=10_000,  
        post_steps=10_000,  
        pre_log_freq=200,  
        post_log_freq=200,  
        n_correlated=40,
        n_hidden=10,
        ae_n_hidden=40,
        ae_n_input=10,
    )

    folder='plots'
    suffix='high_dim'

    # Run the experiment
    # model_data_log_3, continued_model_data_log_3, sae_data_log_3, model, autoencoder = run_experiment(exp_cfg_3, 100, 
    # folder='plots',     
    # suffix='high_dim', 
    # plot2d=False)

        # Plot feature dynamics
    plot_feature_dynamics(model_data_log_3, continued_model_data_log_3, exp_cfg_3, folder, suffix)

    # Plot SAE loss during fine-tuning
    # plot_sae_loss(model, autoencoder, continued_model_data_log, exp_cfg_3, folder, suffix)

    # Plot cosine similarities
    plot_cosine_similarities(model_data_log_3, continued_model_data_log_3, exp_cfg_3, folder, suffix)
