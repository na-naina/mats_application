# MATS Application

This repository contains the implementation and experiments for the MATS application.

## Main File

The main file for this application is `main.py`. Most of the experiments can be replicated using the following `ExperimentConfig` class:

```python
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
```

## Plots

The `plots` folder includes visualizations for the runs described in the associated Google Doc.

## Repository Structure

- `main.py`: Main application file
- `data.py`: Data handling and processing
- `plotly_utils.py`: Utilities for creating plots with Plotly
- `utils.py`: General utility functions
- `requirements.txt`: List of Python dependencies
- `plots/`: Folder containing generated plots

## Getting Started

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

For more detailed information about the experiments and their configurations, please refer to the associated Google Doc.
