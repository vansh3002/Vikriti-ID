# Vikriti-ID: Synthetic Fingerprint Generator

## Overview

**Vikriti-ID** is a synthetic fingerprint generator developed to address the challenges in fingerprint recognition research due to the limited availability of extensive and publicly available fingerprint databases. Existing databases often lack a sufficient number of identities and fingerprint impressions, which hinders progress in areas such as fingerprint-based access control.

### Key Features:
- **Large Database**: Vikriti-ID generated a database containing **500,000 unique fingerprints**, each with **10 associated impressions**.
- **Performance Metrics**: The generated data achieved an **Equal Error Rate (EER) of 0.16%** and an **Area Under the Curve (AUC) of 0.89%**.
- **Deep Learning Integration**: A deep neural network, inspired by [13], was trained on both Vikriti-ID generated data as well as publicly available data, demonstrating the usability and effectiveness of the synthetic data.

## Project Structure
```
VIKRIT-ID/
├── Code/
│ ├── Loss/
│ │ └── perceptual_loss.py
│ ├── models/
│ │ ├── composite_model.py
│ │ ├── discriminator.py
│ │ └── generator.py
│ ├── train_dataset_VAE.py
│ ├── train.py
│ ├── utils.py
│ └── VAE.ipynb
├── AtoB_generated_plot_000001.png
├── decoder.h5
├── encoder.h5
├── processed_data.npz
└── requirements.txt
```


- **Loss/**: Contains the loss function implementations.
  - `perceptual_loss.py`: Perceptual loss function used in model training.
- **models/**: Contains model architecture files.
  - `composite_model.py`: Composite model combining different sub-models.
  - `discriminator.py`: Discriminator model implementation.
  - `generator.py`: Generator model implementation.
- **train_dataset_VAE.py**: Script for training the VAE on the dataset.
- **train.py**: Main training script.
- **utils.py**: Utility functions used across the project.
- **VAE.ipynb**: Jupyter notebook for training and experimenting with the VAE model.
- **AtoB_generated_plot_000001.png**: Example plot generated during training.
- **decoder.h5**: Pretrained decoder weights.
- **encoder.h5**: Pretrained encoder weights.
- **processed_data.npz**: Processed dataset used for training.
- **requirements.txt**: Python dependencies for the project.

## Citation

If you use Vikriti-ID in your research, please cite our paper:

[Link to the paper](https://openaccess.thecvf.com/content/WACV2024/papers/Shukla_Vikriti-ID_A_Novel_Approach_for_Real_Looking_Fingerprint_Data-Set_Generation_WACV_2024_paper.pdf)

## Contact

For any inquiries, please reach out to [Vansh Singh(vansh3002singh@gmail.com)] and [Aditya Sinha(adityasinha6078@gmail.com)].

