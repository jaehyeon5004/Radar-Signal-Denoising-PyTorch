<div align="center">
  
  <h1>🛰️ Radar Signal Denoising using 1D-CNN</h1>
  
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white" />
    <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat-square&logo=WeightsAndBiases&logoColor=white" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
  </p>

  <p align="center">
    <strong>Efficient Noise Reduction for Radar Chirp Signals using Deep Learning</strong>
  </p>

  <div align="justify">
    This repository provides a deep learning-based solution for denoising radar chirp signals contaminated with additive noise. By utilizing a 1D Convolutional Neural Network (1D-CNN), this project demonstrates how to effectively reconstruct high-fidelity signals from noisy measurements, which is crucial for modern electronic warfare and radar systems.
  </div>
</div>

<hr />

## 🚀 Key Features

<table>
  <tr>
    <td><b>Complex Signal Processing</b></td>
    <td>Implemented a custom <code>RadarDataset</code> that splits complex-valued signals into <b>Real and Imaginary (I/Q) channels</b>, enabling the CNN to process phase information effectively.</td>
  </tr>
  <tr>
    <td><b>Encoder-Decoder Architecture</b></td>
    <td>A deep 1D-CNN architecture designed to extract latent features and reconstruct the original signal while suppressing noise components.</td>
  </tr>
  <tr>
    <td><b>Quantitative Evaluation</b></td>
    <td>Performance is measured using <b>PSNR (Peak Signal-to-Noise Ratio)</b> to ensure high-quality signal reconstruction.</td>
  </tr>
  <tr>
    <td><b>MLOps Integration</b></td>
    <td>Fully integrated with <b>Weights & Biases (W&B)</b> for real-time experiment tracking, loss visualization, and hyperparameter tuning.</td>
  </tr>
  <tr>
    <td><b>Multi-domain Visualization</b></td>
    <td>Includes automated plotting for both <b>Time-domain waveforms</b> and <b>Frequency-domain Spectrograms</b>.</td>
  </tr>
</table>

<hr />

## 🏗️ Architecture & Methodology

### 1. Signal Processing Pipeline
<p align="left">
  <ul>
    <li><b>Preprocessing:</b> Load <code>.mat</code> files and convert raw complex data into a 2-channel 1D tensor $(2 \times 1024)$.</li>
    <li><b>Feature Extraction:</b> Two-stage Encoder using wide kernel sizes $(9, 5)$ to capture both local and global signal patterns.</li>
    <li><b>Reconstruction:</b> Symmetric Decoder to map features back to the signal space.</li>
  </ul>
</p>



### 2. Performance Metric
The model optimizes the Mean Squared Error (MSE) and is evaluated via $PSNR$:

<div align="center">
  
$$PSNR = 20 \cdot \log_{10} \left( \frac{MAX_{signal}}{\sqrt{MSE}} \right)$$

</div>

<hr />

## 📊 Results

### Denoising Performance
The model effectively removes the noise floor while preserving the **linear frequency modulation (LFM)** characteristics of the chirp signal.

| Category | Description |
| :--- | :--- |
| **Noisy Input** | Heavy interference obscuring the signal. |
| **AI Denoised** | Successfully recovered waveform with high PSNR. |
| **Clean Target** | The ground truth for comparison. |

<br />

> [!TIP]
> **Spectrogram analysis** confirms that the AI model significantly reduces the background noise floor across the entire frequency spectrum, maintaining the integrity of the original chirp slope.



<hr />

<div align="center">
  <p>Copyright © 2026 Jaehyeon Lee (382). All rights reserved.</p>
</div>
