# ECHO — CNN-Transformer Audio Detection Model

This document outlines the design, feature engineering, and performance details of the upgraded **CNN-Transformer** sound event classifier.

---

## 1. Feature Preprocessing (Spatial Derivatives)

To improve detection in noisy bus and urban environments, the raw audio log-mel spectrogram is augmented with spatial edge detection filters, treating the spectrogram as a 2D time-frequency image.

### Augmented Input Spectrogram ($\hat{X}$)
1. **Mel Spectrogram ($X$)**: Standard spectrogram computed with $1024$ FFT bins, $512$ hop length, and $64$ Mel bins.
2. **First-order Sobel Derivative ($X'$)**: Computed using horizontal ($H_u$) and vertical ($H_v$) Sobel kernels:
   $$H_u = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}, \quad H_v = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$
   $$X' = \sqrt{(X * H_u)^2 + (X * H_v)^2 + \epsilon}$$
3. **Second-order Laplacian Derivative ($X''$)**: Computed using a Laplacian kernel:
   $$H_l = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
   $$X'' = X * H_l$$

These features are concatenated along the frequency axis to yield an augmented spectrogram $\hat{X}$ of shape:
$$\text{Shape} = (M \text{ time frames} \times 192 \text{ frequency bins})$$

---

## 2. CNN-Transformer Architecture

The model `EchoTransformer` consists of a 2D CNN downsampler followed by a Multi-Head Self-Attention Transformer Encoder:

```
[Input: 1x192xT] ➔ [ConvBlock 16] ➔ [ConvBlock 32] ➔ [ConvBlock 64] ➔ [Projector] ➔ [Positional Enc] ➔ [Transformer Encoder] ➔ [Pooling] ➔ [Dense Output]
```

* **CNN Extractor**: 3 layers of double 2D convolutions downsample the frequency axis by $8\times$ (from $192$ down to $24$).
* **Linear Projector**: Maps the output $64 \text{ channels} \times 24 \text{ frequency bins} = 1536$ features to a $128$-dimensional temporal token sequence.
* **Transformer Encoder**: 2 layers of multi-head self-attention ($nhead=4$, feed-forward dimension $512$) process the sequence in parallel to capture global temporal context.
* **Global Pooling**: Averages frame-level embeddings across the time sequence to generate clip-level predictions.

---

## 3. Training & Validation Performance

The model was trained on the compiled metadata dataset containing **1,170 records** mapped to 5 active classes:

* **Training Duration**: 15 epochs per iteration.
* **Optimal Learning Rate**: `0.0005` (Iteration 2).
* **Validation Accuracy**: **`94.29%`**
* **Test Accuracy**: **`96.59%`**
* **Test F1 Score**: **`93.58%`** (Target Quality Gate: `90.00%`).

---

## 4. Pipeline Logic: Verification Bypass

To ensure safety in real-world environments, the detection logic treats continuous and transient sounds differently:

* **Continuous Threat (Siren, Alarm, Shouting)**: Runs Pass 1 (2-second sliding window). If triggered, requests Pass 2 (5-second block) to verify continuous wailing/screaming.
* **Transient Threat (Gunshot, Explosion, Glass Breaking)**: Runs Pass 1. If a transient candidate is detected, the verification pass is **immediately bypassed** to trigger the emergency alert instantly.
