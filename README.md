# 🌩️ CloudRiskNet: Detecting High-Risk Weather for Power Grids

> A deep learning model to classify weather images into **High-Risk** vs **Low-Risk** categories for electrical infrastructure resilience.

![Model Accuracy](https://via.placeholder.com/400x200?text=Training+Curves+Here)  
*(Replace with actual plot when publishing)*

## 🎯 Objective

Power grids are vulnerable to extreme weather events. This project uses **computer vision and transfer learning** to automatically detect high-risk atmospheric conditions from sky/cloud images — enabling early warnings for grid operators.

## 🔍 Risk Categorization Rationale

We grouped weather phenomena based on their potential impact on power grid infrastructure:

🔴 **High-Risk (5,081 samples)**: Events that can cause mechanical stress, outages, or equipment damage:  
- **rime, frost, glaze** → ice accumulation on power lines  
- **snow** → weight-induced line breakage  
- **hail, lightning, rain** → direct physical damage  
- **sandstorm** → abrasion and short circuits  

🟢 **Low-Risk (1,781 samples)**: Mostly visual or non-disruptive conditions:  
- **dew, rainbow** → minimal operational impact  
- **fogsmog** → included as low-risk but may need re-evaluation in high-humidity scenarios  

⚖️ **Note**: The 2.85:1 class imbalance reflects real-world monitoring priorities — safety-critical systems often emphasize **sensitivity over specificity**.

## 🧠 Model Architecture

- **Backbone**: EfficientNetB0 (pre-trained on ImageNet)  
- **Transfer Learning**: Only the final dense layer is trainable (1,281 parameters)  
- **Input Size**: 224×224 RGB images  
- **Output**: Binary probability (High Risk = 1, Low Risk = 0)

### Key Components:
- ✅ On-the-fly data augmentation (flip, rotation, zoom)  
- ✅ Global Average Pooling + Dropout (0.3) for regularization  
- ✅ Optimized TensorFlow data pipeline (`cache`, `prefetch`, `shuffle`)

## 📊 Results

After 10 epochs of training:

| Metric               | Validation Score |
|----------------------|------------------|
| **Accuracy**         | 93.6%            |
| **Precision**        | 89.4%            |
| **Recall**           | 85.9%            |

✅ **No overfitting** — training and validation curves converge smoothly  
✅ **High recall** ensures critical threats are rarely missed  
✅ **Stable loss** indicates robust learning

## ▶️ How to Run

1. Open the notebook in **Google Colab**:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link-to-your-colab)

2. The dataset is downloaded automatically via `kagglehub` (no API key needed):
   ```python
   import kagglehub
   path = kagglehub.dataset_download("jehanbhathena/weather-dataset")
