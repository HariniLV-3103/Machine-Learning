# Assignment 5 - Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning  

## 🎯 Aim
To implement and compare the performance of:
- **Model A:** Single-Layer Perceptron Learning Algorithm (PLA).  
- **Model B:** Multilayer Perceptron (MLP) with hidden layers and nonlinear activations.  

---

## 🛠️ Libraries Used
- **NumPy** – Numerical computations and array operations  
- **Pandas** – Dataset handling and preprocessing  
- **Matplotlib** – Data visualization  
- **Seaborn** – Advanced visualization and heatmaps  
- **Scikit-learn** – Preprocessing, evaluation metrics, model utilities  
- **TensorFlow/Keras** – Building and training MLP models  
- **Pillow** – Image loading and preprocessing  

---

## 🎯 Objectives
- Implement PLA from scratch with step activation.  
- Build and train an MLP with hyperparameter tuning.  
- Compare both models on the **English Handwritten Characters dataset**.  

---

## 📖 Theoretical Background
### 🔹 Perceptron Learning Algorithm (PLA)
- Weight update rule:  
  \[
  w^{t+1} = w^t + \eta (y - \hat{y}) x
  \]  
- Works only for linearly separable datasets.  

### 🔹 Multilayer Perceptron (MLP)
- Architecture: Input → Hidden Layers → Output  
- Uses nonlinear activations (ReLU, Sigmoid, Tanh)  
- Loss function: **Cross-Entropy** for classification  
- Optimizers: **SGD, Adam**  
- Learns nonlinear decision boundaries using **backpropagation**  

---

## 📂 Dataset
- **Dataset:** English Handwritten Characters Dataset  
- **Samples:** 3,410 images  
- **Classes:** 62 (0–9, A–Z, a–z)  
- **Preprocessing:** Resize (32×32), flatten, normalize pixel values  

---

## 📝 Implementation Steps
1. Preprocess dataset (resize, flatten, normalize).  
2. Implement **PLA** from scratch.  
3. Implement **MLP** with multiple configurations.  
4. Perform hyperparameter tuning (activation, optimizer, LR, batch size).  
5. Evaluate using Accuracy, Precision, Recall, F1, Confusion Matrix, ROC curves.  

---

## ⚙️ Hyperparameters
- **PLA:**  
  - Step activation  
  - Learning rate = 0.01  
  - Epochs = 30  

- **MLP:**  
  - 2 Hidden Layers (512, 256 neurons)  
  - Activation: ReLU (hidden), Softmax (output)  
  - Loss: Categorical Cross-Entropy  
  - Optimizer: Adam  
  - Learning Rate = 0.001  
  - Batch Size = 32  
  - Epochs = 25  

---

## 📊 Results
### 🔹 Perceptron (PLA)
- Test Accuracy: **17.7%**  
- Precision: 0.2708  
- Recall: 0.1774  
- F1-score: 0.1576  

### 🔹 Multilayer Perceptron (MLP - best config: ReLU + Adam, lr=0.001, batch=32)
- Test Accuracy: **29.8%**  
- Precision: 0.3207  
- Recall: 0.2977  
- F1-score: 0.2752  

---

## 📉 Comparison
- PLA underperformed due to its **linear separability limitation**.  
- MLP outperformed PLA by learning **nonlinear decision boundaries**.  
- **Adam + ReLU** gave the best performance.  
- Batch size 32 generalized better than 64.  
- More hidden layers helped initially, but too many caused **diminishing returns**.  

---

## 🔎 Observations
- PLA failed for nonlinear data → Accuracy only **17.7%**.  
- MLP achieved **29.8% accuracy**, showing better representational capacity.  
- **Optimizer choice** mattered: Adam >> SGD.  
- **Learning rate 0.001** stabilized training, while **0.01 diverged**.  
- No strong overfitting detected, but deeper models may need dropout/regularization.  

---

## ✅ Final Summary
| Model | Epochs | LR | Test Accuracy | Precision | Recall | F1-score |
|-------|--------|----|---------------|-----------|--------|----------|
| **PLA** | 30 | 0.01 | 0.1774 | 0.2708 | 0.1774 | 0.1576 |
| **MLP (best)** | 25 | 0.001 | 0.2977 | 0.3207 | 0.2977 | 0.2752 |

---

## 📌 Conclusion
- PLA is insufficient for complex datasets with nonlinear class boundaries.  
- MLP demonstrates significant improvements, but requires **careful hyperparameter tuning**.  
- For further improvement: add dropout, batch normalization, or increase training epochs.  
