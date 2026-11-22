# MNIST Classifier Benchmark: 9 Methods, With and Without PCA

**A head-to-head comparison of nine classification methods on handwritten digits, and a study of how PCA helps distance-based models but hurts tree-based ones.**


---

## Overview

Every model is trained on the same stratified 80/20 split of 42,000 MNIST images (33,600 train / 8,400 test), with pixels scaled to [0, 1]. A shared evaluation helper records train and test accuracy, macro F1, over-fitting gap, and training and prediction time.

The project then retrains every method on the first *k* principal components, for k = 10, 20, 50, 100, 200 and 350, to see how compression changes each one.

**Methods compared:** Logistic Regression · Decision Tree · Bagging · Random Forest · Gradient Boosting (histogram) · XGBoost · SVM (RBF kernel) · SVM (polynomial kernel) · Neural network (PyTorch MLP, 784 → 256 → 128 → 10)

## Results

### Raw pixels (784 features)

| Rank | Model | Test accuracy | Macro F1 | Train time | Predict time |
|---:|---|---:|---:|---:|---:|
| 1 | SVM (RBF) | **97.92%** | 0.979 | 100 s | 48.0 s |
| 2 | Neural network (PyTorch) | 97.80% | 0.978 | 52 s | 0.07 s |
| 3 | XGBoost | 97.52% | 0.975 | 146 s | 0.09 s |
| 4 | Gradient Boosting (histogram) | 97.43% | 0.974 | 98 s | 1.06 s |
| 5 | SVM (polynomial) | 97.42% | 0.974 | 79 s | 18.3 s |
| 6 | Random Forest | 96.39% | 0.964 | 4.5 s | 0.10 s |
| 7 | Bagging | 94.63% | 0.946 | 90 s | 0.84 s |
| 8 | Logistic Regression | 91.36% | 0.912 | 6.0 s | 0.02 s |
| 9 | Decision Tree | 85.54% | 0.854 | 7.0 s | <0.01 s |

### With PCA (best k per model)

| Model | Raw pixels | Best k | Accuracy at best k | Change | Train time |
|---|---:|---:|---:|---:|---|
| **SVM (RBF)** | 97.92% | 50 | **98.39%** | +0.47 pp | 100 s → 5.4 s |
| SVM (polynomial) | 97.42% | 100 | 98.15% | +0.73 pp | 79 s → 14 s |
| Neural network | 97.80% | 50 | 98.17% | +0.37 pp | 52 s → 28 s |
| Logistic Regression | 91.36% | 200 | 91.45% | +0.09 pp | 6.0 s → 1.5 s |
| XGBoost | 97.52% | 50 | 96.44% | −1.08 pp | 146 s → 17 s |
| Gradient Boosting | 97.43% | 100 | 96.36% | −1.07 pp | 98 s → 37 s |
| Random Forest | 96.39% | 100 | 94.77% | −1.62 pp | 4.5 s → 15 s |
| Decision Tree | 85.54% | 20 | 83.64% | −1.90 pp | 7.0 s → 3.1 s |
| Bagging | 94.63% | 50 | 92.46% | −2.17 pp | 90 s → 43 s |

**Key findings**
- **The best model in the project is an RBF SVM on just 50 principal components: 98.39% accuracy, trained 19× faster than on raw pixels.**
- MNIST is highly compressible: 43 components (5.5% of pixels) keep 80% of the variance, and 87 keep 90%.
- PCA helps models built on distances and dot products (SVMs, neural network, logistic regression), because dropping low-variance components removes pixel noise.
- PCA hurts every tree-based model by 1–2 points, because each PCA feature mixes all 784 pixels, so a single split no longer corresponds to "is there ink here?".
- The top five models are within 0.5 points of each other, but prediction cost differs by about 700×: the SVM needs 48 s to classify the test set, the neural network 0.07 s.
- k = 50–100 (83–92% of variance) is the sweet spot for almost every model.

## Repository contents

| File | Description |
|---|---|
| `MNIST_Classifier_Comparison.ipynb` | The nine classifiers on raw pixels, with the shared evaluation helper |
| `MNIST_Classifier_PCA_Analysis.ipynb` | PCA analysis: explained variance, reconstructions, every model retrained at each k |
| `M3 Project Report – Classification Methods and PCA on MNIST.pdf` | Full report, including pros and cons of each method |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook M3_Code.ipynb
```

**Data:** download `train.csv` from Kaggle's [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data) competition, save it in this folder as `mnist_train.csv`, and point the `read_csv` path in the first code cell at it. The dataset is not included in this repo because of its size (77 MB).

Timings were measured on an 8-core laptop CPU and will vary between machines. Running every model at every k takes a while; the SVMs and XGBoost are the slowest.

## Tech stack

Python · scikit-learn · XGBoost · PyTorch · NumPy · pandas · seaborn · Matplotlib

---

**Author:** Varun Chander · [GitHub](https://github.com/varunchander1)
