

# Dimensionality Reduction and Classification – Hands On Overview

## Hands On Objective

This Hands On demonstrates the application of **dimensionality reduction techniques** to improve machine learning model performance. Dimensionality reduction reduces the number of features while preserving important information, addresses the **curse of dimensionality**, and enhances computational efficiency.

Techniques used:

1. **Principal Component Analysis (PCA)** – linear, unsupervised
2. **Linear Discriminant Analysis (LDA)** – linear, supervised
3. **Kernel PCA** – non-linear, unsupervised

---

## 1. Principal Component Analysis (PCA)

### Concept

* PCA transforms original features into a new set of **orthogonal components (principal components)**
* First components capture **maximum variance** in the data
* Reduces feature dimensionality while retaining most of the information

### Implementation

* Converted categorical data to numerical and applied **standardization**
* Created reusable functions for:

  * PCA transformation
  * Model training and evaluation
  * Accuracy calculation
* Explored **2–14 components** for a dataset with 27 features
* Models trained on PCA-transformed data:

  * Logistic Regression, KNN, SVM (Linear & Non-linear), Naive Bayes, Decision Tree, Random Forest

### Observations

* High accuracy achieved across all models:

  * Logistic Regression, KNN, Naive Bayes → ~98%
  * SVM and Random Forest → up to 100%
* Too few components (2–3) → loss of important information
* Too many components (13–14) → added noise
* **Optimal component range: 5–7** for model generalization

### Visualization

* Plotted **Accuracy vs Number of PCA Components** using Matplotlib
* Highlighted top-performing models: SVM (Linear & Non-linear) and Decision Tree

---

## 2. Linear Discriminant Analysis (LDA)

### Concept

* LDA is a **supervised learning technique** that reduces dimensionality while maximizing **class separability**
* Unlike PCA (variance-based), LDA focuses on **distance between classes**
* Number of components restricted by:
  [
  n_{\text{components}} \le \min(n_{\text{features}}, n_{\text{classes}} - 1)
  ]

### Implementation

* Imported `LinearDiscriminantAnalysis` from `sklearn.discriminant_analysis`
* Fitted LDA with both `X_train` and `y_train` (supervised)
* Transformed `X_test` using the fitted model
* For the CKD dataset:

  * Features = 27
  * Classes = 2 (CKD / Non-CKD)
  * Maximum components = 1

### Observations

* Trained models on LDA-transformed data:

  * Logistic Regression → 95%
  * KNN → 96%
  * SVM (Linear & Non-linear) → 96%
  * Naive Bayes → 95%
  * Decision Tree → 96%
  * Random Forest → 96%
* Best-performing models: KNN, SVM, Decision Tree, Random Forest
* LDA efficiently reduced dimensionality **while preserving class separation**

---

## 3. Kernel PCA

### Concept

* Kernel PCA extends PCA to **non-linear relationships**
* Uses **kernel functions** (e.g., RBF, poly, sigmoid) to map data into higher-dimensional space
* Captures complex patterns that standard PCA may miss
* Unsupervised → only input features (`X_train`) are used

### Implementation

* Imported `KernelPCA` from `sklearn.decomposition`
* Applied **RBF kernel** and transformed `X_train` and `X_test`
* Explored **component range: 2–14** for the dataset
* Trained models:

  * Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest

### Observations

* Most models achieved similar accuracy
* Decision Tree performed best with **4–6 components** → 93% accuracy
* Optimal range avoids information loss (too few components) and noise (too many components)

---

## Conclusion

* **PCA** → effective for linear dimensionality reduction; optimal components: 5–7
* **LDA** → supervised, maximizes class separability; suitable for classification with limited components (1 in binary dataset)
* **Kernel PCA** → handles non-linear datasets; optimal components: 4–6 for this dataset
* Top-performing classifiers across experiments: **Decision Tree, Random Forest, SVM, KNN**
* Dimensionality reduction improved accuracy, reduced computational complexity, and addressed overfitting

--- 
