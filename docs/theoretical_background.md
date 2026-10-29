# Theoretical background

## Activity recognition as pattern classification

A window is represented by a feature vector or multichannel signal and assigned to one of six activity classes. A classifier estimates a mapping \(f(x)\to y\). The evaluation emphasizes macro-F1 so that each class has equal contribution even when class supports differ.

## Discrete signal summaries

For a window \(x_1,\ldots,x_N\), the mean is \(\bar{x}=N^{-1}\sum_t x_t\), variance is \(N^{-1}\sum_t(x_t-\bar{x})^2\), RMS is \(\sqrt{N^{-1}\sum_t x_t^2}\), and mean absolute value is \(N^{-1}\sum_t|x_t|\). The interquartile range is \(Q_{.75}-Q_{.25}\); energy per sample is \(N^{-1}\sum_t x_t^2\). Skewness and excess kurtosis summarize distributional asymmetry and tail weight. Zero-crossing rate counts sign changes in adjacent samples and divides by \(N-1\). Cross-axis Pearson correlation is covariance normalized by the product of axis standard deviations.

The discrete Fourier transform is \(X_k=\sum_{n=0}^{N-1}x_n e^{-j2\pi kn/N}\). The implementation uses the real FFT and sampling frequency 50 Hz. Power weights are \(P_k=|X_k/N|^2\); the spectral centroid is \(\sum_k f_kP_k/\sum_kP_k\); normalized spectral entropy is calculated from \(p_k=P_k/\sum_jP_j\) as \(-\sum_k p_k\log_2(p_k)\). Dominant frequency is the non-DC bin of maximum power. Band energy sums spectral power in the specified frequency interval. These summaries compress a window and cannot preserve all temporal ordering.

## Standardization, SVM and kernels

A standard score is \(z_{ij}=(x_{ij}-\mu_j)/s_j\), with mean and standard deviation estimated on training data only. Scaling matters because margin and distance-based SVM objectives are sensitive to feature magnitude. A binary soft-margin linear SVM minimizes \(\frac12\lVert w\rVert^2+C\sum_i\xi_i\) subject to \(y_i(w^Tx_i+b)\ge 1-\xi_i\) and \(\xi_i\ge0\). Support vectors are training observations that constrain the margin or lie within it. In multiclass use, scikit-learn combines binary decision functions. The RBF kernel \(K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)\) produces nonlinear boundaries. The penalty C controls the trade-off between a wider margin and training errors; gamma controls locality of RBF similarity.

## Random forests

A decision tree recursively partitions feature space to reduce impurity. A random forest fits trees to bootstrap resamples and considers a random subset of predictors at each split; class decisions are aggregated by voting. This reduces the variance of individual trees. Impurity importance sums split impurity reduction and can favor continuous or high-cardinality features and distribute credit among correlated predictors. Permutation importance instead measures score reduction after shuffling a feature on held-out observations; correlated predictors can mask each other.

## Mutual information, ANOVA and PCA

Mutual information \(I(X;Y)=\sum_{x,y}p(x,y)\log[p(x,y)/(p(x)p(y))]\) measures dependence, not merely linear association. The k-nearest-neighbor estimator used here is fitted only on the training data. The ANOVA F statistic compares between-class variance to within-class variance for each feature. Both methods rank original variables and retain a subset, supporting direct feature inspection.

PCA centers scaled data and finds orthogonal directions that maximize variance. If \(X=U\Sigma V^T\), the first k columns of V define the retained principal axes. Components mix original measurements and maximize total variance without using labels. Consequently PCA and supervised feature selection are not equivalent: high variance need not mean high class discrimination, whereas selecting named variables preserves feature identity. Explained variance is descriptive and computed from training data only.

## One-dimensional convolutional neural network

A temporal convolution computes \(h_{t,c}=\sigma(b_c+\sum_{\tau,j}W_{\tau,j,c}x_{t+\tau,j})\), sharing filters across time to detect local patterns. Batch normalization stabilizes intermediate activations; ReLU provides nonlinearity; max pooling reduces temporal resolution, and global average pooling aggregates the learned maps. A dense layer produces six logits, optimized with multiclass cross entropy \(-\log p(y_i\mid x_i)\) and Adam gradient updates. Dropout and weight decay regularize the model. Early stopping based on participant-held-out validation macro F1 limits overfitting and saves the best checkpoint. Occlusion importance is the decrease in the true-class log probability after masking a temporal/channel region; it is sensitive to the masking choice and is not a causal attribution.

## Metrics

Accuracy is the proportion correct. Per-class precision is \(TP/(TP+FP)\), recall is \(TP/(TP+FN)\), and F1 is their harmonic mean. Macro averages are unweighted across the six classes; weighted F1 weights each class by support. A confusion matrix has true classes in rows and predicted classes in columns. Group-aware cross-validation ensures that windows from a participant cannot occur in both the fit and validation fold, addressing identity-dependent correlation between overlapping windows.
