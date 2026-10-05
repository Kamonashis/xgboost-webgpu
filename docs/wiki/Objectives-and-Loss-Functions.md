# Objectives & Loss Functions

`xgboost-webgpu` supports an extensive suite of loss functions and objective formulations matching standard XGBoost. In second-order gradient boosting, each objective provides the first-order gradient ($g_i$) and positive second-order Hessian ($h_i$) with respect to the raw margin output $\hat{y}_i$:

$$g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$$

---

## 1. Regression: Squared Error (`reg:squarederror`)

Standard regression with mean squared error loss:
$$\mathcal{L}(y_i, \hat{y}_i) = \frac{1}{2} (\hat{y}_i - y_i)^2$$

* **First-order gradient**: $g_i = \hat{y}_i - y_i$
* **Second-order Hessian**: $h_i = 1.0$
* **Prediction Transform**: Identity ($\hat{y}_i$)

---

## 2. Binary Classification: Logistic Loss (`binary:logistic`)

Logistic regression for binary targets $y_i \in \{0, 1\}$:
$$\mathcal{L}(y_i, \hat{y}_i) = - \left[ w_i y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$
where $p_i = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}$.

* **Class Weighting (`scale_pos_weight`)**: For imbalanced classes, positive instances ($y_i = 1$) are weighted by $w_i = w_{\text{pos}}$, and negative instances by $w_i = 1$.
* **First-order gradient**:
  $$g_i = p_i (1 + y_i (w_{\text{pos}} - 1)) - w_{\text{pos}} y_i$$
  *(When $w_{\text{pos}} = 1$, simplifies to $g_i = p_i - y_i$)*
* **Second-order Hessian**:
  $$h_i = p_i (1 - p_i) (1 + y_i (w_{\text{pos}} - 1))$$
* **Prediction Transform**: Sigmoid $\sigma(\hat{y}_i) \in [0, 1]$

---

## 3. Multi-Class Classification (`multi:softprob`, `multi:softmax`)

Multinomial logistic regression for $K$ discrete classes $y_i \in \{0, 1, \dots, K-1\}$. For each sample, the model maintains $K$ separate margins $\hat{\mathbf{y}}_i = [\hat{y}_{i, 0}, \dots, \hat{y}_{i, K-1}]^T$:

$$p_{i, k} = \frac{e^{\hat{y}_{i, k}}}{\sum_{j=0}^{K-1} e^{\hat{y}_{i, j}}}$$

$$\mathcal{L}(y_i, \hat{\mathbf{y}}_i) = - \sum_{k=0}^{K-1} \mathbb{I}(y_i = k) \log(p_{i, k})$$

* **First-order gradient**:
  $$g_{i, k} = p_{i, k} - \mathbb{I}(y_i = k)$$
* **Second-order Hessian**:
  $$h_{i, k} = 2 \cdot p_{i, k} (1 - p_{i, k})$$
* **Prediction Transform**:
  * `multi:softprob`: Full probability vector $[p_{i, 0}, \dots, p_{i, K-1}]$
  * `multi:softmax`: Most probable class index $\arg\max_k p_{i, k}$

---

## 4. Count Data: Poisson Regression (`count:poisson`)

Models count or rate data where targets $y_i \ge 0$. The raw margin is linked via an exponential transform $\lambda_i = e^{\hat{y}_i}$:

$$\mathcal{L}(y_i, \hat{y}_i) = \lambda_i - y_i \hat{y}_i = e^{\hat{y}_i} - y_i \hat{y}_i$$

* **First-order gradient**: $g_i = e^{\hat{y}_i} - y_i$
* **Second-order Hessian**: $h_i = e^{\hat{y}_i}$
* **Prediction Transform**: Exponential $\hat{\lambda}_i = e^{\hat{y}_i}$

---

## 5. Dispersion Models: Gamma Regression (`reg:gamma`)

Models strictly positive continuous variables with constant coefficient of variation (e.g. insurance claim amounts, wait times):

$$\mathcal{L}(y_i, \hat{y}_i) = \frac{y_i}{e^{\hat{y}_i}} + \hat{y}_i$$

* **First-order gradient**: $g_i = 1 - \frac{y_i}{e^{\hat{y}_i}}$
* **Second-order Hessian**: $h_i = \frac{y_i}{e^{\hat{y}_i}}$
* **Prediction Transform**: Exponential $e^{\hat{y}_i}$

---

## 6. Compound Poisson-Gamma: Tweedie Regression (`reg:tweedie`)

Tweedie distribution with variance power $1 < \rho < 2$, ideal for zero-inflated continuous positive data (e.g. insurance loss costs):

$$\text{Var}(Y) = \mu^\rho, \quad \mu = e^{\hat{y}}$$

$$\mathcal{L}(y_i, \hat{y}_i) = - y_i \frac{e^{(1-\rho)\hat{y}_i}}{1 - \rho} + \frac{e^{(2-\rho)\hat{y}_i}}{2 - \rho}$$

* **First-order gradient**: $g_i = - y_i e^{(1-\rho)\hat{y}_i} + e^{(2-\rho)\hat{y}_i}$
* **Second-order Hessian**: $h_i = - (1-\rho) y_i e^{(1-\rho)\hat{y}_i} + (2-\rho) e^{(2-\rho)\hat{y}_i}$
* **Prediction Transform**: Exponential $e^{\hat{y}_i}$

---

## 7. Quantile Loss: Pinball Regression (`reg:quantileerror`)

Predicts conditional quantiles $\tau \in (0, 1)$ (e.g. median for $\tau = 0.5$, 90th percentile for $\tau = 0.9$):

$$\mathcal{L}_\tau(y_i, \hat{y}_i) = \max\left( \tau (y_i - \hat{y}_i), (\tau - 1)(y_i - \hat{y}_i) \right)$$

* **First-order gradient**:
  $$g_i = \begin{cases} -\tau & \text{if } y_i > \hat{y}_i \\ 1 - \tau & \text{if } y_i \le \hat{y}_i \end{cases}$$
* **Second-order Hessian**: Constant regularized proxy $h_i = 1.0$
* **Prediction Transform**: Identity ($\hat{y}_i$)

---

## 8. Learning-to-Rank: Pairwise Loss (`rank:pairwise`)

LambdaMART-style pairwise ranking using query group boundaries:
$$\mathcal{L}(\hat{\mathbf{y}}) = \sum_{q} \sum_{i, j \in \text{Group}_q, y_i > y_j} \log_2 \left( 1 + e^{-(\hat{y}_i - \hat{y}_j)} \right)$$

* For pairs within the same query group where $y_i > y_j$:
  $$\rho_{ij} = \frac{1}{1 + e^{\hat{y}_i - \hat{y}_j}}$$
  Accumulates gradients $g_i \mathrel{+}= -\rho_{ij}$ and $g_j \mathrel{+}= \rho_{ij}$.
* Hessians: $h_i \mathrel{+}= \rho_{ij}(1 - \rho_{ij})$ and $h_j \mathrel{+}= \rho_{ij}(1 - \rho_{ij})$.
