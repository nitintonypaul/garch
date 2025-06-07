# GARCH(1,1) Volatility Estimation with Gradient Ascent

This documentation outlines the development of a GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity) model, built entirely from the ground up. This project demonstrates a comprehensive understanding of financial time series modeling, numerical optimization, and the integration of Python with high-performance C++ code using pybind11.

---

## 1. Project Overview

This project presents a C++ implementation of the **Generalized Autoregressive Conditional Heteroskedasticity (GARCH) (1,1) model** for estimating financial market volatility. Utilizing a **Maximum Likelihood Estimation (MLE)** approach, the model parameters are optimized through an iterative **Gradient Ascent** algorithm. The core functionality is encapsulated within a C++ script, with seamless integration provided by `pybind11`, allowing the volatility estimation function to be called and utilized from Python. This project demonstrates a strong understanding of quantitative financial modeling, numerical optimization techniques, and C++ programming with Python interoperability.

---
## 2. Introduction to Volatility and GARCH Models

### What is Volatility?

In finance, **volatility** is a crucial measure of the dispersion of returns for a given security or market index. It quantifies the degree of variation of a trading price series over time, often expressed as the standard deviation of returns. High volatility implies greater risk, as prices can fluctuate drastically, while low volatility suggests more stable prices. Accurate volatility forecasting is essential for:
* **Risk Management:** Quantifying potential losses in portfolios.
* **Option Pricing:** Volatility is a key input in models like Black-Scholes.
* **Portfolio Optimization:** Allocating assets based on their risk-return profiles.

Traditional methods of measuring volatility, such as calculating the historical standard deviation over a fixed window, often fall short because financial market volatility is not constant over time. It exhibits phenomena like **volatility clustering**, where large price changes tend to be followed by large price changes (of either sign), and small changes tend to be followed by small changes. This suggests that volatility itself is time-varying.

### Introduction to GARCH(1,1)

To address the time-varying nature of financial volatility, **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** models were introduced. These models capture the observation that the conditional variance (the variance of future returns, conditional on past information) changes over time.

The **GARCH(1,1) model** is one of the most widely used specifications, representing the conditional variance ($\sigma_t^2$) as a function of past squared residuals (shocks) and past conditional variances. The core equation for the GARCH(1,1) model is:

$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

Where:
* $\sigma_t^2$: The conditional variance at time $t$. This is the variance of the returns for the upcoming period, estimated using information available up to time $t-1$.
* $\epsilon_{t-1}^2$: The squared return (or "shock") at time $t-1$. This term captures the impact of past market movements on current volatility.
* $\sigma_{t-1}^2$: The conditional variance at time $t-1$. This term accounts for the persistence of volatility, meaning that high (or low) volatility tends to be followed by high (or low) volatility.
* $\omega$: A constant term, representing the long-run average variance or the unconditional variance component. It ensures that the conditional variance is always positive.
* $\alpha$: The ARCH coefficient, which measures the extent to which volatility reacts to market shocks ($\epsilon_{t-1}^2$). A higher $\alpha$ means a greater impact of past squared returns on current variance.
* $\beta$: The GARCH coefficient, which measures the persistence of volatility, i.e., how much previous conditional variance influences the current conditional variance. A higher $\beta$ indicates that volatility takes longer to revert to its long-run mean.

### Constraints on Parameters

For the GARCH(1,1) model to be meaningful and stable (stationarity), the following constraints are typically imposed on its parameters:
* $\omega > 0$: The constant term must be positive to ensure that the long-run variance is positive.
* $\alpha \ge 0$: The coefficient for past shocks must be non-negative.
* $\beta \ge 0$: The coefficient for past variance must be non-negative.
* $\alpha + \beta < 1$: This crucial constraint ensures the **stationarity** of the GARCH process, meaning that the impact of past shocks eventually decays, and volatility reverts to a finite long-run average. If $\alpha + \beta \ge 1$, volatility could explode to infinity.

---

## 3. Parameter Estimation: Maximum Likelihood and Gradient Ascent

### The Challenge of Parameter Estimation

The parameters of the GARCH(1,1) model ($\omega, \alpha, \beta$) are not directly observable. Instead, they must be estimated from historical data (e.g., a time series of financial returns). The goal of estimation is to find the set of parameters that best explains the observed data.

### Maximum Likelihood Estimation (MLE)

**Maximum Likelihood Estimation (MLE)** is a powerful statistical method for estimating the parameters of a probability distribution. The principle behind MLE is to find the parameter values that maximize the likelihood of observing the given sample data. In the context of GARCH, we seek parameters that maximize the probability of the observed time series of returns, given the GARCH model.

Assuming that the financial shocks (or standardized residuals) $\epsilon_t$ are independently and identically distributed (i.i.d.) and follow a normal distribution with mean zero and conditional variance $\sigma_t^2$, i.e., $\epsilon_t \sim N(0, \sigma_t^2)$, the probability density function (PDF) for a single observation $\epsilon_t$ is:

$$
f(\epsilon_t; \omega, \alpha, \beta, \sigma_{t-1}^2) = \frac{1}{\sqrt{2\pi\sigma_t^2}} \exp\left(-\frac{\epsilon_t^2}{2\sigma_t^2}\right)
$$

For a series of $T$ observations, the **likelihood function** $L$ is the product of the individual probability densities:

$$
L(\omega, \alpha, \beta | \epsilon_1, \dots, \epsilon_T) = \prod_{t=1}^{T} \frac{1}{\sqrt{2\pi\sigma_t^2}} \exp\left(-\frac{\epsilon_t^2}{2\sigma_t^2}\right)
$$

To simplify calculations and avoid numerical underflow (due to multiplying many small probabilities), it is common practice to maximize the **log-likelihood function** instead. Since the logarithm is a monotonically increasing function, maximizing the log-likelihood is equivalent to maximizing the likelihood. Taking the natural logarithm of the likelihood function yields:

$$
\text{log}L(\omega, \alpha, \beta) = \sum_{t=1}^{T} \left[ -\frac{1}{2} \log(2\pi) - \frac{1}{2} \log(\sigma_t^2) - \frac{\epsilon_t^2}{2\sigma_t^2} \right]
$$

This is the function that the `L` function in the C++ code aims to compute and optimize.

### Gradient Ascent for Optimization

Direct analytical solutions for maximizing the log-likelihood function for GARCH models are generally not feasible due to the complex, non-linear relationship between the parameters and the likelihood. Therefore, iterative numerical optimization algorithms are employed.

**Gradient Ascent** is an iterative optimization algorithm used to find the maximum of a function. It works by taking steps proportional to the positive gradient of the function at the current point. The magnitude of the step is determined by the learning rate. For our GARCH parameter estimation, the update rule for a parameter $\theta$ (which can be $\omega$, $\alpha$, or $\beta$) at iteration $k+1$ is:

$$
\theta_{k+1} = \theta_k + \text{learning\_rate} \times \frac{\partial \text{log}L}{\partial \theta_k}
$$

Where $\frac{\partial \text{log}L}{\partial \theta_k}$ is the partial derivative (gradient) of the log-likelihood function with respect to the parameter $\theta$ at the current values. These partial derivatives indicate the direction of the steepest ascent on the log-likelihood surface.

In this implementation, the partial derivatives are approximated numerically using a small step size:

$$
\frac{\partial \text{log}L}{\partial \theta} \approx \frac{\text{log}L(\theta + \text{step}) - \text{log}L(\theta)}{\text{step}}
$$

The optimization process iteratively adjusts $\omega$, $\alpha$, and $\beta$ in the direction that increases the log-likelihood, subject to the necessary constraints, until convergence is reached or a maximum number of iterations is performed.

The **learning rate** controls the size of the steps taken during optimization. A small learning rate leads to slow convergence but can be more stable, while a large learning rate can lead to faster convergence but risks overshooting the maximum or oscillating. The **step size** used in approximating the partial derivatives influences the accuracy of the gradient calculation.

Crucially, throughout the optimization, the parameters are constrained to satisfy the conditions for a valid GARCH(1,1) model: $\omega > 0, \alpha \ge 0, \beta \ge 0$, and $\alpha + \beta < 1$. These constraints are handled within the `estimate_volatility` function's optimization loop.

---
## 4. C++ Implementation Details

The GARCH(1,1) volatility estimation is implemented through two core C++ functions: `estimate_volatility` for optimization and `L` for log-likelihood calculation. `Pybind11` integrates this C++ functionality with Python.

```cpp
// Common headers and Pybind11 setup
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Function declarations and global variables
double estimate_volatility(int no_of_days, double past_vol, const std::vector<double> &shock_array);
double L(double a, double b, double w, const std::vector<double> &arr);

int days;
double vol;
double expected_volatility;
```

### 4.1. `L` (Log-Likelihood) Function

This function calculates the log-likelihood of the observed `shock_array` for given GARCH parameters ($\alpha, \beta, \omega$). It directly implements the GARCH(1,1) equation and the log-likelihood formula.

```cpp
// Log-Likelihood Function for GARCH(1,1) Model
double L(double a, double b, double w, const std::vector<double> &arr) {
    double prev_variance = vol * vol;
    double total_likelihood = 0.000;
    double tiny_eps = 0.0000000001; // For numerical stability

    for (int i = 1; i < days; i++) {
        // GARCH(1,1) conditional variance equation: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        double sig_t_square = w + (a * std::pow(arr[i-1], 2)) + (b * prev_variance);
        total_likelihood += -0.5 * (std::log(2 * M_PI) + std::log(sig_t_square + tiny_eps) + (std::pow(arr[i], 2) / sig_t_square));
        prev_variance = sig_t_square;
    }
    double variance = w + (a * std::pow(arr[days-1], 2)) + (b * prev_variance);
    expected_volatility = std::sqrt(variance);
    return total_likelihood / days; // Return average for smoother optimization
}
```

**Key Points:**
* Implements the **GARCH(1,1) conditional variance equation** and the log-likelihood summation.
* Uses `tiny_eps` to prevent `log(0)` for numerical stability.
* Calculates the next period's expected volatility and updates a global variable.

### 4.2. `estimate_volatility` Function

This function performs **Gradient Ascent** to find the optimal GARCH parameters ($\omega, \alpha, \beta$) that maximize the log-likelihood. It iteratively adjusts parameters while enforcing model constraints.

```cpp
// Volatility Estimation Function (Gradient Ascent Optimization)
double estimate_volatility(int no_of_days, double past_vol, const std::vector<double> &shock_array) {
    days = no_of_days;
    vol = past_vol;

    // Initial parameter guesses and optimization hyperparameters
    double w = 0.000001; double a = 0.1; double b = 0.8;
    double learning_rate = 0.001;
    double step = 1e-5;

    for (int i = 0; i < 1000; i++) {
        double likelihood = L(a, b, w, shock_array);
        double step_w = std::max(w * 1e-3, 1e-6); // Adaptive step for omega

        // Numerical approximation of gradients
        double da = (L(a + step, b, w, shock_array) - likelihood) / step;
        double db = (L(a, b + step, w, shock_array) - likelihood) / step;
        double dw = (L(a, b, w + step_w, shock_array) - likelihood) / step_w;

        // Parameter updates applying non-negativity constraints
        if (a + learning_rate * da >= 0) a += learning_rate * da;
        if (b + learning_rate * db >= 0) b += learning_rate * db;
        if (w + learning_rate * dw >= 0) w += learning_rate * dw;

        // Stationarity constraint: alpha + beta < 1
        if (a + b >= 1) {
            double scale = 0.99 / (a + b);
            a *= scale;
            b *= scale;
        }
        if (w > 0.0001) w = 0.0001; // Upper bound for omega

        if (abs(da) + abs(db) + abs(dw) < 1e-8) break; // Convergence check
    }
    L(a, b, w, shock_array); // Final calculation for expected_volatility
    return expected_volatility;
}
```

**Key Points:**
* Implements an iterative **Gradient Ascent** to maximize the log-likelihood.
* Numerically approximates gradients for `w`, `a`, `b`.
* Enforces critical GARCH parameter constraints: non-negativity ($\omega, \alpha, \beta \ge 0$) and **stationarity** ($\alpha + \beta < 1$).
* Includes practical safeguards like an upper bound for $\omega$ and a convergence check.

### 4.3. Pybind11 Integration

`pybind11` provides a clean interface to expose the C++ `estimate_volatility` function to Python, enabling easy scripting and data analysis.

```cpp
// Pybind11 Module Definition
PYBIND11_MODULE(garch_est, m) {
    m.def("estimate_vol", &estimate_volatility, "Estimates GARCH(1,1) volatility.");
}
```

**Key Points:**
* Defines the Python module `garch_est`.
* Exposes `estimate_volatility` as `estimate_vol` in Python, complete with a docstring.

---

## 5. Opportunities for Improvement

While this project successfully demonstrates the core mechanics of GARCH(1,1) volatility estimation, several enhancements could be considered for a production-grade system or a more comprehensive analysis:

* **Robust Optimization Algorithms:** While Gradient Ascent is effective for demonstration, more sophisticated optimization algorithms like BFGS (Broyden–Fletcher–Goldfarb–Shanno) or Nelder-Mead could offer faster convergence and better handling of complex likelihood surfaces.
* **Error Handling and Input Validation:** Implementing robust error handling (e.g., for empty `shock_array`, non-positive `past_vol`, or non-convergence) and comprehensive input validation would make the function more resilient and user-friendly.
* **Model Diagnostics:** Incorporating statistical tests (e.g., Ljung-Box test on standardized residuals, ARCH-LM test) to check for remaining autocorrelation or heteroskedasticity in the residuals would allow for proper model validation.
* **Alternative GARCH Specifications:** Extending the model to other GARCH family variants (e.g., GARCH(p,q), EGARCH, GJR-GARCH) could capture asymmetric responses to positive vs. negative shocks.
* **Performance Optimization:** For extremely large datasets, techniques like vectorization, parallel processing, or using specialized numerical libraries could significantly improve computation speed.
* **Configuration and Hyperparameters:** Externalizing hyperparameters (learning rate, step size, max iterations) instead of hardcoding them would allow for easier experimentation and tuning.
* **Comprehensive Testing:** Developing a suite of unit and integration tests would ensure the correctness and reliability of the implementation under various scenarios.

---

## Demo
This demo showcases the program's current capabilities, reflecting its state as of May 2025. All reported volatility values are non-annualized GARCH outputs, expressed in percentages (%). Please note these may differ from Implied Volatility (IV) values.

```
#Test GME (GameStop)

Enter a stock: GME
==============================================================
Stock chosen for analysis: GME
Volatility before 1 year from today: 6.790427866155607%
Volatility predicted for tomorrow: 5.565664774067116%
==============================================================
```

```
#Test RELIANCE.NS (Reliance Industries)

Enter a stock: RELIANCE.NS
==============================================================
Stock chosen for analysis: RELIANCE.NS
Volatility before 1 year from today: 0.9338607456029148%
Volatility predicted for tomorrow: 1.4546634867429722%
==============================================================
```

```
#Test NVDA (Nvidia)

Enter a stock: NVDA
==============================================================
Stock chosen for analysis: NVDA
Volatility before 1 year from today: 1.907780008258419%
Volatility predicted for tomorrow: 2.869704427047115%
==============================================================
```
