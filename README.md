# GARCH(1,1) Volatility Estimation from Scratch
This documentation outlines the development of a GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity) model, built entirely from the ground up. This project demonstrates a comprehensive understanding of financial time series modeling, numerical optimization, and the integration of Python with high-performance C++ code using pybind11.

---

## What is GARCH?
In financial markets, volatility (the degree of variation of a trading price series over time) is not constant. Periods of high volatility tend to be followed by periods of high volatility, and similarly for low volatility. This phenomenon is known as volatility clustering. Traditional models, like simple moving averages, struggle to capture this dynamic behavior.

The GARCH model (Generalized Autoregressive Conditional Heteroskedasticity), introduced by Bollerslev (1986) as an extension of Engle's (1982) ARCH model, is a statistical model used to forecast the future variance (or volatility) of a time series. It's particularly powerful in finance because it explicitly accounts for volatility clustering, making it a cornerstone for risk management, option pricing, and portfolio optimization.

A GARCH(p,q) model uses past squared residuals (ARCH terms) and past conditional variances (GARCH terms) to predict the current conditional variance. For this project, we focus on the GARCH(1,1) model, which is the most commonly used and often sufficient for many financial applications.

---

## GARCH(1,1) Mathematics
The GARCH(1,1) model specifies the conditional variance, $$\sigma_t^2$$, as a function of three terms:

1. A long-run average variance (constant term, $$\omega$$)
2. The squared residual from the previous period (ARCH term, $$\alpha$$)
3. The conditional variance from the previous period (GARCH term, $$\beta$$)


The equation for the GARCH(1,1) conditional variance is:

$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

Where:

- $$\sigma_t^2$$: The conditional variance at time t. This is our forecast for the variance of the next period's returns.
- $$\omega$$: A constant term, representing the long-run average variance. This parameter must be positive ($$\omega > 0$$).
- $$\epsilon_{t-1}^2$$: The squared residual (or error) from the previous period ($$t − 1$$). It represents the impact of past "shocks" on current volatility. $$\epsilon_{t-1}$$ can be computed from

$$
r_t = \mu + \epsilon_{t-1}
$$

$$r_t$$ = return on day $$t$$ <br>
$$\mu$$ = average expected return per day

- $$\sigma_{t-1}^2$$: The conditional variance from the previous period ($$t − 1$$). It signifies the persistence of volatility from the past.
- $$\alpha$$: The coefficient for the ARCH term. It captures the impact of new information (shocks) on volatility. This parameter must be non-negative ($$\alpha \ge 0$$).
- $$\beta$$: The coefficient for the GARCH term. It measures the persistence of volatility, indicating how much past volatility influences current volatility. This parameter must be non-negative ($$\beta \ge 0$$).

For stationarity of the variance process, the sum of the ARCH and GARCH coefficients must be less than one: $$\alpha + \beta < 1$$. This condition ensures that the impact of past shocks eventually diminishes, and the volatility process reverts to its long-run average.

The parameters $$\omega$$, $$\alpha$$ and $$\beta$$ are estimated using an optimization technique, typically by maximizing a likelihood function (e.g., assuming normally distributed errors) or minimizing errors.

---

## Process of Implementation
Building this **GARCH(1,1)** model involved several key steps, all implemented without relying on pre-built econometric libraries for the core estimation, thus showcasing a deep understanding of each component.

### 1. Data Acquisition and Preparation
- `yfinance`: Stock price data (e.g., daily closing prices) was fetched using the yfinance library. This provides a convenient way to access historical market data.
- `NumPy`: Once fetched, the raw price data was processed using NumPy. This involved:
  - Calculating logarithmic returns, as financial models often operate on returns rather than raw prices.
  - Handling any missing data points and ensuring the data was in a suitable format for the subsequent optimization process.

### 2. Custom Gradient Ascent Optimization
This is where the "from scratch" aspect truly shines. Instead of using established optimization libraries (like SciPy's optimizers), the parameters ($$\omega$$, $$\alpha$$, $$\beta$$) of the GARCH(1,1) model were estimated using a manual implementation of **Gradient Ascent**.

**Objective:** The goal of the optimization is to find the parameters that maximize the log-likelihood function of the GARCH(1,1) model, assuming normally distributed errors. Gradient Ascent works by iteratively adjusting the parameters in the direction of the steepest ascent of the likelihood function.

**Manual Implementation:**
- The log-likelihood function for the GARCH(1,1) model was explicitly defined. Which is:

$$
L(\omega, \alpha, \beta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \log(2\pi) + \log(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right]
$$

- The gradients (partial derivatives) of the log-likelihood function with respect to each parameter ($\omega$, $\alpha$, $\beta$) were **derived analytically**. The initial guesses for the parameters themselves were set to heuristical values to begin the iterative gradient ascent optimization process.
- An iterative loop was set up where, in each step, the parameters were updated by moving a small step (determined by a learning rate) in the direction of the calculated gradients.
- **Acknowledgement of Imperfection**: It's important to note that this manual Gradient Ascent implementation is not perfectly optimized or robust compared to highly sophisticated, industrial-grade optimization algorithms found in specialized libraries. It serves to demonstrate a foundational understanding of numerical optimization principles and the ability to implement them from first principles.

### 3. C++ for Performance with pybind11 Integration
To achieve better performance for the computationally intensive parts of the model (especially the iterative calculation of conditional variances and gradients), the core GARCH estimation logic and the Gradient Ascent algorithm were implemented in C++.

- `pybind11`: This powerful library was used to create seamless bindings between the Python frontend and the C++ backend. pybind11 allows Python code to call C++ functions and classes directly, enabling the benefits of C++'s speed while retaining Python's ease of use for data handling and scripting.

- **Workflow:**
  - Python (`yfinance`, `NumPy`) handles data fetching and initial preparation.
  - The prepared data is passed to the C++ module via pybind11.
  - The C++ module performs the heavy lifting: calculating conditional variances, the log-likelihood, and executing the custom Gradient Ascent optimization to find the optimal GARCH parameters.
  - The estimated parameters and other relevant outputs are returned to Python for further analysis or presentation.

---

## Demo
This demo represents a working snapshot of the program's capabilities, current as of May 2025. All resulting values are in Percentage (%).

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
