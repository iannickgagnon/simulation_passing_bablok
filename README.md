# Passing-Bablok Regression Simulator

An interactive demo that shows how Ordinary Least Squares (OLS) is less reliable than Passing-Bablok in method comparison analysis.

Clicking here will create a copy in your own Google Drive: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iannickgagnon/simulation_passing_bablok/blob/main/passing_bablok_demo.ipynb)

---

## User interface

!['simulator screenshot]('/docs/screenshot.png')

---

## Background

Method comparison involves plotting measurements from two analytical methods against each other and fitting a regression line to assess agreement. This introduces two common pitfalls:

1. **Regression dilution**: When the independent variable (x-axis) has measurement error, OLS underestimates the true slope, leading to a biased assessment of agreement.
2. **Outlier skewing**: OLS is sensitive to outliers, which can disproportionately influence the slope and intercept, further distorting the agreement analysis.

Passing-Bablok regression addresses both issues: it is robust to measurement error in x and resistant to outliers, making it a standard choice for method comparison in clinical and laboratory settings. It achieves this by estimating the slope from the median of all pairwise slopes between observations, rather than by minimizing squared vertical residuals, thereby avoiding the ordinary least-squares assumption that the reference method is error-free and reducing the influence of extreme observations.

> Source: [Wikipedia article: Passing-Bablok regression](https://en.wikipedia.org/wiki/Passing%E2%80%93Bablok_regression)

---

## How the simulator works

Synthetic data is generated with a known true slope of 1.0 and true intercept of 0.0. Both OLS and Passing-Bablok are fitted to the same data, and a stats table shows each method's parameter estimates alongside their absolute bias from ground truth.

### Controls

| Control | Effect |
|---|---|
| **X-Axis Error** `slider` | Adds Gaussian noise to x-values, inducing regression dilution in OLS.|
| **Add Outliers** `checkbox` | Injects three severe low-value outliers to demonstrate outlier skewing. |
| **Download dataset (CSV)** `button` | Exports the current synthetic dataset |
