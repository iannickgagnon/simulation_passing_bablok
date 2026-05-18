# pyright: reportUnknownMemberType=false

import tempfile
from typing import Any, cast, Final

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gradio.themes import Soft
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

# Fix random seed for reproducibility
np.random.seed(seed=42)


def _calculate_passing_bablok(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """
    Compute Passing-Bablok slope and intercept.

    This implementation computes all valid pairwise slopes, takes their median
    as the slope (m), and then computes the median intercept (b) as b = y - m * x.

    Args:
        x (ArrayLike): Independent variable measurements.
        y (ArrayLike): Dependent variable measurements.

    Returns:
        A tuple containing:
            - slope (float): The Passing-Bablok slope.
            - intercept (float): The Passing-Bablok intercept.
    """
    
    # Convert inputs to numpy arrays of type float
    x_arr: NDArray[np.float64] = np.asarray(a=x, dtype=float)
    y_arr: NDArray[np.float64] = np.asarray(a=y, dtype=float)

    # Calculate all pairwise slopes
    n_observations: int = len(x_arr)
    all_slopes: list[float] = []
    for i in range(n_observations):
        for j in range(i + 1, n_observations):
            # Avoid division by zero if two x-values are exactly identical
            if x_arr[j] != x_arr[i]:
                all_slopes.append((y_arr[j] - y_arr[i]) / (x_arr[j] - x_arr[i]))
                
    # Fallback: Slope of 1.0 and intercept of 0.0 if no valid slopes are found (e.g., all x-values are identical)
    if not all_slopes:
        return 1.0, 0.0 

    # The slope is the median of the pairwise slopes
    slope: float = cast(float, np.median(a=all_slopes))

    # The intercept is the median of (Y - slope * X)
    intercept: float = cast(float, np.median(a=y_arr - slope * x_arr))
    
    return slope, intercept


# True parameters baked into the data-generating process
_TRUE_SLOPE: Final[float] = 1.0
_TRUE_INTERCEPT: Final[float] = 0.0


def _generate_and_plot(x_error: float, inject_outliers: bool) -> tuple[Figure, str, pd.DataFrame]:
    """
    Generate synthetic data, fit regressions, and build output artifacts.

    Args:
        x_error (float): Standard deviation of the Gaussian noise added to x-values.
        inject_outliers (bool): Whether to append predefined outlier points.

    Returns:
        A tuple containing:
            - fig (Figure): The Matplotlib figure object with the plot.
            - csv_path (str): The file path to the temporary CSV containing the dataset.
            - stats (pd.DataFrame): Per-method parameter estimates and bias vs. ground truth.
    """

    # 1. Generate baseline data with baseline variability
    x_true: NDArray[np.float64] = np.linspace(start=2, stop=18, num=30)
    y_true: NDArray[np.float64] = x_true + np.random.normal(loc=0, scale=0.8, size=30)
    
    # 2. Inject X-Axis variability
    x_obs: NDArray[np.float64] = x_true + np.random.normal(loc=0, scale=x_error, size=30)
    y_obs: NDArray[np.float64] = y_true.copy()
    
    # 3. Add negative proportional outliers
    if inject_outliers:
        outlier_x: NDArray[np.float64] = np.array(object=[14, 16, 17], dtype=float)
        outlier_y: NDArray[np.float64] = np.array(object=[3, 4, 2], dtype=float)
        x_obs = np.append(arr=x_obs, values=outlier_x)
        y_obs = np.append(arr=y_obs, values=outlier_y)

    # 4. Calculate Regressions
    
    # OLS
    slope_ols, intercept_ols = cast(tuple[float, float], np.polyfit(x=x_obs, y=y_obs, deg=1))
    
    # Passing-Bablok
    slope_pb, intercept_pb = _calculate_passing_bablok(x=x_obs, y=y_obs)

    # 5. Build plot
    fig: Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x=x_obs, y=y_obs, color='#2c3e50', alpha=0.7, label='Dataset')
    
    # Create regression lines extending across the plot
    x_range: NDArray[np.float64] = np.array(object=[min(x_obs) - 1, max(x_obs) + 1], dtype=float)
    ax.plot(x_range,
            intercept_ols + slope_ols * x_range,
            color='#e74c3c', 
            linestyle='--', 
            linewidth=2, 
            label=f'OLS (Slope: {slope_ols:.2f})')
    
    ax.plot(x_range, intercept_pb + slope_pb * x_range,
            color='#2980b9',
            linestyle='-', 
            linewidth=2, 
            label=f'Passing-Bablok (Slope: {slope_pb:.2f})')

    # Decorate
    ax.set_xlabel(xlabel="Method 1 (with varying measurement error)")
    ax.set_ylabel(ylabel="Method 2")
    ax.set_title(label="Method Comparison: OLS vs Passing-Bablok")
    ax.legend(loc='upper left')
    ax.grid(visible=True, linestyle=':', alpha=0.6)
    
    # 6. Build stats table comparing each method against the known ground truth
    ols_slope_deviation: float = abs(slope_ols - _TRUE_SLOPE)
    ols_slope_deviation_pct: float = (ols_slope_deviation / abs(_TRUE_SLOPE)) * 100 if _TRUE_SLOPE != 0 else 0.0
    ols_intercept_deviation: float = abs(intercept_ols - _TRUE_INTERCEPT)
    
    pb_slope_deviation: float = abs(slope_pb - _TRUE_SLOPE)
    pb_slope_deviation_pct: float = (pb_slope_deviation / abs(_TRUE_SLOPE)) * 100 if _TRUE_SLOPE != 0 else float('inf')
    pb_intercept_deviation: float = abs(intercept_pb - _TRUE_INTERCEPT)

    stats: pd.DataFrame = pd.DataFrame(
        data={
            "Metric": ["Slope", 
                       "Intercept", 
                       "|Slope - True slope (1.0)|", 
                       "Slope deviation from truth (%)",
                       "|Intercept - True intercept (0.0)|"
            ],
            "OLS": [
                round(slope_ols, 3),
                round(intercept_ols, 3),
                round(ols_slope_deviation, 3),
                round(ols_slope_deviation_pct, 3),
                round(ols_intercept_deviation, 3),
            ],
            "Passing-Bablok": [
                round(slope_pb, 3),
                round(intercept_pb, 3),
                round(pb_slope_deviation, 3),
                round(pb_slope_deviation_pct, 3),
                round(pb_intercept_deviation, 3),
            ],
        }
    )

    # 7. Save data to temporary CSV for download
    df = pd.DataFrame({
        "Method_1_X": x_obs,
        "Method_2_Y": y_obs
    })

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(path_or_buf=temp_file.name, index=False)

    return fig, temp_file.name, stats


with gr.Blocks(theme=Soft()) as app:
    
    ##########
    # HEADER #
    ##########
    
    gr.Markdown(
        value="""
        # Passing-Bablok Regression Simulator

        Method comparison consists of plotting measurements from two methods against each other and fitting a regression line to assess agreement. This introduces two common pitfalls:
        1. **Regression dilution**: When the independent variable (x-axis) has measurement error, Ordinary Least Squares (OLS) regression underestimates the true slope, leading to a biased assessment of agreement.
        2. **Outlier skewing**: OLS is sensitive to outliers, which can disproportionately influence the slope and intercept, further distorting the agreement analysis.

        >Source: [click here](https://en.wikipedia.org/wiki/Passing%E2%80%93Bablok_regression)

        Adjust the measurement error and outliers to see how OLS is impacted by regression dilution and outlier skewing, while Passing-Bablok remains robust.
        """
    )
    
    ##########
    # LAYOUT #
    ##########

    with gr.Row():
      
        # Column no.1 (left): Controls and Download
        with gr.Column(scale=1):

            # Controls area
            gr.Markdown(value="### Controls")
            
            error_slider = gr.Slider(minimum=0.0, 
                                     maximum=5.0, 
                                     value=0.0, 
                                     step=0.1, 
                                     label="X-Axis Error")
            
            outlier_checkbox = gr.Checkbox(label="Add outliers", value=False)
            
            # Download area
            gr.Markdown(value="### Export")
            download_btn = gr.File(label="Download dataset (CSV)")
            
        # Column no.2 (right): Plot and stats area
        with gr.Column(scale=3):
            plot_output = gr.Plot(show_label=False)
            stats_output = gr.DataFrame(
                label="Parameter estimates vs. ground truth (true slope = 1.0, true intercept = 0.0)",
                interactive=False,
            )

    ###################
    # EVENT LISTENERS #
    ###################

    # When slider or checkbox changes, update the plot, file download, and stats table
    inputs: list[Any] = [error_slider, outlier_checkbox]
    outputs: list[Any] = [plot_output, download_btn, stats_output]
    
    error_slider.change(fn=_generate_and_plot, inputs=inputs, outputs=outputs)
    outlier_checkbox.change(fn=_generate_and_plot, inputs=inputs, outputs=outputs)
    
    # Initialize app with default values
    app.load(fn=_generate_and_plot, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    app.launch()
