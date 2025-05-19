import numpy as np
import hvplot.pandas


def fit_and_plot_gamma(data, title="Gamma Distribution Fit"):
    """
    Fits a gamma distribution to data, plots histogram and fitted density.

    Args:
        data (np.ndarray or pd.Series): The dataset.
        title (str): The title for the plot.

    Returns:
        panel.Row: A Panel object containing the plot.
    """
    # Ensure data is a pandas Series for hvplot compatibility
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # Fit the gamma distribution
    params = st.gamma.fit(data)
    print(params)
    arg, loc, scale = params

    # Create the x values for the fitted distribution plot
    x = np.linspace(0, data.max() * 1.1, 1000)
    pdf_fitted = st.gamma.pdf(x, arg, loc=loc, scale=scale)

    # Create the histogram plot
    histogram_plot = data.hvplot(kind='hist', bins=30, normed=True, title=title, legend=False)

    # Create the fitted distribution plot
    fitted_density_plot = pd.DataFrame({'x': x, 'pdf': pdf_fitted}).hvplot(x='x', y='pdf', color='red', legend=False)

    # Overlay the plots
    overlay = histogram_plot * fitted_density_plot

    return overlay

def generate_gamma_data(
    n_samples = 100000,
    n_features = 1,
    linkfunc = np.log,
    beta = [-1,2000],
    dispersion = .001,
    random_state = 42,
    X_scale = 100,
    intercept=1
):
    np.random.seed(random_state)
    
    # Generate predictor matrix with uniform distribution
    X = np.random.uniform(1000, X_scale, (n_samples, n_features)).reshape(-1, n_features)
    # print(X)
    if beta is None:
        beta = np.random.randn(n_features)
    beta = np.array(beta)
    # Calculate linear predictor and expected value
    linear_predictor = intercept + X @ beta
    # mu = np.exp(linear_predictor)  # Log-link transformation
    mu = -1/linear_predictor
    
    shape = 1/dispersion
    scale = shape*mu
    
    y = np.random.gamma(shape=shape, scale=scale)
    
    return y