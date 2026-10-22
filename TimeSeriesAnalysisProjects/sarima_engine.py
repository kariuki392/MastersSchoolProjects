"""Small statsmodels-backed API used by the candy production notebook."""

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf as _acf, pacf as _pacf


def acf(values, nlags):
    return _acf(np.asarray(values, dtype=float), nlags=nlags, fft=False)


def pacf(values, nlags):
    return _pacf(np.asarray(values, dtype=float), nlags=nlags)


def adf(values):
    statistic, pvalue, lag, *_ = adfuller(np.asarray(values, dtype=float))
    return {"stat": statistic, "pvalue": pvalue, "lag": lag}


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "MSE": float(np.mean(errors ** 2)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAPE": float(np.mean(np.abs(errors / actual)) * 100),
    }


def ljung_box(residuals, lags, model_df=0):
    result = acorr_ljungbox(
        np.asarray(residuals, dtype=float), lags=[lags], model_df=model_df, return_df=True
    )
    return float(result["lb_stat"].iloc[0]), float(result["lb_pvalue"].iloc[0]), int(lags - model_df)


class SARIMA:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        result = ARIMA(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
        ).fit()
        self._result = result
        self.params = np.asarray(result.params, dtype=float)
        self.bse = np.asarray(result.bse, dtype=float)
        self.k = len(self.params)

        parameter_names = result.param_names
        variance_index = next(
            (index for index, name in enumerate(parameter_names) if name == "sigma2"),
            None,
        )
        self.sigma2 = (
            float(self.params[variance_index])
            if variance_index is not None
            else float(result.scale)
        )

        burn_in = self.order[1] + self.seasonal_order[1] * self.seasonal_order[3]
        self.resid_full = np.asarray(result.resid[burn_in:], dtype=float)
        self.resid = self.resid_full
        return self

    def param_names(self):
        return list(self._result.param_names)

    def forecast(self, steps):
        prediction = self._result.get_forecast(steps=steps)
        forecast = np.asarray(prediction.predicted_mean, dtype=float)
        interval = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
        return forecast, interval[:, 0], interval[:, 1]
