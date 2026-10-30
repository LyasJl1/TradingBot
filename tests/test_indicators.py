"""Fixed reference tables rather than a second implementation in the tests."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tradingbot.indicators import adx, atr, donchian, ema, true_range


@pytest.fixture
def prices() -> pd.DataFrame:
    # TR: 2,3,3,4,4,5,3,3,4,5,5,7; +DM: 0,2,1,0,0,2,1,0,2,0,0,3;
    # -DM: 0,0,0,2,1,0,0,1,0,2,1,0. Exact rational Wilder arithmetic was
    # used to derive the fixed ATR/ADX reference tables below.
    return pd.DataFrame(
        {
            "high": [10, 12, 13, 12, 11, 13, 14, 13, 15, 14, 13, 16],
            "low": [8, 9, 10, 8, 7, 9, 11, 10, 11, 9, 8, 12],
            "close": [9, 11, 12, 9, 8, 12, 13, 11, 14, 10, 9, 15],
        },
        dtype=float,
    )


def test_ema_sma_seed_and_known_reference() -> None:
    close = pd.Series([2.0, 4.0, 6.0, 10.0, 8.0, 6.0])
    # Seed = (2+4+6)/3 = 4; alpha = 1/2 thereafter.
    expected = pd.Series([np.nan, np.nan, 4.0, 7.0, 7.5, 6.75], name="ema")
    assert_series_equal(ema(close, 3), expected)


def test_true_range_accounts_for_gaps_and_first_bar(prices: pd.DataFrame) -> None:
    np.testing.assert_array_equal(
        true_range(prices), [2, 3, 3, 4, 4, 5, 3, 3, 4, 5, 5, 7]
    )


def test_wilder_atr_known_reference(prices: pd.DataFrame) -> None:
    # First seed 8/3; next value (2*(8/3)+4)/3 = 28/9.
    expected = [
        np.nan,
        np.nan,
        2.6666666666666665,
        3.111111111111111,
        3.4074074074074074,
        3.9382716049382718,
        3.625514403292181,
        3.4170096021947876,
        3.611339734796525,
        4.07422648986435,
        4.382817659909566,
        5.255211773273045,
    ]
    np.testing.assert_allclose(atr(prices, 3), expected, rtol=1e-13, equal_nan=True)


def test_wilder_atr_fourteen_bar_seed() -> None:
    # All previous closes are 0: TR is exactly 1,2,...,16.
    frame = pd.DataFrame({"high": range(1, 17), "low": 0.0, "close": 0.0})
    actual = atr(frame, 14)
    assert actual.iloc[:13].isna().all()
    np.testing.assert_allclose(
        actual.iloc[13:], [7.5, 8.035714285714286, 8.604591836734695], rtol=1e-13
    )


def test_wilder_adx_mixed_direction_reference(prices: pd.DataFrame) -> None:
    # First DX values are 20, 100/13, 400/11; ADX seed is 9160/429.
    expected = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        21.35198135198135,
        31.336103510016553,
        24.856437709785954,
        33.41813055501976,
        25.34534778790466,
        26.905638256086064,
        29.94218424961686,
    ]
    np.testing.assert_allclose(adx(prices, 3), expected, rtol=1e-13, equal_nan=True)


def test_wilder_adx_fourteen_bar_reference(prices: pd.DataFrame) -> None:
    frame = pd.concat([prices] * 3, ignore_index=True)
    actual = adx(frame, 14)
    assert actual.iloc[:27].isna().all()
    # Same explicit OHLC table repeated three times; fixed reference derived
    # using exact rational directional-movement sums and Wilder recursion.
    expected = [
        6.258290180076284,
        6.175386854252565,
        5.9610506622719726,
        6.043229120419693,
        5.79858352446434,
        6.147822660632563,
        5.8243555657891815,
        5.599354711836097,
        5.889254737729303,
    ]
    np.testing.assert_allclose(actual.iloc[27:], expected, rtol=1e-13)


def test_adx_flat_market_is_zero_after_warmup() -> None:
    flat = pd.DataFrame({"high": [4.0] * 40, "low": [4.0] * 40, "close": [4.0] * 40})
    result = adx(flat, 14)
    assert result.iloc[:27].isna().all()
    assert (result.iloc[27:] == 0.0).all()
    assert not np.isinf(result).any()


def test_adx_tied_directional_movement_counts_neither_side() -> None:
    # Range expands symmetrically each candle, so +DM and -DM tie.
    frame = pd.DataFrame({"high": range(10, 18), "low": range(9, 1, -1), "close": 9.5})
    result = adx(frame, 3)
    assert (result.iloc[5:] == 0.0).all()


def test_donchian_reference_excludes_current_candle(prices: pd.DataFrame) -> None:
    actual = donchian(prices.iloc[:6], 3)
    expected = pd.DataFrame(
        {"upper": [np.nan, np.nan, np.nan, 13.0, 13.0, 13.0],
         "lower": [np.nan, np.nan, np.nan, 8.0, 8.0, 7.0]}
    )
    assert_frame_equal(actual, expected)
    changed = prices.copy()
    changed.loc[5, ["high", "low"]] = [999.0, -999.0]
    assert_frame_equal(donchian(prices, 3).iloc[:6], donchian(changed, 3).iloc[:6])
    assert donchian(changed, 3).iloc[6]["upper"] == 999.0


def test_indicators_preserve_inputs_and_index(prices: pd.DataFrame) -> None:
    prices.index = pd.date_range("2026-01-01", periods=len(prices), freq="4h", tz="UTC")
    before = prices.copy(deep=True)
    for result in (atr(prices, 3), adx(prices, 3), donchian(prices, 3), ema(prices.close, 3)):
        assert result.index.equals(prices.index)
    assert_frame_equal(prices, before)


@pytest.mark.parametrize("period", [0, -1, 1.5, True, "14"])
def test_invalid_period_fails_explicitly(prices: pd.DataFrame, period: object) -> None:
    for function, values in ((ema, prices.close), (atr, prices), (adx, prices), (donchian, prices)):
        with pytest.raises(ValueError, match="positive integer"):
            function(values, period)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_fail_explicitly(prices: pd.DataFrame, invalid: float) -> None:
    broken = prices.copy()
    broken.loc[3, "high"] = invalid
    for function in (atr, adx, donchian):
        with pytest.raises(ValueError, match="finite"):
            function(broken, 3)
    with pytest.raises(ValueError, match="finite"):
        ema(pd.Series([1.0, invalid, 2.0]), 2)


def test_missing_or_inconsistent_prices_fail(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing price columns"):
        atr(prices.drop(columns="close"), 3)
    broken = prices.copy()
    broken.loc[0, "low"] = 100
    with pytest.raises(ValueError, match="high"):
        adx(broken, 3)
    broken = prices.copy()
    broken.loc[0, "close"] = 100
    with pytest.raises(ValueError, match="close must lie"):
        atr(broken, 3)


def test_short_inputs_do_not_invent_warmup_values(prices: pd.DataFrame) -> None:
    assert atr(prices, 14).isna().all()
    assert adx(prices, 14).isna().all()
    assert ema(prices.close, 14).isna().all()
    assert donchian(prices, 14).isna().all().all()
