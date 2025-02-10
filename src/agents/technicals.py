import math
from typing import Dict

from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from tools.api import prices_to_df


##### 技术分析师 #####
def technical_analyst_agent(state: AgentState):
    """
    复杂的技术分析系统，结合多种交易策略：
    1. 趋势跟随
    2. 均值回归
    3. 动量
    4. 波动率分析
    5. 统计套利信号
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)

    # 计算指标
    # 1. MACD（移动平均收敛发散）
    macd_line, signal_line = calculate_macd(prices_df)

    # 2. RSI（相对强弱指数）
    rsi = calculate_rsi(prices_df)

    # 3. 布林带（Bollinger Bands）
    upper_band, lower_band = calculate_bollinger_bands(prices_df)

    # 4. OBV（平衡交易量）
    obv = calculate_obv(prices_df)

    # 生成单独信号
    signals = []

    # MACD信号
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append('bullish')  # 看涨
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append('bearish')  # 看跌
    else:
        signals.append('neutral')  # 中性

    # RSI信号
    if rsi.iloc[-1] < 30:
        signals.append('bullish')  # 看涨
    elif rsi.iloc[-1] > 70:
        signals.append('bearish')  # 看跌
    else:
        signals.append('neutral')  # 中性

    # 布林带信号
    current_price = prices_df['close'].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append('bullish')  # 看涨
    elif current_price > upper_band.iloc[-1]:
        signals.append('bearish')  # 看跌
    else:
        signals.append('neutral')  # 中性

    # OBV信号
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append('bullish')  # 看涨
    elif obv_slope < 0:
        signals.append('bearish')  # 看跌
    else:
        signals.append('neutral')  # 中性

    # 添加推理收集
    reasoning = {
        "MACD": {
            "signal": signals[0],
            "details": f"MACD线穿越了{'上方' if signals[0] == 'bullish' else '下方' if signals[0] == 'bearish' else '既不在上方也不在下方'}信号线"
        },
        "RSI": {
            "signal": signals[1],
            "details": f"RSI为{rsi.iloc[-1]:.2f}（{'超卖' if signals[1] == 'bullish' else '超买' if signals[1] == 'bearish' else '中性'}）"
        },
        "Bollinger": {
            "signal": signals[2],
            "details": f"价格{'低于下轨' if signals[2] == 'bullish' else '高于上轨' if signals[2] == 'bearish' else '在带内'}"
        },
        "OBV": {
            "signal": signals[3],
            "details": f"OBV斜率为{obv_slope:.2f}（{signals[3]}）"
        }
    }

    # 确定整体信号
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')

    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'  # 看涨
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'  # 看跌
    else:
        overall_signal = 'neutral'  # 中性

    # 根据指标一致性计算置信水平
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals

    # 生成消息内容
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": {
            "MACD": reasoning["MACD"],
            "RSI": reasoning["RSI"],
            "Bollinger": reasoning["Bollinger"],
            "OBV": reasoning["OBV"]
        }
    }

    # 1. 趋势跟随策略
    trend_signals = calculate_trend_signals(prices_df)

    # 2. 均值回归策略
    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

    # 3. 动量策略
    momentum_signals = calculate_momentum_signals(prices_df)

    # 4. 波动率策略
    volatility_signals = calculate_volatility_signals(prices_df)

    # 5. 统计套利信号
    stat_arb_signals = calculate_stat_arb_signals(prices_df)

    # 使用加权集成方法组合所有信号
    strategy_weights = {
        'trend': 0.25,
        'mean_reversion': 0.20,
        'momentum': 0.25,
        'volatility': 0.15,
        'stat_arb': 0.15
    }

    combined_signal = weighted_signal_combination({
        'trend': trend_signals,
        'mean_reversion': mean_reversion_signals,
        'momentum': momentum_signals,
        'volatility': volatility_signals,
        'stat_arb': stat_arb_signals
    }, strategy_weights)

    # 生成详细分析报告
    analysis_report = {
        "signal": combined_signal['signal'],
        "confidence": f"{round(combined_signal['confidence'] * 100)}%",
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals['signal'],
                "confidence": f"{round(trend_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(trend_signals['metrics'])
            },
            "mean_reversion": {
                "signal": mean_reversion_signals['signal'],
                "confidence": f"{round(mean_reversion_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(mean_reversion_signals['metrics'])
            },
            "momentum": {
                "signal": momentum_signals['signal'],
                "confidence": f"{round(momentum_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(momentum_signals['metrics'])
            },
            "volatility": {
                "signal": volatility_signals['signal'],
                "confidence": f"{round(volatility_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(volatility_signals['metrics'])
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals['signal'],
                "confidence": f"{round(stat_arb_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(stat_arb_signals['metrics'])
            }
        }
    }

    # 创建技术分析师消息
    message = HumanMessage(
        content=json.dumps(analysis_report),
        name="technical_analyst_agent",
    )

    if show_reasoning:
        show_agent_reasoning(analysis_report, "技术分析师")

    return {
        "messages": [message],
        "data": data,
    }


def calculate_trend_signals(prices_df):
    """
    高级趋势跟随策略，使用多个时间框架和指标
    """
    # 计算多个时间框架的EMA
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # 计算ADX以评估趋势强度
    adx = calculate_adx(prices_df, 14)

    # 计算一目均衡表
    ichimoku = calculate_ichimoku(prices_df)

    # 确定趋势方向和强度
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # 结合信号与置信度加权
    trend_strength = adx['adx'].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = 'bullish'  # 看涨
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = 'bearish'  # 看跌
        confidence = trend_strength
    else:
        signal = 'neutral'  # 中性
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'adx': float(adx['adx'].iloc[-1]),
            'trend_strength': float(trend_strength),
            # 'ichimoku': ichimoku
        }
    }


def calculate_mean_reversion_signals(prices_df):
    """
    使用统计指标和布林带的均值回归策略
    """
    # 计算价格相对于移动平均的z-score
    ma_50 = prices_df['close'].rolling(window=50).mean()
    std_50 = prices_df['close'].rolling(window=50).std()
    z_score = (prices_df['close'] - ma_50) / std_50

    # 计算布林带
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # 计算多个时间框架的RSI
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # 均值回归信号
    extreme_z_score = abs(z_score.iloc[-1]) > 2
    price_vs_bb = (prices_df['close'].iloc[-1] - bb_lower.iloc[-1]
                   ) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # 结合信号
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = 'bullish'  # 看涨
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = 'bearish'  # 看跌
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = 'neutral'  # 中性
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'z_score': float(z_score.iloc[-1]),
            'price_vs_bb': float(price_vs_bb),
            'rsi_14': float(rsi_14.iloc[-1]),
            'rsi_28': float(rsi_28.iloc[-1])
        }
    }


def calculate_momentum_signals(prices_df):
    """
    多因素动量策略
    """
    # 价格动量
    returns = prices_df['close'].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # 交易量动量
    volume_ma = prices_df['volume'].rolling(21).mean()
    volume_momentum = prices_df['volume'] / volume_ma

    # 相对强度
    # （在实际实现中会与市场/行业进行比较）

    # 计算动量得分
    momentum_score = (
        0.4 * mom_1m +
        0.3 * mom_3m +
        0.3 * mom_6m
    ).iloc[-1]

    # 交易量确认
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = 'bullish'  # 看涨
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = 'bearish'  # 看跌
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = 'neutral'  # 中性
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'momentum_1m': float(mom_1m.iloc[-1]),
            'momentum_3m': float(mom_3m.iloc[-1]),
            'momentum_6m': float(mom_6m.iloc[-1]),
            'volume_momentum': float(volume_momentum.iloc[-1])
        }
    }


def calculate_volatility_signals(prices_df):
    """
    基于波动率的交易策略
    """
    # 计算各种波动率指标
    returns = prices_df['close'].pct_change()

    # 历史波动率
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # 波动率状态检测
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # 波动率均值回归
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR比率
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df['close']

    # 根据波动率状态生成信号
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = 'bullish'  # 低波动状态，潜在扩张
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = 'bearish'  # 高波动状态，潜在收缩
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = 'neutral'  # 中性
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'historical_volatility': float(hist_vol.iloc[-1]),
            'volatility_regime': float(current_vol_regime),
            'volatility_z_score': float(vol_z),
            'atr_ratio': float(atr_ratio.iloc[-1])
        }
    }


def calculate_stat_arb_signals(prices_df):
    """
    基于价格行为分析的统计套利信号
    """
    # 计算价格分布统计
    returns = prices_df['close'].pct_change()

    # 偏度和峰度
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # 使用赫斯特指数测试均值回归
    hurst = calculate_hurst_exponent(prices_df['close'])

    # 相关性分析
    # （在实际实现中会包括与相关证券的相关性）

    # 根据统计特性生成信号
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = 'bullish'  # 看涨
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = 'bearish'  # 看跌
        confidence = (0.5 - hurst) * 2
    else:
        signal = 'neutral'  # 中性
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'hurst_exponent': float(hurst),
            'skewness': float(skew.iloc[-1]),
            'kurtosis': float(kurt.iloc[-1])
        }
    }


def weighted_signal_combination(signals, weights):
    """
    使用加权方法组合多个交易信号
    """
    # 将信号转换为数值
    signal_values = {
        'bullish': 1,
        'neutral': 0,
        'bearish': -1
    }

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal['signal']]
        weight = weights[strategy]
        confidence = signal['confidence']

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # 归一化加权和
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # 转换回信号
    if final_score > 0.2:
        signal = 'bullish'  # 看涨
    elif final_score < -0.2:
        signal = 'bearish'  # 看跌
    else:
        signal = 'neutral'  # 中性

    return {
        'signal': signal,
        'confidence': abs(final_score)
    }


def normalize_pandas(obj):
    """将pandas系列/数据框转换为原始Python类型"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(
    prices_df: pd.DataFrame,
    window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    计算指数移动平均

    参数：
        df: 包含价格数据的数据框
        window: EMA周期

    返回：
        pd.Series: EMA值
    """
    return df['close'].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算平均方向指数（ADX）

    参数：
        df: 包含OHLC数据的数据框
        period: 计算周期

    返回：
        包含ADX值的数据框
    """
    # 计算真实范围
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # 计算方向性移动
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # 计算ADX
    df['+di'] = 100 * (df['plus_dm'].ewm(span=period).mean() /
                       df['tr'].ewm(span=period).mean())
    df['-di'] = 100 * (df['minus_dm'].ewm(span=period).mean() /
                       df['tr'].ewm(span=period).mean())
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].ewm(span=period).mean()

    return df[['adx', '+di', '-di']]


def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    计算一目均衡表指标

    参数：
        df: 包含OHLC数据的数据框

    返回：
        包含一目均衡表组件的字典
    """
    # 转换线（Tenkan-sen）：（9期高 + 9期低）/2
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # 基准线（Kijun-sen）：（26期高 + 26期低）/2
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # 领先跨度A（Senkou Span A）：（转换线 + 基准线）/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # 领先跨度B（Senkou Span B）：（52期高 + 52期低）/2
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # 滞后跨度（Chikou Span）：收盘价向后移动26期
    chikou_span = df['close'].shift(-26)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算平均真实范围

    参数：
        df: 包含OHLC数据的数据框
        period: ATR计算周期

    返回：
        pd.Series: ATR值
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    计算赫斯特指数以确定时间序列的长期记忆
    H < 0.5: 均值回归序列
    H = 0.5: 随机游走
    H > 0.5: 趋势序列

    参数：
        price_series: 类似数组的价格数据
        max_lag: R/S计算的最大滞后

    返回：
        float: 赫斯特指数
    """
    lags = range(2, max_lag)
    # 添加小的epsilon以避免log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:],
               price_series[:-lag])))) for lag in lags]

    # 从线性拟合返回赫斯特指数
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # 赫斯特指数是斜率
    except (ValueError, RuntimeWarning):
        # 如果计算失败，返回0.5（随机游走）
        return 0.5


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']
