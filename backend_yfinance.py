"""
Covered Call Dashboard — yFinance Backend
Uses Yahoo Finance; no API key required, works at all hours.
Greeks calculated via Black-Scholes using the IV from each contract.

Run: /usr/bin/python3 backend_yfinance.py
Then open index.html and select the yFinance backend.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, date
import math

app = Flask(__name__)
CORS(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_hv(closes, days=30):
    """Annualised historical volatility. closes = list newest-first."""
    subset = closes[:days + 1]
    if len(subset) < 2:
        return None
    log_rets = [math.log(subset[i] / subset[i + 1]) for i in range(len(subset) - 1)]
    return round(float(np.std(log_rets) * math.sqrt(252) * 100), 1)


def bs_greeks(S, K, T, r, sigma):
    """Black-Scholes call delta, gamma, theta/day, vega/1%IV."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    try:
        d1    = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2    = d1 - sigma * math.sqrt(T)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        vega  = S * norm.pdf(d1) * math.sqrt(T) / 100
        return round(delta, 3), round(gamma, 4), round(theta, 3), round(vega, 3)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def to_date_str(val):
    """Normalise a date/datetime/timestamp/str to YYYY-MM-DD string."""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            return val[:10]
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val).strftime("%Y-%m-%d")
        if hasattr(val, 'strftime'):
            return val.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def days_until(date_str):
    try:
        return (datetime.strptime(date_str, "%Y-%m-%d").date() - date.today()).days
    except Exception:
        return None


def build_vol_history(hist_df, days=5):
    """Return per-session HV10/HV30 for the last `days` trading sessions."""
    closes_all = list(reversed(hist_df["Close"].tolist()))   # newest first
    dates_all  = list(reversed([str(d.date()) for d in hist_df.index]))
    history = []
    for i in range(min(days, len(closes_all))):
        closes_from_i = closes_all[i:]
        close_i    = closes_from_i[0]
        prev_close = closes_from_i[1] if len(closes_from_i) > 1 else close_i
        daily_ret  = round((close_i / prev_close - 1) * 100, 2) if prev_close else 0.0
        history.append({
            "date":         dates_all[i],
            "close":        round(close_i, 2),
            "daily_return": daily_ret,
            "hv10":         calc_hv(closes_from_i, 10),
            "hv30":         calc_hv(closes_from_i, 30),
        })
    return history


# ── Route ─────────────────────────────────────────────────────────────────────

@app.route("/api/dashboard/<symbol>")
def dashboard(symbol):
    symbol = symbol.upper().strip()
    today  = date.today()
    ticker = yf.Ticker(symbol)

    # ── Price ─────────────────────────────────────────────────────────────────
    fi         = ticker.fast_info
    price      = safe_float(fi.last_price)
    prev_close = safe_float(fi.previous_close)
    volume     = safe_int(fi.last_volume)

    if price == 0:
        return jsonify({"error": f"No data found for symbol '{symbol}'."}), 404

    change     = round(price - prev_close, 2)
    change_pct = round((change / prev_close * 100), 2) if prev_close else 0.0

    # ── Company info ──────────────────────────────────────────────────────────
    info     = ticker.info
    name     = info.get("longName") or info.get("shortName") or symbol
    sector   = info.get("sector", "—")
    industry = info.get("industry", "—")
    beta     = info.get("beta", "—")
    mkt_cap_raw = info.get("marketCap", 0)
    mkt_cap  = f"${mkt_cap_raw / 1e9:.1f}B" if mkt_cap_raw else "—"
    week52_high = str(info.get("fiftyTwoWeekHigh", "—"))
    week52_low  = str(info.get("fiftyTwoWeekLow",  "—"))
    div_rate    = info.get("dividendRate", "—")
    div_yield   = info.get("dividendYield", 0)
    div_yield_pct = f"{div_yield * 100:.2f}%" if div_yield else "—"

    # ── Historical volatility ─────────────────────────────────────────────────
    hist    = ticker.history(period="3mo")
    closes  = list(reversed(hist["Close"].tolist()))  # newest first
    hv10    = calc_hv(closes, 10)
    hv30    = calc_hv(closes, 30)
    vol_history = build_vol_history(hist)

    # ── Risk-free rate (10Y Treasury via ^TNX) ────────────────────────────────
    try:
        r = safe_float(yf.Ticker("^TNX").fast_info.last_price) / 100
        if r == 0:
            r = 0.045
    except Exception:
        r = 0.045

    # ── Options chain — 30-60 DTE OTM calls only ─────────────────────────────
    try:
        expirations = ticker.options
    except Exception:
        expirations = []

    calls_window = []
    all_ivs      = []

    for exp_str in expirations:
        try:
            exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp - today).days
        except Exception:
            continue

        if not (30 <= dte <= 60):
            continue

        try:
            chain    = ticker.option_chain(exp_str)
            calls_df = chain.calls
        except Exception:
            continue

        T = dte / 365.0

        for _, row in calls_df.iterrows():
            strike = safe_float(row.get("strike"))
            iv     = safe_float(row.get("impliedVolatility"))
            bid    = safe_float(row.get("bid"))
            ask    = safe_float(row.get("ask"))
            last   = safe_float(row.get("lastPrice"))
            oi     = safe_int(row.get("openInterest"))
            vol    = safe_int(row.get("volume"))

            if strike <= price:     # OTM only
                continue
            if iv <= 0.001:         # skip contracts with no meaningful IV
                continue

            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
            if mid <= 0:
                continue

            all_ivs.append(iv)
            delta, gamma, theta, vega = bs_greeks(price, strike, T, r, iv)
            ann_yield = round((mid / price) * (365 / dte) * 100, 2) if dte > 0 else 0

            calls_window.append({
                "contract":    str(row.get("contractSymbol", "")),
                "expiry":      exp_str,
                "dte":         dte,
                "strike":      round(strike, 2),
                "pct_otm":     round((strike / price - 1) * 100, 2),
                "bid":         round(bid, 2),
                "ask":         round(ask, 2),
                "mid":         round(mid, 2),
                "delta":       delta,
                "theta":       theta,
                "gamma":       gamma,
                "vega":        vega,
                "iv":          round(iv * 100, 1),
                "oi":          oi,
                "volume":      vol,
                "ann_yield":   ann_yield,
                "_delta_dist": abs(delta - 0.25),
            })

    current_iv  = round(float(np.median(all_ivs)) * 100, 1) if all_ivs else None
    iv_hv_ratio = round(current_iv / hv30, 2) if current_iv and hv30 else None

    # Recommend one per expiry — closest delta to 0.25
    seen        = set()
    recommended = []
    for c in sorted(calls_window, key=lambda x: (x["expiry"], x["_delta_dist"])):
        if c["expiry"] not in seen:
            seen.add(c["expiry"])
            recommended.append({k: v for k, v in c.items() if k != "_delta_dist"})

    all_calls = sorted(
        [{k: v for k, v in c.items() if k != "_delta_dist"} for c in calls_window],
        key=lambda x: (x["expiry"], x["strike"]),
    )

    # ── Calendar events ───────────────────────────────────────────────────────
    next_earnings = None
    ex_div_date   = None
    try:
        cal = ticker.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                next_earnings = to_date_str(ed[0])
            elif ed:
                next_earnings = to_date_str(ed)
            ex_div_date = to_date_str(cal.get("Ex-Dividend Date"))
    except Exception:
        pass

    # Fallback to info timestamps
    if not next_earnings:
        next_earnings = to_date_str(info.get("nextEarningsDate") or info.get("earningsTimestamp"))
    if not ex_div_date:
        ex_div_date = to_date_str(info.get("exDividendDate"))

    earnings_dte = days_until(next_earnings) if next_earnings else None
    exdiv_dte    = days_until(ex_div_date)   if ex_div_date   else None

    events = {
        "earnings_date":      next_earnings or "—",
        "earnings_dte":       earnings_dte,
        "earnings_in_window": (30 <= earnings_dte <= 60) if earnings_dte is not None else False,
        "ex_div_date":        ex_div_date or "—",
        "exdiv_dte":          exdiv_dte,
        "exdiv_in_window":    (30 <= exdiv_dte <= 60) if exdiv_dte is not None else False,
        "div_amount":         str(div_rate),
        "div_yield":          div_yield_pct,
    }

    return jsonify({
        "symbol":      symbol,
        "name":        name,
        "sector":      sector,
        "industry":    industry,
        "price":       price,
        "price_date":  str(today),
        "data_date":   str(today),
        "change":      change,
        "change_pct":  str(change_pct),
        "prev_close":  prev_close,
        "volume":      volume,
        "market_cap":  mkt_cap,
        "beta":        str(beta),
        "week52_high": week52_high,
        "week52_low":  week52_low,
        "volatility": {
            "current_iv":   current_iv,
            "hv10":         hv10,
            "hv30":         hv30,
            "iv_hv_ratio":  iv_hv_ratio,
            "premium_rich": iv_hv_ratio > 1.1 if iv_hv_ratio else None,
            "history":      vol_history,
        },
        "events":      events,
        "recommended": recommended,
        "all_calls":   all_calls,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    print("Starting Covered Call Dashboard (yFinance) backend on http://localhost:5052")
    app.run(port=5052, debug=False)
