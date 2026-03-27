"""
Covered Call Dashboard — Previous Trading Day Backend
Uses historical end-of-day data; works at all hours (market open or closed).

Run: /usr/bin/python3 backend_prev_day.py
Then open index.html in a browser (it will auto-detect this backend on port 5051).
"""

from flask import Flask, jsonify
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime, date
import io
import csv

app = Flask(__name__)
CORS(app)

API_KEY = "XW4LEB4MUOHZRXTT"
BASE = "https://www.alphavantage.co/query"


def av(function, **params):
    r = requests.get(BASE, params={"function": function, "apikey": API_KEY, **params}, timeout=15)
    r.raise_for_status()
    return r.json()


def av_csv(function, **params):
    r = requests.get(BASE, params={"function": function, "apikey": API_KEY, **params}, timeout=15)
    r.raise_for_status()
    return r.text


def calc_hv(closes, days=30):
    """Annualised historical volatility from a list of closing prices (newest first)."""
    subset = closes[:days + 1]
    if len(subset) < 2:
        return None
    log_rets = [np.log(subset[i] / subset[i + 1]) for i in range(len(subset) - 1)]
    return round(float(np.std(log_rets) * np.sqrt(252) * 100), 1)


def safe_float(val, default=0.0):
    try:
        return float(val or default)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val or default)
    except (ValueError, TypeError):
        return default


def days_until(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (d - date.today()).days
    except Exception:
        return None


@app.route("/api/dashboard/<symbol>")
def dashboard(symbol):
    symbol = symbol.upper().strip()
    today = date.today()

    # ── Fetch all data ────────────────────────────────────────────────────────
    # Use TIME_SERIES_DAILY for price — always available, not market-hours dependent
    daily_raw    = av("TIME_SERIES_DAILY", symbol=symbol, outputsize="compact", datatype="json")
    overview     = av("COMPANY_OVERVIEW", symbol=symbol)
    # HISTORICAL_OPTIONS with no date defaults to previous trading session
    opts_raw     = av("HISTORICAL_OPTIONS", symbol=symbol, datatype="json")
    earnings_csv = av_csv("EARNINGS_CALENDAR", symbol=symbol, horizon="6month")

    # ── Price from most recent trading day ────────────────────────────────────
    ts = daily_raw.get("Time Series (Daily)", {})
    if not ts:
        return jsonify({"error": f"No data found for symbol '{symbol}'."}), 404

    dates_sorted = sorted(ts.keys(), reverse=True)   # newest first
    last_date    = dates_sorted[0]
    last_bar     = ts[last_date]
    prev_bar     = ts[dates_sorted[1]] if len(dates_sorted) > 1 else {}

    price      = safe_float(last_bar.get("4. close"))
    volume     = safe_int(last_bar.get("5. volume"))
    prev_close = safe_float(prev_bar.get("4. close"))
    change     = round(price - prev_close, 2)
    change_pct = round((change / prev_close * 100), 2) if prev_close else 0.0

    if price == 0:
        return jsonify({"error": f"No price data found for symbol '{symbol}'."}), 404

    # ── Historical volatility ─────────────────────────────────────────────────
    closes = [safe_float(ts[d]["4. close"]) for d in dates_sorted]  # newest first
    hv10   = calc_hv(closes, 10)
    hv30   = calc_hv(closes, 30)

    # ── Options chain (previous trading session) ──────────────────────────────
    raw_options  = opts_raw.get("data", [])
    # Determine the data date from the first option record
    data_date = raw_options[0].get("date", last_date) if raw_options else last_date

    calls_window = []
    all_ivs = []

    for o in raw_options:
        if o.get("type", "").lower() != "call":
            continue

        try:
            exp    = datetime.strptime(o["expiration"], "%Y-%m-%d").date()
            dte    = (exp - today).days
            strike = safe_float(o.get("strike"))
            iv     = safe_float(o.get("implied_volatility"))
            delta  = safe_float(o.get("delta"))
            theta  = safe_float(o.get("theta"))
            gamma  = safe_float(o.get("gamma"))
            vega   = safe_float(o.get("vega"))
            bid    = safe_float(o.get("bid"))
            ask    = safe_float(o.get("ask"))
            # Historical data: use mark price as mid if bid/ask are stale/zero
            mark   = safe_float(o.get("mark"))
            last   = safe_float(o.get("last"))
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif mark > 0:
                mid = mark
            else:
                mid = last
            oi     = safe_int(o.get("open_interest"))
            vol    = safe_int(o.get("volume"))
        except Exception:
            continue

        if iv > 0:
            all_ivs.append(iv)

        if not (30 <= dte <= 60):
            continue
        if strike <= price:      # OTM only
            continue
        if mid == 0:             # no usable price
            continue

        ann_yield = round((mid / price) * (365 / dte) * 100, 2) if dte > 0 and price > 0 else 0

        calls_window.append({
            "contract":    o.get("contractID", ""),
            "expiry":      o["expiration"],
            "dte":         dte,
            "strike":      round(strike, 2),
            "pct_otm":     round((strike / price - 1) * 100, 2),
            "bid":         round(bid, 2),
            "ask":         round(ask, 2),
            "mid":         round(mid, 2),
            "delta":       round(delta, 3),
            "theta":       round(theta, 3),
            "gamma":       round(gamma, 4),
            "vega":        round(vega, 3),
            "iv":          round(iv * 100, 1),
            "oi":          oi,
            "volume":      vol,
            "ann_yield":   ann_yield,
            "_delta_dist": abs(delta - 0.25),
        })

    # Current IV: median of 30-60 DTE options
    current_iv  = round(float(np.median(all_ivs)) * 100, 1) if all_ivs else None
    iv_hv_ratio = round(current_iv / hv30, 2) if current_iv and hv30 else None

    # Recommend one contract per expiry — closest delta to 0.25
    seen = set()
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
    try:
        reader = csv.DictReader(io.StringIO(earnings_csv))
        for row in reader:
            if row.get("symbol", "").upper() == symbol:
                next_earnings = row.get("reportDate") or row.get("date")
                break
    except Exception:
        pass

    if not next_earnings:
        next_earnings = overview.get("NextEarningsDate") or overview.get("LatestQuarter")

    ex_div_date   = overview.get("ExDividendDate")
    div_amount    = overview.get("DividendPerShare", "—")
    div_yield_pct = overview.get("DividendYield", "0")
    try:
        div_yield_pct = f"{float(div_yield_pct) * 100:.2f}%"
    except Exception:
        div_yield_pct = "—"

    earnings_dte = days_until(next_earnings) if next_earnings else None
    exdiv_dte    = days_until(ex_div_date) if ex_div_date else None

    events = {
        "earnings_date":      next_earnings or "—",
        "earnings_dte":       earnings_dte,
        "earnings_in_window": (30 <= earnings_dte <= 60) if earnings_dte is not None else False,
        "ex_div_date":        ex_div_date or "—",
        "exdiv_dte":          exdiv_dte,
        "exdiv_in_window":    (30 <= exdiv_dte <= 60) if exdiv_dte is not None else False,
        "div_amount":         div_amount,
        "div_yield":          div_yield_pct,
    }

    # ── Company info ──────────────────────────────────────────────────────────
    mkt_cap_raw = overview.get("MarketCapitalization", "0")
    try:
        mkt_cap = f"${int(mkt_cap_raw) / 1e9:.1f}B"
    except Exception:
        mkt_cap = "—"

    return jsonify({
        "symbol":      symbol,
        "name":        overview.get("Name", symbol),
        "sector":      overview.get("Sector", "—"),
        "industry":    overview.get("Industry", "—"),
        "price":       price,
        "price_date":  last_date,
        "data_date":   data_date,
        "change":      change,
        "change_pct":  str(change_pct),
        "prev_close":  prev_close,
        "volume":      volume,
        "market_cap":  mkt_cap,
        "beta":        overview.get("Beta", "—"),
        "week52_high": overview.get("52WeekHigh", "—"),
        "week52_low":  overview.get("52WeekLow", "—"),
        "volatility": {
            "current_iv":   current_iv,
            "hv10":         hv10,
            "hv30":         hv30,
            "iv_hv_ratio":  iv_hv_ratio,
            "premium_rich": iv_hv_ratio > 1.1 if iv_hv_ratio else None,
        },
        "events":      events,
        "recommended": recommended,
        "all_calls":   all_calls,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    print("Starting Covered Call Dashboard (prev-day) backend on http://localhost:5051")
    app.run(port=5051, debug=False)
