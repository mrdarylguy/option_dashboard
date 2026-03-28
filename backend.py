"""
Covered Call Dashboard — Backend
Run: python backend.py
Then open index.html in a browser.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime, date
from pathlib import Path
import io
import csv

app = Flask(__name__)
CORS(app)  # Allow requests from file:// origin

def _load_api_key(filename="alphavantage_api_key.txt"):
    for directory in [Path(__file__).parent, Path(__file__).parent.parent]:
        path = directory / filename
        if path.exists():
            return path.read_text().strip()
    raise FileNotFoundError(f"'{filename}' not found in script directory or parent directory.")

API_KEY = _load_api_key()
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


def build_vol_history(dates_sorted, ts, days=5):
    """Return per-session HV10/HV30 for the last `days` trading sessions."""
    all_closes = [safe_float(ts[d]["4. close"]) for d in dates_sorted]
    history = []
    for i in range(min(days, len(dates_sorted))):
        closes_from_i = all_closes[i:]
        close_i    = closes_from_i[0]
        prev_close = closes_from_i[1] if len(closes_from_i) > 1 else close_i
        daily_ret  = round((close_i / prev_close - 1) * 100, 2) if prev_close else 0.0
        history.append({
            "date":         dates_sorted[i],
            "close":        round(close_i, 2),
            "daily_return": daily_ret,
            "hv10":         calc_hv(closes_from_i, 10),
            "hv30":         calc_hv(closes_from_i, 30),
        })
    return history


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
    quote_raw    = av("GLOBAL_QUOTE", symbol=symbol).get("Global Quote", {})
    overview     = av("COMPANY_OVERVIEW", symbol=symbol)
    opts_raw     = av("REALTIME_OPTIONS", symbol=symbol, require_greeks="true", datatype="json")
    daily_raw    = av("TIME_SERIES_DAILY", symbol=symbol, outputsize="compact", datatype="json")

    # Earnings calendar (CSV)
    earnings_csv = av_csv("EARNINGS_CALENDAR", symbol=symbol, horizon="6month")

    # ── Quote ─────────────────────────────────────────────────────────────────
    price      = safe_float(quote_raw.get("05. price"))
    change     = safe_float(quote_raw.get("09. change"))
    change_pct = quote_raw.get("10. change percent", "0%").replace("%", "")
    volume     = safe_int(quote_raw.get("06. volume"))
    prev_close = safe_float(quote_raw.get("08. previous close"))

    if price == 0:
        return jsonify({"error": f"No data found for symbol '{symbol}'."}), 404

    # ── Historical volatility ─────────────────────────────────────────────────
    ts = daily_raw.get("Time Series (Daily)", {})
    dates_sorted_hv = sorted(ts.keys(), reverse=True)
    closes = [safe_float(ts[d]["4. close"]) for d in dates_sorted_hv]  # newest first
    hv10  = calc_hv(closes, 10)
    hv30  = calc_hv(closes, 30)
    vol_history = build_vol_history(dates_sorted_hv, ts)

    # ── Options chain ─────────────────────────────────────────────────────────
    raw_options = opts_raw.get("data", [])
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
            mid    = (bid + ask) / 2
            oi     = safe_int(o.get("open_interest"))
            vol    = safe_int(o.get("volume"))
        except Exception:
            continue

        if iv > 0:
            all_ivs.append(iv)

        if not (30 <= dte <= 60):
            continue
        if strike <= price:          # OTM only
            continue
        if bid == 0 and ask == 0:    # no market
            continue

        ann_yield = round((mid / price) * (365 / dte) * 100, 2) if dte > 0 and price > 0 else 0

        calls_window.append({
            "contract":  o.get("contractID", ""),
            "expiry":    o["expiration"],
            "dte":       dte,
            "strike":    round(strike, 2),
            "pct_otm":   round((strike / price - 1) * 100, 2),
            "bid":       round(bid, 2),
            "ask":       round(ask, 2),
            "mid":       round(mid, 2),
            "delta":     round(delta, 3),
            "theta":     round(theta, 3),
            "gamma":     round(gamma, 4),
            "vega":      round(vega, 3),
            "iv":        round(iv * 100, 1),
            "oi":        oi,
            "volume":    vol,
            "ann_yield": ann_yield,
            "_delta_dist": abs(delta - 0.25),
        })

    # Current IV: median of 30-60 DTE near-ATM options
    current_iv = round(float(np.median(all_ivs)) * 100, 1) if all_ivs else None
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
    # Parse earnings CSV
    next_earnings = None
    try:
        reader = csv.DictReader(io.StringIO(earnings_csv))
        for row in reader:
            if row.get("symbol", "").upper() == symbol:
                next_earnings = row.get("reportDate") or row.get("date")
                break
    except Exception:
        pass

    # Fallback to overview fields
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
        "symbol":     symbol,
        "name":       overview.get("Name", symbol),
        "sector":     overview.get("Sector", "—"),
        "industry":   overview.get("Industry", "—"),
        "price":      price,
        "change":     round(change, 2),
        "change_pct": change_pct,
        "prev_close": prev_close,
        "volume":     volume,
        "market_cap": mkt_cap,
        "beta":       overview.get("Beta", "—"),
        "week52_high": overview.get("52WeekHigh", "—"),
        "week52_low":  overview.get("52WeekLow", "—"),
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
    print("Starting Covered Call Dashboard backend on http://localhost:5050")
    app.run(port=5050, debug=False)
