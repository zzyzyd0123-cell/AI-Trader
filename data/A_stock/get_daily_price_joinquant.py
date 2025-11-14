"""Utility to fetch SSE 50 data via JoinQuant DataAPI token.

This mirrors the behaviour of `get_daily_price_tushare.py` but calls
JoinQuant's REST API so teams that only拥有聚宽 token 也能准备上证50
行情以及指数基准数据。输出文件与原脚本保持一致：
- `daily_prices_sse_50.csv`: 个股日线（字段同 Tushare 版本）
- `index_daily_sse_50.json`: 指数日线 JSON（供性能评估使用）
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

JOINQUANT_API_URL = "https://dataapi.joinquant.com/apis"
EXCHANGE_TO_JQ = {"SH": "XSHG", "SZ": "XSHE"}
EXCHANGE_FROM_JQ = {v: k for k, v in EXCHANGE_TO_JQ.items()}
DEFAULT_INDEX_CODE = "000016.XSHG"  # 上证50


class JoinQuantAPIError(RuntimeError):
    """Raised when JoinQuant returns an error payload."""


def _normalize_date(date_str: str, end: bool = False) -> str:
    """Convert YYYYMMDD string to 'YYYY-MM-DD HH:MM:SS' for JoinQuant."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    time_part = "23:59:59" if end else "00:00:00"
    return f"{dt.strftime('%Y-%m-%d')} {time_part}"


def _get_env_token() -> str:
    token = os.getenv("JOINQUANT_TOKEN") or os.getenv("JQDATA_TOKEN")
    if not token:
        raise JoinQuantAPIError(
            "JOINQUANT_TOKEN not configured. Please add it to your .env file."
        )
    return token.strip()


def _call_joinquant_api(
    method: str,
    token: str,
    *,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    **params: str,
) -> str:
    payload = {"method": method, "token": token}
    payload.update(params)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(JOINQUANT_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            text = resp.text.strip()
            if text.lower().startswith("error"):
                raise JoinQuantAPIError(text)
            return text
        except (JoinQuantAPIError, requests.RequestException) as exc:
            if attempt == max_retries:
                raise JoinQuantAPIError(str(exc))
            wait = retry_backoff * attempt
            print(f"⚠️ JoinQuant API failed ({exc}), retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise JoinQuantAPIError("Unexpected retry exhaustion")


def _parse_table_response(text: str) -> pd.DataFrame:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return pd.DataFrame()
    header = lines[0].split(",")
    rows = [row.split(",") for row in lines[1:]]
    normalized_rows = []
    for values in rows:
        if len(values) < len(header):
            values += [""] * (len(header) - len(values))
        normalized_rows.append(dict(zip(header, values)))
    return pd.DataFrame(normalized_rows)


def jq_code_to_ts_code(code: str) -> str:
    symbol, market = code.split(".")
    suffix = EXCHANGE_FROM_JQ.get(market, "")
    if not suffix:
        raise ValueError(f"Unsupported market suffix: {market}")
    return f"{symbol}.{suffix}"


def ts_code_to_jq_code(ts_code: str) -> str:
    symbol, market = ts_code.split(".")
    jq_suffix = EXCHANGE_TO_JQ.get(market)
    if not jq_suffix:
        raise ValueError(f"Unsupported TS code suffix: {market}")
    return f"{symbol}.{jq_suffix}"


def get_last_month_dates() -> Tuple[str, str]:
    today = datetime.now()
    first_day_this_month = today.replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    first_day_last_month = last_day_last_month.replace(day=1)
    return (
        first_day_last_month.strftime("%Y-%m-%d"),
        last_day_last_month.strftime("%Y-%m-%d"),
    )


def get_index_constituents(
    token: str,
    index_code: str = DEFAULT_INDEX_CODE,
    fallback_csv: Optional[Path] = None,
) -> pd.DataFrame:
    start, end = get_last_month_dates()
    print(f"🔍 Fetching index weights via JoinQuant ({start} ~ {end})")
    response = _call_joinquant_api(
        "get_index_weights",
        token,
        code=index_code,
        date=end,
    )
    df = _parse_table_response(response)

    if df.empty and fallback_csv and Path(fallback_csv).exists():
        print(f"⚠️ JoinQuant returned empty data, using fallback file: {fallback_csv}")
        df = pd.read_csv(fallback_csv)
        df.rename(columns={"con_code": "ts_code", "stock_name": "display_name"}, inplace=True)
    elif not df.empty:
        df.rename(columns={"code": "jq_code"}, inplace=True)
        df["ts_code"] = df["jq_code"].apply(jq_code_to_ts_code)
    else:
        raise JoinQuantAPIError("Failed to fetch index constituents and no fallback available")

    unique_count = df["ts_code"].nunique()
    print(f"✅ Loaded {unique_count} constituents")
    return df


def fetch_price_period(
    token: str,
    jq_code: str,
    start_date: str,
    end_date: str,
    fields: str = "open,close,high,low,volume",
) -> pd.DataFrame:
    start_dt = _normalize_date(start_date, end=False)
    end_dt = _normalize_date(end_date, end=True)
    response = _call_joinquant_api(
        "get_price_period",
        token,
        code=jq_code,
        date=start_dt,
        end_date=end_dt,
        unit="1d",
        fq="pre",
        fields=fields,
    )
    df = _parse_table_response(response)
    return df


def assemble_daily_price_df(
    token: str,
    ts_codes: Iterable[str],
    start_date: str,
    end_date: str,
    throttle: float = 0.5,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for idx, ts_code in enumerate(ts_codes, start=1):
        jq_code = ts_code_to_jq_code(ts_code)
        print(f"[{idx:02d}/{len(ts_codes)}] Fetching prices for {ts_code} ({jq_code})")
        df = fetch_price_period(token, jq_code, start_date, end_date)
        if df.empty:
            print(f"⚠️ No data returned for {ts_code}")
            continue
        df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            volume_series = pd.to_numeric(df["volume"], errors="coerce")
            df["vol"] = (volume_series / 100).fillna(0)
        else:
            df["vol"] = 0
        df["ts_code"] = ts_code
        frames.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "vol"]])
        time.sleep(throttle)

    if not frames:
        raise JoinQuantAPIError("No price data collected from JoinQuant")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["trade_date", "open", "close"])
    combined = combined.sort_values(by=["trade_date", "ts_code"])
    print(f"✅ Combined dataframe shape: {combined.shape}")
    return combined


def save_daily_prices(
    df: pd.DataFrame,
    output_dir: Path,
    index_code: str = DEFAULT_INDEX_CODE,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sse_50" if index_code.startswith("000016") else index_code.replace(".", "_")
    file_path = output_dir / f"daily_prices_{suffix}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"💾 Saved constituent prices to {file_path}")
    return file_path


def convert_index_daily_to_json(
    df: pd.DataFrame,
    symbol: str,
    output_file: Path,
) -> None:
    if df.empty:
        print("⚠️ Empty index dataframe, skip JSON export")
        return

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(by="trade_date", ascending=False)

    json_data: Dict[str, Dict[str, Dict[str, str]]] = {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": symbol,
            "3. Last Refreshed": df.iloc[0]["trade_date"],
            "4. Output Size": "Compact",
            "5. Time Zone": "Asia/Shanghai",
        },
        "Time Series (Daily)": {},
    }

    for _, row in df.iterrows():
        json_data["Time Series (Daily)"][row["trade_date"]] = {
            "1. open": f"{row['open']:.4f}",
            "2. high": f"{row['high']:.4f}",
            "3. low": f"{row['low']:.4f}",
            "4. close": f"{row['close']:.4f}",
            "5. volume": str(int(row.get("volume", 0) or 0)),
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(json_data, fout, indent=2, ensure_ascii=False)
    print(f"💾 Saved index benchmark JSON to {output_file}")


def get_index_daily_data(
    token: str,
    index_code: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> None:
    df = fetch_price_period(token, index_code, start_date, end_date)
    if df.empty:
        print("⚠️ No index data returned, skip benchmark export")
        return
    output_file = output_dir / "index_daily_sse_50.json"
    convert_index_daily_to_json(df, symbol=index_code, output_file=output_file)


def get_daily_price_a_stock_joinquant(
    index_code: str = DEFAULT_INDEX_CODE,
    daily_start_date: str = "20250101",
    fallback_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    token = _get_env_token()

    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    constituents = get_index_constituents(token, index_code=index_code, fallback_csv=fallback_csv)
    ts_codes = sorted(constituents["ts_code"].unique())

    end_date = datetime.now().strftime("%Y%m%d")
    df_daily = assemble_daily_price_df(token, ts_codes, daily_start_date, end_date)
    save_daily_prices(df_daily, output_dir, index_code=index_code)

    # Export benchmark JSON
    get_index_daily_data(token, index_code, daily_start_date, end_date, output_dir)

    return df_daily


if __name__ == "__main__":
    print("=" * 60)
    print("JoinQuant SSE 50 Data Downloader")
    print("=" * 60)
    local_fallback = Path(__file__).parent / "sse_50_weight.csv"
    if not local_fallback.exists():
        alt = Path(__file__).resolve().parents[1] / "sse_50_weight.csv"
        if alt.exists():
            local_fallback = alt
    df = get_daily_price_a_stock_joinquant(fallback_csv=local_fallback)
    print(f"Rows fetched: {len(df)}")
    print("=" * 60)
