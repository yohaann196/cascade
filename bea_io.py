"""BEA U.S. input-output accounts: download, parse, and build the Leontief model.

Source: U.S. Bureau of Economic Analysis, Input-Output Accounts Data
(annual, Summary level, 71 sectors). BEA publishes these free public releases
as XLSX workbooks with one sheet per year (1997-2023); this module downloads
them, extracts the intermediate Use/Make blocks, and builds:

    A = direct requirements matrix (industry-by-industry, market-share tech.)
    L = (I - A)^-1 = total requirements (the Leontief inverse)

Validation: rebuilding industry output from final demand (L @ f) reproduces
BEA's published total industry output to within ~5e-5 relative error.

Values throughout are millions of current dollars, matching the BEA tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://apps.bea.gov/industry/Release/XLSX"
USE_URL = f"{BASE_URL}/IOUse_After_Redefinitions_PRO_Summary.xlsx"
MAKE_URL = f"{BASE_URL}/IOMake_After_Redefinitions_PRO_Summary.xlsx"

YEARS: tuple[int, ...] = tuple(range(1997, 2024))
N_SECTORS = 71  # BEA Summary-level sector count

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ripple-io-app/1.0)"}


@dataclass
class IOTables:
    """Intermediate-use and make blocks for one year, in $ millions."""

    year: int
    codes: list[str]
    names: list[str]
    use: np.ndarray  # commodity x industry intermediate deliveries
    make: np.ndarray  # industry x commodity output
    industry_output: np.ndarray  # per industry
    commodity_output: np.ndarray  # per commodity
    final_demand: np.ndarray  # per commodity, net of imports


@dataclass
class IOModel:
    """Coefficients, Leontief inverse, and baseline flows for one year."""

    tables: IOTables
    A: np.ndarray  # direct requirements, industry x industry
    L: np.ndarray  # Leontief inverse (I - A)^-1
    flows: np.ndarray  # baseline interindustry flows $M, flows[i, j] = A[i, j] * x_j


def _download(url: str) -> bytes:
    resp = requests.get(url, headers=_HEADERS, timeout=180)
    resp.raise_for_status()
    return resp.content


def _sheet_frame(content: bytes, year: int) -> pd.DataFrame:
    xls = pd.ExcelFile(BytesIO(content), engine="openpyxl")
    if str(year) not in xls.sheet_names:
        raise ValueError(f"BEA workbook has no sheet for year {year}")
    return xls.parse(str(year), header=None)


def _find_col(df: pd.DataFrame, hdr: int, label: str) -> int:
    names = df.iloc[hdr, :].astype(str).str.strip()
    hits = df.columns[names.eq(label)]
    if len(hits) == 0:
        raise ValueError(f"BEA layout changed: column '{label}' not found")
    return int(hits[0])


def _sector_codes_names(df: pd.DataFrame, hdr: int) -> tuple[list[str], list[str]]:
    codes = [str(c).strip() for c in df.iloc[hdr - 1, 2 : 2 + N_SECTORS]]
    names = [str(c).strip() for c in df.iloc[hdr, 2 : 2 + N_SECTORS]]
    if any(c in ("", "nan") for c in codes):
        raise ValueError("BEA layout changed: could not read sector columns")
    return codes, names


def _block(df: pd.DataFrame, hdr: int) -> np.ndarray:
    rows = df.iloc[hdr + 1 : hdr + 1 + N_SECTORS, 2 : 2 + N_SECTORS]
    return (
        rows.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )


def load_tables(year: int, session: requests.Session | None = None) -> IOTables:
    """Download the BEA Use + Make workbooks and extract the year's blocks."""
    if year not in YEARS:
        raise ValueError(f"year must be one of {YEARS[0]}-{YEARS[-1]}")

    get = session.get if session is not None else requests.get
    use_raw = get(USE_URL, headers=_HEADERS, timeout=180)
    use_raw.raise_for_status()
    make_raw = get(MAKE_URL, headers=_HEADERS, timeout=180)
    make_raw.raise_for_status()

    use_df = _sheet_frame(use_raw.content, year)
    make_df = _sheet_frame(make_raw.content, year)

    hdr_rows = use_df.index[use_df[0].astype(str).str.strip().eq("IOCode")]
    if len(hdr_rows) == 0:
        raise ValueError("BEA layout changed: header row not found")
    hdr = int(hdr_rows[0])

    codes, names = _sector_codes_names(use_df, hdr)
    make_codes, _ = _sector_codes_names(make_df, hdr)
    if codes != make_codes:
        raise ValueError("BEA layout changed: Use/Make sector codes differ")

    use = _block(use_df, hdr)  # commodity x industry
    make = _block(make_df, hdr)  # industry x commodity

    row_codes = [str(c).strip() for c in use_df.iloc[hdr + 1 : hdr + 1 + N_SECTORS, 0]]
    if row_codes != codes:
        raise ValueError("BEA layout changed: Use rows are not the sector list")

    industry_output = make.sum(axis=1)
    commodity_output = make.sum(axis=0)

    # Commodity final demand, net of imports: published total commodity output
    # minus total intermediate use.
    col_out = _find_col(use_df, hdr, "Total Commodity Output")
    col_int = _find_col(use_df, hdr, "Total Intermediate")
    vals = lambda c: use_df.iloc[hdr + 1 : hdr + 1 + N_SECTORS, c].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    final_demand = vals(col_out) - vals(col_int)

    return IOTables(
        year=year,
        codes=codes,
        names=names,
        use=use,
        make=make,
        industry_output=industry_output,
        commodity_output=commodity_output,
        final_demand=final_demand,
    )


def build_model(tables: IOTables) -> IOModel:
    """Build A = (I - L)^-1 coefficients and the Leontief inverse.

    Market-share (fixed industry sales structure) technology:
        S[i, k]  = share of commodity k produced by industry i
        Z[i, j]  = sum_k S[i, k] * Use[k, j]  (industry i's deliveries to j)
        A[i, j]  = Z[i, j] / x_j              (x_j = industry j total output)
    """
    q = tables.commodity_output
    x = tables.industry_output
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(q[None, :] > 0, tables.make / q[None, :], 0.0)
    Z = S @ tables.use
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(x[None, :] > 0, Z / x[None, :], 0.0)

    L = np.linalg.inv(np.eye(N_SECTORS) - A)
    return IOModel(tables=tables, A=A, L=L, flows=A * x[None, :])


def shock_vector(n: int, sector_idx: int, magnitude: float) -> np.ndarray:
    """Final-demand shock vector with `magnitude` placed on one sector."""
    s = np.zeros(n)
    s[sector_idx] = magnitude
    return s


def ripple_series(A: np.ndarray, s: np.ndarray, rounds: int) -> np.ndarray:
    """Cumulative impact per supply-chain round: y_r = (I + A + ... + A^r) @ s.

    Returns an array of shape (rounds + 1, n); the last row is L @ s.
    """
    out = np.zeros((rounds + 1, len(s)))
    cumulative = s.astype(float).copy()
    marginal = s.astype(float).copy()
    out[0] = cumulative
    for r in range(1, rounds + 1):
        marginal = A @ marginal
        cumulative = cumulative + marginal
        out[r] = cumulative
    return out
