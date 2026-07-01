"""Price-history data layer for the mean-reversion strategy.

A small SQLite time-series of daily median price + volume per market_hash_name,
fed from two sources:

  * ``steam``   — Steam Community Market pricehistory (years of daily data;
                  used for *trend shape* and cold-start bootstrap only)
  * ``csfloat`` — daily snapshot of the cheapest CSFloat listing (the real,
                  actionable price the bot decides on)

Steam's absolute prices differ from CSFloat (15% Steam fee + locked wallet) so
they are NEVER used as a buy/sell price — only their relative shape matters.
The live buy decision always anchors on CSFloat data.

Only stdlib (``sqlite3``) is used — no extra dependency.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote as url_quote

import aiohttp

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_history (
    market_hash_name TEXT    NOT NULL,
    source           TEXT    NOT NULL,   -- 'steam' | 'csfloat'
    date             TEXT    NOT NULL,   -- ISO yyyy-mm-dd (daily)
    median_price     INTEGER NOT NULL,   -- cents
    volume           INTEGER NOT NULL,
    PRIMARY KEY (market_hash_name, source, date)
);
CREATE TABLE IF NOT EXISTS watchlist (
    market_hash_name TEXT    PRIMARY KEY,
    added_at         TEXT    NOT NULL,
    last_volume      INTEGER NOT NULL DEFAULT 0
);
"""


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class PriceDB:
    """Thin async wrapper over a SQLite price-history store.

    All writes are serialised behind a single :class:`asyncio.Lock` and run in a
    worker thread so the event loop is never blocked.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("PriceDB ready at %s", self._path)

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self._path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── price_history ─────────────────────────────────────────────────────────

    async def upsert_many(self, rows: list[tuple[str, str, str, int, int]]) -> None:
        """rows: (market_hash_name, source, date, median_price, volume)."""
        if not rows:
            return
        async with self._lock:
            await asyncio.to_thread(self._upsert_many_sync, rows)

    def _upsert_many_sync(self, rows) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO price_history"
                "(market_hash_name, source, date, median_price, volume) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(market_hash_name, source, date) DO UPDATE SET "
                "median_price=excluded.median_price, volume=excluded.volume",
                rows,
            )

    async def series(self, name: str, source: str, days: int) -> list[tuple[str, int, int]]:
        """Most recent ``days`` rows for (name, source), chronological order."""
        return await asyncio.to_thread(self._series_sync, name, source, days)

    def _series_sync(self, name, source, days) -> list[tuple[str, int, int]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT date, median_price, volume FROM price_history "
                "WHERE market_hash_name=? AND source=? ORDER BY date DESC LIMIT ?",
                (name, source, days),
            )
            rows = cur.fetchall()
        rows.reverse()
        return rows

    async def has_source(self, name: str, source: str) -> bool:
        return await asyncio.to_thread(self._has_source_sync, name, source)

    def _has_source_sync(self, name, source) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM price_history WHERE market_hash_name=? AND source=? LIMIT 1",
                (name, source),
            )
            return cur.fetchone() is not None

    # ── watchlist ──────────────────────────────────────────────────────────────

    async def add_to_watchlist(self, name: str, volume: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._add_watch_sync, name, volume)

    def _add_watch_sync(self, name, volume) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO watchlist(market_hash_name, added_at, last_volume) "
                "VALUES(?, ?, ?) ON CONFLICT(market_hash_name) DO UPDATE SET "
                "last_volume=excluded.last_volume",
                (name, datetime.now(timezone.utc).isoformat(), volume),
            )

    async def watchlist(self) -> list[str]:
        return await asyncio.to_thread(self._watchlist_sync)

    def _watchlist_sync(self) -> list[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT market_hash_name FROM watchlist ORDER BY last_volume DESC"
            )
            return [r[0] for r in cur.fetchall()]

    async def prune_watchlist(self, min_volume: int, max_size: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._prune_sync, min_volume, max_size)

    def _prune_sync(self, min_volume, max_size) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM watchlist WHERE last_volume < ?", (min_volume,))
            conn.execute(
                "DELETE FROM watchlist WHERE market_hash_name NOT IN "
                "(SELECT market_hash_name FROM watchlist "
                " ORDER BY last_volume DESC LIMIT ?)",
                (max_size,),
            )


# ── Steam Community Market pricehistory fetcher ─────────────────────────────────


def _parse_steam_date(s: str) -> str | None:
    """'Jul 18 2020 01: +0' → '2020-07-18' (drops the hour bucket)."""
    try:
        head = s.split(":")[0].strip()  # 'Jul 18 2020 01'
        dt = datetime.strptime(head, "%b %d %Y %H")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


async def fetch_steam_history(
    session: aiohttp.ClientSession, cookies: str, name: str
) -> list[tuple[str, int, int]] | None:
    """Fetch Steam daily median price history for ``name``.

    Returns ``[(date_iso, median_cent, volume), ...]`` sorted ascending, or
    ``None`` on failure (bad cookie, rate-limit, no data). Multiple intraday
    points are collapsed into one volume-weighted daily median.
    """
    url = (
        "https://steamcommunity.com/market/pricehistory/"
        f"?appid=730&market_hash_name={url_quote(name, safe='')}"
    )
    headers = {
        "Cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://steamcommunity.com/market/",
    }
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.warning("Steam pricehistory HTTP %d for %r", resp.status, name)
                return None
            data = await resp.json(content_type=None)
    except aiohttp.ClientError as exc:
        logger.warning("Steam pricehistory network error %r: %s", name, exc)
        return None
    except Exception as exc:  # malformed JSON, etc.
        logger.warning("Steam pricehistory parse error %r: %s", name, exc)
        return None

    if not isinstance(data, dict) or not data.get("success") or "prices" not in data:
        return None

    daily: dict[str, list[tuple[float, int]]] = {}
    for entry in data["prices"]:
        try:
            day = _parse_steam_date(entry[0])
            price_f = float(entry[1])
            vol = int(entry[2])
        except (IndexError, ValueError, TypeError):
            continue
        if day is None:
            continue
        daily.setdefault(day, []).append((price_f, vol))

    out: list[tuple[str, int, int]] = []
    for day, points in daily.items():
        total_vol = sum(v for _, v in points)
        if total_vol <= 0:
            wmean = sum(p for p, _ in points) / len(points)
        else:
            wmean = sum(p * v for p, v in points) / total_vol
        out.append((day, int(round(wmean * 100)), total_vol))
    out.sort()
    return out or None
