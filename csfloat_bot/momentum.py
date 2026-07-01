"""Mean-reversion buying strategy built on price history.

Runs ALONGSIDE the existing reactive sniper (``DecisionEngine``). It buys items
trading well below their own recent baseline (z-score), gated by liquidity and a
long-term trend filter that rejects falling knives / structural declines. Items
are sold on recovery to baseline through the shared ``PortfolioManager``
lifecycle, tagged ``strategy="momentum"`` with a ``target_price`` = baseline.

Core principle: Steam history only shapes the trend / bootstraps cold start;
the actionable price and the buy always come from live CSFloat data.

Deliberately simple — no RSI/MACD/Bollinger. Few knobs ⇒ little to overfit,
and every buy logs a one-sentence, auditable reason.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime, timezone

from pricedb import PriceDB, fetch_steam_history

logger = logging.getLogger(__name__)

# Statuses that count as "capital still tied up in a momentum position".
_ACTIVE_STATUSES = {"pending_asset", "pending_trade", "locked", "listed"}


def _zscore(baseline: list[int], current: float) -> float | None:
    """How many population std-devs ``current`` sits from the baseline mean."""
    if len(baseline) < 5:
        return None
    sd = statistics.pstdev(baseline)
    if sd == 0:
        return None
    return (current - statistics.mean(baseline)) / sd


def _norm_slope(values: list[int]) -> float | None:
    """Least-squares slope of a *median-normalised* series, per day.

    Normalising by the series median makes the slope a scale-free fraction-per-day,
    so it is comparable across items and unaffected by the Steam↔CSFloat price
    offset (only the sign/shape matters here).
    """
    n = len(values)
    if n < 5:
        return None
    med = statistics.median(values)
    if med <= 0:
        return None
    ys = [v / med for v in values]
    mx = (n - 1) / 2
    my = statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in range(n))
    if den == 0:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(range(n), ys))
    return num / den


class MomentumStrategy:
    def __init__(
        self,
        config,
        session,
        pricedb: PriceDB,
        client,
        engine,
        executor,
        tracker,
        notifier,
    ) -> None:
        self.config = config
        self.session = session
        self.pricedb = pricedb
        self.client = client
        self.engine = engine
        self.executor = executor
        self.tracker = tracker
        self.notifier = notifier
        self.steam_cookies = config.steam_cookies

    # ── budget accounting (derived from the shared tracker) ─────────────────────

    def _active_cents(self) -> int:
        return sum(
            it.buy_price
            for it in self.tracker.all()
            if getattr(it, "strategy", "sniper") == "momentum"
            and it.status in _ACTIVE_STATUSES
        )

    # ── background loop: one watchlist item per tick (no bursts) ─────────────────

    async def run_loop(self) -> None:
        """Round-robin the watchlist: process one item per tick.

        Spreading the work (snapshot + signal) one item at a time keeps momentum
        under the tight shared CSFloat rate limit — it sips ~1 request per tick
        instead of bursting through every item at once and starving the poller.
        A full cycle takes ``len(watchlist) × mr_tick_interval`` (~1h for 60 @ 60s),
        which also refreshes each item's price snapshot roughly hourly.
        """
        names: list[str] = []
        idx = scanned = signals = bought = 0
        while True:
            if not self.config.mr_enabled:
                await asyncio.sleep(self.config.mr_tick_interval)
                continue

            # End of a cycle: emit summary, then prune + reload the watchlist.
            if idx >= len(names):
                if names and (signals or bought) and self.notifier is not None:
                    await self.notifier.send_momentum_summary(
                        scanned, signals, bought,
                        self._active_cents(), self.config.mr_budget_cents,
                    )
                if names:
                    logger.info(
                        "momentum cycle done: scanned=%d signals=%d bought=%d budget=%d/%d¢",
                        scanned, signals, bought,
                        self._active_cents(), self.config.mr_budget_cents,
                    )
                await self.pricedb.prune_watchlist(
                    self.config.mr_min_volume, self.config.mr_watchlist_max
                )
                names = await self.pricedb.watchlist()
                idx = scanned = signals = bought = 0
                if not names:
                    logger.info("momentum: watchlist empty — bot pazarı taradıkça dolacak")
                    await asyncio.sleep(self.config.mr_warmup_interval)
                    continue

            name = names[idx]
            idx += 1
            scanned += 1
            try:
                got_signal, did_buy = await self._process_item(name)
                signals += got_signal
                bought += did_buy
            except Exception as exc:
                logger.error("momentum process %r failed: %s", name, exc, exc_info=True)
            await asyncio.sleep(self.config.mr_tick_interval)

    async def _process_item(self, name: str) -> tuple[int, int]:
        """One item: fetch live price once, snapshot it, bootstrap Steam, evaluate.

        A single CSFloat request serves both the daily snapshot and the live-price
        signal — no double fetch. Returns (signal?, bought?) as 0/1 ints.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cheapest = await self.client.fetch_cheapest_listing(name)

        # Record today's CSFloat snapshot (upsert keyed on date → one row/day).
        if cheapest is not None:
            try:
                price = int(cheapest["price"])
                vol = int((cheapest.get("reference") or {}).get("quantity") or 0)
                await self.pricedb.upsert_many([(name, "csfloat", today, price, vol)])
            except (KeyError, TypeError, ValueError):
                pass

        # Steam deep-history bootstrap: once per item (separate domain — free of
        # the CSFloat rate budget).
        if self.steam_cookies and not await self.pricedb.has_source(name, "steam"):
            rows = await fetch_steam_history(self.session, self.steam_cookies, name)
            if rows:
                await self.pricedb.upsert_many(
                    [(name, "steam", d, p, v) for d, p, v in rows]
                )

        decision = await self._signal(name, cheapest)
        if decision is None:
            return 0, 0
        did_buy = await self._try_buy(name, decision)
        return 1, (1 if did_buy else 0)

    async def _signal(self, name: str, cheapest: dict | None) -> dict | None:
        """Mean-reversion decision from an already-fetched cheapest listing.

        No network here — the caller supplies ``cheapest`` (or None). Returns a
        buy-decision dict if every gate passes, else None.
        """
        if cheapest is None:
            return None
        try:
            live = int(cheapest["price"])
            volume = int((cheapest.get("reference") or {}).get("quantity") or 0)
        except (KeyError, TypeError, ValueError):
            return None

        # Price-band sanity (reuse the sniper's bounds).
        if live < self.config.min_item_price or live > self.config.max_item_price:
            return None

        # 1) Liquidity gate — must be able to exit after the 8-day hold.
        if volume < self.config.mr_min_volume:
            return None

        # 2) Baseline for the dip test requires REAL CSFloat history (accumulated
        #    from daily snapshots). Steam's absolute level differs from CSFloat
        #    (≈15% fee + wallet premium), so without a stable anchor it cannot
        #    measure an absolute CSFloat dip — Steam is used only for the trend
        #    filter below. Exclude today's own snapshot so live isn't compared
        #    against itself.
        csf = await self.pricedb.series(name, "csfloat", self.config.mr_lookback_days)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        baseline_prices = [p for d, p, _ in csf if d != today]
        if len(baseline_prices) < self.config.mr_min_points:
            return None

        # 3) z-score: is live well below its own recent baseline?
        z = _zscore(baseline_prices, live)
        if z is None or z > self.config.mr_z_entry:
            return None

        # 4) Long-term trend filter — reject falling knives.
        slope = await self._trend_slope(name)
        if slope is None or slope < self.config.mr_slope_min:
            return None

        # 5) Expected profit on recovery to baseline, after the seller fee.
        baseline = int(statistics.median(baseline_prices))
        if baseline <= 0:
            return None
        profit_frac = (baseline - live) / baseline
        if profit_frac < self.config.mr_min_profit_margin + self.config.csfloat_seller_fee:
            return None

        # 6) Separate momentum budget.
        if self._active_cents() + live > self.config.mr_budget_cents:
            logger.info(
                "momentum: %r passes signal but budget full (%d+%d > %d) — skip",
                name, self._active_cents(), live, self.config.mr_budget_cents,
            )
            return None

        return {
            "listing_raw": cheapest,
            "live": live,
            "baseline": baseline,
            "z": z,
            "slope": slope,
            "volume": volume,
        }

    async def _trend_slope(self, name: str) -> float | None:
        """Normalised slope of the *established* trend (excludes today's dip).

        Prefers the deeper Steam series; falls back to CSFloat history with the
        current day removed so the dip being evaluated can't trip its own
        falling-knife filter.
        """
        steam = [p for _, p, _ in await self.pricedb.series(
            name, "steam", self.config.mr_trend_days)]
        if len(steam) >= self.config.mr_min_points:
            return _norm_slope(steam)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        csf = [p for d, p, _ in await self.pricedb.series(
            name, "csfloat", self.config.mr_trend_days) if d != today]
        return _norm_slope(csf)

    async def _try_buy(self, name: str, decision: dict) -> bool:
        listing = self.engine.extract_listing_data(decision["listing_raw"])
        if listing is None:
            return False
        baseline = decision["baseline"]
        listing["fair_value"] = baseline        # sell target / base for tracker
        listing["base_price"] = baseline
        listing["target_price"] = baseline
        listing["strategy"] = "momentum"
        listing["buy_reason"] = (
            f"mean-reversion z={decision['z']:.2f} "
            f"baseline={baseline}¢ live={decision['live']}¢ "
            f"slope={decision['slope']:+.4f} vol={decision['volume']}"
        )
        logger.info("momentum BUY signal %r — %s", name, listing["buy_reason"])
        if self.notifier is not None:
            await self.notifier.send_match(listing)
        return await self.executor.buy(listing)
