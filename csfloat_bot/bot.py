"""
CSFloat Async Trading Bot
=========================
Run: python bot.py
Logs: bot.log (written alongside this script)

Credentials go in a .env file next to this script (copy .env.example).

Auto-discovery mode
-------------------
The bot scans the entire CSFloat market feed automatically — no manual item list
needed. It buys any listing that passes all four filters:

  1. Not in BLACKLIST
  2. price in [MIN_ITEM_PRICE, MAX_ITEM_PRICE]
  3. reference.quantity >= MIN_VOLUME   (liquidity / resellability proxy)
  4. fair_value >= MIN_FAIR_VALUE       (filters junk items with no reference data)
  5. price <= fair_value * (1 - MIN_PROFIT_MARGIN)

Fair value = CSFloat's predicted_price (ML estimate), falling back to base_price.
CSFloat charges ~2% seller fee, so MIN_PROFIT_MARGIN=0.15 → ~13% net profit.

How it works:
  Polls GET /api/v1/listings?sort_by=most_recent every poll_interval seconds.
  First poll seeds the seen-set (no buys). Subsequent polls act on new listings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import aiohttp
from dotenv import load_dotenv


# ── Filters ───────────────────────────────────────────────────────────────────

# Minimum profit margin: buy only when listing_price <= fair_value * (1 - margin)
# 0.15 = need 15% discount (≈13% net after CSFloat ~2% fee)
# 0.10 = more aggressive, more matches
# 0.20 = conservative, fewer matches, bigger margin
MIN_PROFIT_MARGIN: float = 0.25

# Minimum active CSFloat listings for an item (reference.quantity)
# This is a liquidity proxy — items with few listings are hard to resell.
# 50  = balanced (common skins and cases)
# 100 = only high-volume items
MIN_VOLUME: int = 50

# Minimum predicted/base price to be considered (cents)
# Filters out junk items that have no real reference data (pred=1)
MIN_FAIR_VALUE: int = 20   # $0.20

# Price range gate — never buy outside this band regardless of other filters
MIN_ITEM_PRICE: int = 10    # $0.10 — below this there's no profit potential
MAX_ITEM_PRICE: int = 1000  # $10.00 — safety cap per item

# Items to never buy, even if all other conditions are met.
# Use exact market_hash_name strings (copy from CSFloat listing JSON).
BLACKLIST: frozenset[str] = frozenset(
    {
        # "AK-47 | Redline (Minimal Wear)",
        # "Recoil Case",
    }
)

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    log_path = Path(__file__).parent / "bot.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    logging.getLogger("aiohttp").setLevel(logging.WARNING)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class BotConfig:
    bearer_token: str
    cookies: str
    min_profit_margin: float
    min_volume: int
    min_fair_value: int
    min_item_price: int
    max_item_price: int

    # ── Polling settings ──────────────────────────────────────────────────────
    poll_interval: float = 15.0
    poll_pages: int = 3        # cursor-pages per cycle; 3 × 50 = 150 listings
    listings_url: str = (
        "https://csfloat.com/api/v1/listings?sort_by=most_recent&limit=50"
    )

    # ── Execution / balance settings ──────────────────────────────────────────
    buy_url_template: str = "https://csfloat.com/api/v1/listings/{listing_id}/buy"
    me_url: str = "https://csfloat.com/api/v1/me"
    rate_limit_pause: float = 60.0
    balance_refresh_interval: float = 300.0   # re-fetch balance every 5 minutes
    max_seen_ids: int = 10_000

    def auth_header_value(self) -> str | None:
        """
        Returns the Authorization header value, or None for cookie-only sessions.

        CSFloat browser sessions send no Authorization header — cookies suffice.
        Developer API key (from csfloat.com/developer): sent as-is.
        JWT session token (starts with 'eyJ'): sent with 'Bearer ' prefix.
        """
        if not self.bearer_token:
            return None
        if self.bearer_token.startswith("eyJ"):
            return f"Bearer {self.bearer_token}"
        return self.bearer_token

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls) -> "BotConfig":
        load_dotenv()
        token = os.getenv("CSFLOAT_BEARER_TOKEN", "").strip()
        cookies = os.getenv("CSFLOAT_COOKIES", "").strip()
        if not cookies:
            raise ValueError(
                "CSFLOAT_COOKIES is not set — copy .env.example to .env and fill it in"
            )
        if token:
            logger.info("Auth: using token (%s...)", token[:8])
        else:
            logger.info("Auth: cookie-only mode")
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        tg_chat  = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if tg_token and tg_chat:
            logger.info("Telegram notifications: enabled (chat_id=%s)", tg_chat)
        else:
            logger.info("Telegram notifications: disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)")
        return cls(
            bearer_token=token,
            cookies=cookies,
            min_profit_margin=MIN_PROFIT_MARGIN,
            min_volume=MIN_VOLUME,
            min_fair_value=MIN_FAIR_VALUE,
            min_item_price=MIN_ITEM_PRICE,
            max_item_price=MAX_ITEM_PRICE,
            telegram_bot_token=tg_token,
            telegram_chat_id=tg_chat,
        )


# ── Telegram Notifier ─────────────────────────────────────────────────────────


class TelegramNotifier:
    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str, session: aiohttp.ClientSession) -> None:
        self._token   = token
        self._chat_id = chat_id
        self._session = session
        self._enabled = bool(token and chat_id)

    async def _post(self, text: str) -> None:
        if not self._enabled:
            return
        url = self._API.format(token=self._token)
        try:
            async with self._session.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram notification failed (HTTP %d): %s", resp.status, body[:200])
        except Exception as exc:
            logger.warning("Telegram send error: %s", exc)

    async def send_startup(self, config: "BotConfig", balance: int) -> None:
        text = (
            f"🤖 <b>CSFloat Bot Başladı</b>\n\n"
            f"⚙️ Margin: <b>{config.min_profit_margin*100:.0f}%</b>  |  "
            f"Vol≥{config.min_volume}  |  Fair≥{config.min_fair_value}¢\n"
            f"💰 Bakiye: <b>${balance/100:.2f}</b>\n"
            f"📡 Poll: {config.poll_interval}s  |  {config.poll_pages} sayfa"
        )
        await self._post(text)

    async def send_buy(self, listing: dict, balance_left: int) -> None:
        name      = listing["item_name"]
        price     = listing["price"]
        fair      = listing.get("fair_value") or 0
        float_val = listing.get("float_value")
        discount  = (1 - price / fair) * 100 if fair else 0.0
        float_str = f"{float_val:.6f}" if float_val is not None else "n/a"
        text = (
            f"🛒 <b>SATIN ALINDI</b>\n\n"
            f"📦 {name}\n"
            f"💰 Fiyat: <b>${price/100:.2f}</b>  |  Fair: ${fair/100:.2f}  |  <b>-{discount:.1f}%</b>\n"
            f"🔬 Float: {float_str}\n"
            f"💼 Kalan bakiye: <b>${balance_left/100:.2f}</b>"
        )
        await self._post(text)

    async def send_error(self, message: str) -> None:
        text = f"🚨 <b>BOT HATASI</b>\n\n{message}"
        await self._post(text)


# ── Decision Engine ───────────────────────────────────────────────────────────


class DecisionEngine:
    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def extract_listing_data(self, raw: dict) -> dict | None:
        """
        Normalise a raw API listing dict into a canonical form.

        Relevant CSFloat listing fields:
          {
            "id":    "uuid",
            "price": 1200,
            "item":  {
              "market_hash_name": "AK-47 | ...",
              "float_value":      0.153,
            },
            "reference": {
              "predicted_price": 1450,
              "base_price":      1400,
              "quantity":        146,   ← active listing count (liquidity proxy)
            }
          }
        Returns None for malformed payloads.
        """
        try:
            listing_id: str = str(raw["id"])
            price: int = int(raw["price"])

            item: dict = raw["item"]
            item_name: str = str(item["market_hash_name"])
            float_value: float | None = (
                float(item["float_value"])
                if item.get("float_value") is not None
                else None
            )

            ref: dict = raw.get("reference") or {}
            predicted = ref.get("predicted_price")
            base = ref.get("base_price")
            fair_value: int | None = None
            if predicted and int(predicted) > 0:
                fair_value = int(predicted)
            elif base and int(base) > 0:
                fair_value = int(base)

            volume: int = int(ref.get("quantity") or 0)

            return {
                "listing_id": listing_id,
                "item_name": item_name,
                "price": price,
                "float_value": float_value,
                "fair_value": fair_value,
                "volume": volume,
            }
        except (KeyError, TypeError, ValueError):
            return None

    def should_buy(self, listing: dict) -> tuple[bool, str]:
        item_name: str = listing["item_name"]
        price: int = listing["price"]
        fair_value: int | None = listing.get("fair_value")
        volume: int = listing.get("volume", 0)

        # 1. Blacklist
        if item_name in BLACKLIST:
            return False, "blacklisted"

        # 2. Price range
        if price < self.config.min_item_price:
            return False, f"price {price}¢ below minimum {self.config.min_item_price}¢"
        if price > self.config.max_item_price:
            return False, f"price {price}¢ above cap {self.config.max_item_price}¢"

        # 3. Fair value availability
        if fair_value is None or fair_value < self.config.min_fair_value:
            return False, f"fair value {fair_value}¢ below minimum {self.config.min_fair_value}¢"

        # 4. Volume / liquidity
        if volume < self.config.min_volume:
            return False, f"volume {volume} below minimum {self.config.min_volume}"

        # 5. Profit margin
        max_buy = int(fair_value * (1 - self.config.min_profit_margin))
        if price > max_buy:
            discount_pct = (1 - price / fair_value) * 100
            return (
                False,
                f"price {price}¢ only {discount_pct:.1f}% below fair {fair_value}¢ "
                f"(need ≥{self.config.min_profit_margin * 100:.0f}%)",
            )

        discount_pct = (1 - price / fair_value) * 100
        return True, f"{discount_pct:.1f}% below fair value {fair_value}¢  vol={volume}"


# ── Execution Module ──────────────────────────────────────────────────────────


class ExecutionModule:
    def __init__(
        self,
        config: BotConfig,
        session: aiohttp.ClientSession,
        notifier: TelegramNotifier,
    ) -> None:
        self.config = config
        self.session = session
        self._notifier = notifier
        self._bought_ids: set[str] = set()
        self._rate_limited_until: float = 0.0
        self._balance: int = 0
        self._balance_loaded: bool = False
        self._last_balance_refresh: float = 0.0

    async def refresh_balance(self) -> bool:
        headers: dict[str, str] = {
            "Cookie": self.config.cookies,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": self._UA,
            "Referer": "https://csfloat.com/",
        }
        auth = self.config.auth_header_value()
        if auth:
            headers["Authorization"] = auth
        try:
            async with self.session.get(self.config.me_url, headers=headers) as resp:
                if resp.status in (401, 403):
                    logger.critical("Balance fetch AUTH FAILURE (HTTP %d)", resp.status)
                    return False
                if resp.status != 200:
                    logger.warning("Balance fetch unexpected HTTP %d", resp.status)
                    return False
                data = await resp.json()
            # Balance may sit at top level or nested under "user"
            balance = data.get("balance") or (data.get("user") or {}).get("balance")
            if balance is None:
                logger.warning(
                    "balance field not found in /api/v1/me — available keys: %s", list(data.keys())
                )
                return False
            self._balance = int(balance)
            self._balance_loaded = True
            self._last_balance_refresh = asyncio.get_event_loop().time()
            logger.info("Balance: $%.2f (%d¢)", self._balance / 100, self._balance)
            return True
        except aiohttp.ClientError as exc:
            logger.warning("Balance fetch error: %s", exc)
            return False

    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Cookie": self.config.cookies,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": self._UA,
            "Referer": "https://csfloat.com/",
            "Origin": "https://csfloat.com",
        }
        auth = self.config.auth_header_value()
        if auth:
            headers["Authorization"] = auth
        return headers

    async def buy(self, listing: dict) -> bool:
        listing_id: str = listing["listing_id"]
        price: int = listing["price"]

        if listing_id in self._bought_ids:
            logger.debug("Already bought listing_id=%s — skipping", listing_id)
            return False

        now = asyncio.get_event_loop().time()
        if now < self._rate_limited_until:
            remaining = self._rate_limited_until - now
            logger.debug("Rate-limit cooldown (%.1fs remaining) — skipping", remaining)
            return False

        # Refresh balance if not loaded or stale
        if not self._balance_loaded or (now - self._last_balance_refresh) > self.config.balance_refresh_interval:
            await self.refresh_balance()

        # Balance gate: skip if insufficient funds
        if self._balance_loaded and self._balance < price:
            logger.warning(
                "Insufficient balance: have $%.2f, need $%.2f for %r — skipping",
                self._balance / 100, price / 100, listing["item_name"],
            )
            return False

        url = self.config.buy_url_template.format(listing_id=listing_id)
        try:
            async with self.session.post(url, headers=self._build_headers()) as resp:
                return await self._handle_response(resp, listing)
        except aiohttp.ClientError as exc:
            logger.error("HTTP client error buying listing_id=%s: %s", listing_id, exc)
            return False
        except Exception as exc:
            logger.error("Unexpected error buying listing_id=%s: %s", listing_id, exc, exc_info=True)
            return False

    async def _handle_response(self, resp: aiohttp.ClientResponse, listing: dict) -> bool:
        listing_id = listing["listing_id"]
        status = resp.status

        if status in (200, 201):
            self._bought_ids.add(listing_id)
            self._balance = max(0, self._balance - listing["price"])
            logger.info(
                "BOUGHT  item=%r  price=%d¢  fair=%s  %s  float=%s  listing_id=%s  balance_left=$%.2f",
                listing["item_name"],
                listing["price"],
                f"{listing['fair_value']}¢" if listing.get("fair_value") else "n/a",
                listing.get("buy_reason", ""),
                f"{listing['float_value']:.6f}" if listing.get("float_value") is not None else "n/a",
                listing_id,
                self._balance / 100,
            )
            await self._notifier.send_buy(listing, self._balance)
            return True

        body = await resp.text()

        if status == 429:
            logger.warning(
                "Rate limited (HTTP 429). Pausing %.0fs. listing_id=%s",
                self.config.rate_limit_pause, listing_id,
            )
            self._rate_limited_until = (
                asyncio.get_event_loop().time() + self.config.rate_limit_pause
            )
            return False

        if status in (401, 403):
            logger.critical(
                "AUTH FAILURE (HTTP %d) — credentials expired. Update .env and restart. body=%s",
                status, body[:300],
            )
            await self._notifier.send_error(
                f"🔐 Satın alma auth hatası (HTTP {status})\n"
                ".env dosyasını güncelle ve botu yeniden başlat."
            )
            return False

        logger.warning("Unexpected HTTP %d for listing_id=%s. body=%s", status, listing_id, body[:300])
        return False


# ── Market Poller ─────────────────────────────────────────────────────────────


class MarketPoller:
    """
    Polls GET /api/v1/listings?sort_by=most_recent every poll_interval seconds.

    First poll: seeds the seen-set (no buys) to ignore pre-existing listings.
    Subsequent polls: calls on_listing for every newly appeared listing.

    Seen-ID deduplication uses a fixed-capacity sliding window so memory stays
    bounded over days of continuous operation.
    """

    def __init__(
        self,
        config: BotConfig,
        session: aiohttp.ClientSession,
        on_listing: Callable[[dict], Awaitable[None]],
        notifier: TelegramNotifier,
    ) -> None:
        self.config = config
        self.session = session
        self.on_listing = on_listing
        self._notifier = notifier
        self._running = True
        self._seen_ids: set[str] = set()
        self._seen_queue: deque[str] = deque(maxlen=config.max_seen_ids)
        self._initialized = False
        self._consecutive_429s: int = 0

    async def run(self) -> None:
        logger.info(
            "Market poller started — interval=%.1fs  margin=%.0f%%  "
            "vol≥%d  fair≥%d¢  price %d-%d¢",
            self.config.poll_interval,
            self.config.min_profit_margin * 100,
            self.config.min_volume,
            self.config.min_fair_value,
            self.config.min_item_price,
            self.config.max_item_price,
        )
        while self._running:
            try:
                await self._poll()
            except aiohttp.ClientError as exc:
                logger.warning("Poll network error: %s", exc)
            except Exception as exc:
                logger.error("Poll unexpected error: %s", exc, exc_info=True)
            if self._running:
                await asyncio.sleep(self.config.poll_interval)

    async def _poll(self) -> None:
        headers: dict[str, str] = {
            "Cookie": self.config.cookies,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://csfloat.com/",
            "Origin": "https://csfloat.com",
        }
        auth = self.config.auth_header_value()
        if auth:
            headers["Authorization"] = auth

        cursor: str | None = None
        new_count = 0
        total_seeded = 0

        for page_num in range(self.config.poll_pages):
            url = self.config.listings_url
            if cursor:
                url += f"&cursor={cursor}"

            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 429:
                    self._consecutive_429s += 1
                    pause = min(60 * (2 ** (self._consecutive_429s - 1)), 600)
                    logger.warning(
                        "Poll rate limited (429) — waiting %.0fs (art arda %d)",
                        pause, self._consecutive_429s,
                    )
                    await self._notifier.send_error(
                        f"⏳ Rate limit (429) — {pause:.0f}s bekleniyor "
                        f"(art arda {self._consecutive_429s}. kez)"
                    ) if self._consecutive_429s == 1 else None
                    await asyncio.sleep(pause)
                    return
                if resp.status in (401, 403):
                    logger.critical("Poll AUTH FAILURE (HTTP %d) — update .env and restart", resp.status)
                    await self._notifier.send_error(
                        f"🔐 Auth başarısız (HTTP {resp.status})\n"
                        ".env dosyasını güncelle ve botu yeniden başlat."
                    )
                    return
                resp.raise_for_status()
                payload = await resp.json()

            # Handle both bare-list and {"data": [...], "cursor": "..."} shapes
            if isinstance(payload, list):
                listings: list[dict] = payload
                next_cursor: str | None = None
            else:
                listings = payload.get("data") or payload.get("listings") or []
                next_cursor = payload.get("cursor")

            found_new = False
            for raw in listings:
                lid = str(raw.get("id", ""))
                if not lid or lid in self._seen_ids:
                    continue
                found_new = True
                self._mark_seen(lid)
                if not self._initialized:
                    total_seeded += 1
                    continue
                new_count += 1
                try:
                    await self.on_listing(raw)
                except Exception as exc:
                    logger.error("Error processing listing %s: %s", lid, exc, exc_info=True)

            # Stop paging: no cursor returned, or page 1+ had nothing new (caught up)
            if not next_cursor or (not found_new and page_num > 0):
                break
            cursor = next_cursor

        self._consecutive_429s = 0

        if not self._initialized:
            self._initialized = True
            logger.info(
                "Poller initialised — %d existing listing(s) seeded (%d page(s))",
                total_seeded, page_num + 1,
            )
        elif new_count:
            logger.debug("Poll: %d new listing(s) evaluated across %d page(s)", new_count, page_num + 1)

    def _mark_seen(self, listing_id: str) -> None:
        if len(self._seen_queue) == self._seen_queue.maxlen:
            self._seen_ids.discard(self._seen_queue[0])
        self._seen_queue.append(listing_id)
        self._seen_ids.add(listing_id)

    def stop(self) -> None:
        self._running = False


# ── Trading Bot (Orchestrator) ────────────────────────────────────────────────


class TradingBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.engine = DecisionEngine(config)
        self._poller: MarketPoller | None = None
        self._executor: ExecutionModule | None = None

    async def run(self) -> None:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            notifier = TelegramNotifier(
                self.config.telegram_bot_token,
                self.config.telegram_chat_id,
                session,
            )
            self._executor = ExecutionModule(self.config, session, notifier)
            await self._executor.refresh_balance()
            await notifier.send_startup(self.config, self._executor._balance)
            self._poller = MarketPoller(self.config, session, self._on_listing, notifier)
            try:
                await self._poller.run()
            except Exception as exc:
                await notifier.send_error(f"Beklenmeyen hata, bot durdu:\n<code>{exc}</code>")
                raise

    async def _on_listing(self, raw: dict) -> None:
        listing = self.engine.extract_listing_data(raw)
        if listing is None:
            return

        should, reason = self.engine.should_buy(listing)
        if not should:
            logger.debug(
                "Skip %r  price=%d¢  fair=%s  vol=%s — %s",
                listing.get("item_name"),
                listing.get("price", 0),
                f"{listing.get('fair_value')}¢" if listing.get("fair_value") else "n/a",
                listing.get("volume", "?"),
                reason,
            )
            return

        listing["buy_reason"] = reason
        logger.info(
            "Match: item=%r  price=%d¢  fair=%d¢  %s  listing_id=%s — buying",
            listing["item_name"],
            listing["price"],
            listing["fair_value"],
            reason,
            listing["listing_id"],
        )
        assert self._executor is not None
        await self._executor.buy(listing)

    def stop(self) -> None:
        if self._poller:
            self._poller.stop()


# ── Entry Point ───────────────────────────────────────────────────────────────


async def main() -> None:
    setup_logging()

    try:
        config = BotConfig.from_env()
    except ValueError as exc:
        print(f"[STARTUP ERROR] {exc}", file=sys.stderr)
        raise

    bot = TradingBot(config)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown() -> None:
        logger.info("Shutdown signal received — stopping bot gracefully")
        bot.stop()
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    logger.info(
        "Bot starting — auto-discovery mode  margin=%.0f%%  vol≥%d  fair≥%d¢  price %d-%d¢  blacklist=%d",
        config.min_profit_margin * 100,
        config.min_volume,
        config.min_fair_value,
        config.min_item_price,
        config.max_item_price,
        len(BLACKLIST),
    )

    await asyncio.gather(bot.run(), stop_event.wait(), return_exceptions=True)

    logger.info("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
