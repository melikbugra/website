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
import json
import logging
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote as url_quote

import aiohttp
from dotenv import load_dotenv

from momentum import MomentumStrategy
from pricedb import PriceDB


# ── Filters ───────────────────────────────────────────────────────────────────

# Minimum profit margin: buy only when listing_price <= base_price * (1 - margin)
# 0.06 = need 6% discount → liste fiyatı +8% ile ~14% gross, ~12% net (CSFloat %2 sonrası)
# Düşük marj + yüksek hacim = "küçük küçük arttır" stratejisi
MIN_PROFIT_MARGIN: float = 0.06

_WEAR_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "Factory New": (0.00, 0.07),
    "Minimal Wear": (0.07, 0.15),
    "Field-Tested": (0.15, 0.38),
    "Well-Worn": (0.38, 0.45),
    "Battle-Scarred": (0.45, 1.00),
}


def _to_bool(s: str) -> bool:
    """/set anahtarları için on/off ayrıştırıcı (bool('0') tuzağını önler)."""
    v = s.strip().lower()
    if v in ("1", "true", "on", "açık", "acik", "yes", "evet"):
        return True
    if v in ("0", "false", "off", "kapalı", "kapali", "no", "hayır", "hayir"):
        return False
    raise ValueError("on/off")


def _strategy_badge(strategy: str | None) -> str:
    """Bildirimlerde gösterilecek strateji etiketi."""
    return "📉 Mean-reversion" if strategy == "momentum" else "🎯 Sniper"


def _float_tier_position(float_val: float, item_name: str) -> float | None:
    """Tier içindeki float pozisyonu: 0.0=en iyi (düşük float), 1.0=en kötü."""
    for wear, (lo, hi) in _WEAR_FLOAT_BOUNDS.items():
        if wear in item_name:
            return (float_val - lo) / (hi - lo)
    return None


# Minimum active CSFloat listings for an item (reference.quantity)
# 200+ = sadece çok likit item'lar (likit olmazsa satış için günler bekleyebilirsin)
MIN_VOLUME: int = 200

# Minimum base_price to be considered (cents)
# Filters out junk items that have no real reference data
MIN_FAIR_VALUE: int = 20  # $0.20

# Price range gate — never buy outside this band regardless of other filters
MIN_ITEM_PRICE: int = 100  # $0.10 — below this there's no profit potential
MAX_ITEM_PRICE: int = 600  # $5.00 — safety cap per item

# Minimum balance to keep in reserve (cents) — bot won't buy if it would drop below this
# 500 = always keep $5.00 untouched
MIN_BALANCE_RESERVE: int = 0

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

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "bot.log"
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
    poll_interval_idle: float = 90.0  # sniper kapalıyken poller bu kadar yavaş döner
    poll_pages: int = 1  # page 1 always has the newest listings
    listings_url: str = (
        "https://csfloat.com/api/v1/listings?sort_by=most_recent&limit=50"
    )

    # ── Execution / balance settings ──────────────────────────────────────────
    buy_url: str = "https://csfloat.com/api/v1/listings/buy"
    me_url: str = "https://csfloat.com/api/v1/me"
    rate_limit_pause: float = 60.0
    balance_refresh_interval: float = 300.0  # re-fetch balance every 5 minutes
    min_balance_reserve: int = 500  # never spend below this balance (cents)
    max_seen_ids: int = 10_000

    # ── Portfolio management (post-buy lifecycle) ─────────────────────────────
    inventory_url: str = "https://csfloat.com/api/v1/me/inventory"
    # Stall is per-user: GET /api/v1/users/{steam_id}/stall?limit=40
    stall_url_template: str = (
        "https://csfloat.com/api/v1/users/{steam_id}/stall?limit=100"
    )
    listings_create_url: str = "https://csfloat.com/api/v1/listings"  # POST to create
    data_dir: str = "data"
    trade_hold_buffer_days: int = 8  # 7-day Steam + 1-day safety

    # Listing fiyat fazları (markup over base_price_at_buy)
    phase_1_markup: float = 1.08  # gün 0-3: yüksek marj, sabırlı
    phase_2_markup: float = 1.05  # gün 3-7: ılımlı
    phase_3_undercut_cents: int = 1  # gün 7-14: en ucuzdan 1¢ altında
    min_listing_markup: float = 1.03  # her zaman alımın en az %3 üstünde
    phase_1_duration_days: int = 3
    phase_2_duration_days: int = 4  # day 3 → 7
    phase_3_duration_days: int = 7  # day 7 → 14

    # Worker intervals (saniye)
    listing_worker_interval: float = 3600.0  # 1 saat
    sales_worker_interval: float = 3600.0  # 1 saat
    repricing_worker_interval: float = 86400.0  # 1 gün
    inventory_resolve_delay: float = 30.0  # buy → inventory query gecikme
    inventory_resolve_retries: int = 6  # fast-path: 6 × 30s = 3 dakika
    pending_trade_retry_interval: float = 300.0  # slow-path: 5 dakikada bir tekrar dene
    pending_trade_max_age_hours: int = (
        24  # 24 saat sonra umudu kes (CSFloat zaten refund eder)
    )
    csfloat_seller_fee: float = 0.02  # CSFloat satıcı komisyonu (%2)
    float_low_threshold: float = 0.33  # tier içinde alt %33 = "düşük float"
    float_low_premium: float = 1.10  # düşük float için liste fiyatı +%10

    # ── Strateji anahtarları (canlı /set ile aç/kapa) ─────────────────────────
    sniper_enabled: bool = True  # eski "en ucuzun altını kap" stratejisi
    mr_enabled: bool = True  # mean-reversion stratejisi

    # ── Mean-reversion strateji (sniper'ın YANINDA, ayrı bütçe) ───────────────
    steam_cookies: str = ""  # steamLoginSecure vb. (boşsa salt-CSFloat mod)
    mr_budget_cents: int = 5000  # bu stratejinin ayrı kasa tavanı ($50)
    mr_lookback_days: int = 60  # z-score baseline penceresi
    mr_trend_days: int = 90  # düşen-bıçak filtresi penceresi
    mr_min_points: int = 10  # istatistik için min veri noktası
    mr_z_entry: float = -1.5  # bu kadar SS altı = "geçici dip" → al
    mr_min_volume: int = 20  # günlük likidite kapısı (çıkış garantisi)
    mr_slope_min: float = (
        -0.003
    )  # normalize trend eğimi ≥ bu (≈-0.3%/gün; düz≈0 geçer, çöküş reddedilir)
    mr_min_profit_margin: float = 0.10  # baseline'a dönüşte beklenen net kâr
    mr_tick_interval: float = (
        60.0  # watchlist'te item başına işleme aralığı (burst yok)
    )
    mr_warmup_interval: float = 1800.0  # watchlist boşsa bu kadar bekle
    mr_watchlist_max: int = 60  # izlenen item üst sınırı (rate-limit dostu)

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
        steam_cookies = os.getenv("STEAM_COOKIES", "").strip()
        if steam_cookies:
            logger.info("Mean-reversion: Steam history enabled (deep bootstrap)")
        else:
            logger.info(
                "Mean-reversion: STEAM_COOKIES not set — CSFloat-only mode "
                "(no deep history; signals after ~3-4 weeks of self-collected data)"
            )
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        tg_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if tg_token and tg_chat:
            logger.info("Telegram notifications: enabled (chat_id=%s)", tg_chat)
        else:
            logger.info(
                "Telegram notifications: disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)"
            )
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
            min_balance_reserve=MIN_BALANCE_RESERVE,
            steam_cookies=steam_cookies,
        )


# ── Bought-items tracker (portfolio state) ────────────────────────────────────


@dataclass
class BoughtItem:
    """One item the bot bought — tracked through its lifecycle until sold."""

    asset_id: str = ""  # Steam inventory asset id (empty until resolved)
    contract_id_buy: str = ""  # CSFloat listing id we bought FROM
    item_name: str = ""
    buy_price: int = 0  # cents we paid
    base_price_at_buy: int = 0  # market base_price when we bought (for repricing)
    bought_at: str = ""  # ISO-8601 UTC
    trade_unlock_at: str = ""  # ISO-8601 UTC (Steam tradable_at + safety)
    status: str = (
        "pending_asset"  # pending_asset | locked | listed | sold | skipped | failed
    )
    listing_id_sell: str | None = None  # CSFloat listing id we created when selling
    list_price: int | None = None
    listed_at: str | None = None
    current_phase: int | None = None  # 1, 2, 3 (or None when not listed)
    sold_at: str | None = None
    net_profit: int | None = None  # cents, after CSFloat fee
    error_count: int = 0
    last_error: str = ""
    float_value: float | None = None
    strategy: str = "sniper"  # "sniper" | "momentum"
    target_price: int | None = None  # momentum: baseline-at-buy = satış hedefi

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BoughtItem":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class BoughtItemsTracker:
    """
    Persistent JSON store for items the bot has bought.

    File layout: <data_dir>/bought_items.json
      Keyed by asset_id when resolved, otherwise by "pending:<contract_id>".
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._file = data_dir / "bought_items.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._items: dict[str, BoughtItem] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            logger.info(
                "BoughtItems: no existing file at %s — starting fresh", self._file
            )
            return
        try:
            with self._file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self._items[k] = BoughtItem.from_dict(v)
            logger.info(
                "BoughtItems: loaded %d items from %s", len(self._items), self._file
            )
        except Exception as exc:
            logger.error("BoughtItems: failed to load %s: %s", self._file, exc)

    async def _save(self) -> None:
        tmp = self._file.with_suffix(".tmp")
        payload = {k: v.to_dict() for k, v in self._items.items()}
        await asyncio.to_thread(self._write_json, tmp, payload)
        tmp.replace(self._file)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def add_pending(
        self,
        contract_id: str,
        item_name: str,
        buy_price: int,
        base_price: int,
        float_value: float | None = None,
        strategy: str = "sniper",
        target_price: int | None = None,
    ) -> str:
        """Insert a freshly-bought item before asset_id is known. Returns lookup key."""
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            key = f"pending:{contract_id}"
            self._items[key] = BoughtItem(
                contract_id_buy=contract_id,
                item_name=item_name,
                buy_price=buy_price,
                base_price_at_buy=base_price,
                bought_at=now,
                status="pending_asset",
                float_value=float_value,
                strategy=strategy,
                target_price=target_price,
            )
            await self._save()
            return key

    async def resolve_asset(
        self, pending_key: str, asset_id: str, trade_unlock_at: str
    ) -> None:
        """Promote a pending item to locked by attaching asset_id + unlock time."""
        async with self._lock:
            item = self._items.pop(pending_key, None)
            if item is None:
                return
            item.asset_id = asset_id
            item.trade_unlock_at = trade_unlock_at
            item.status = "locked"
            self._items[asset_id] = item
            await self._save()

    async def rekey_asset(self, old_key: str, new_asset_id: str) -> None:
        """Re-key an item under a new asset_id (Steam reassigns ids on unlock)."""
        async with self._lock:
            item = self._items.pop(old_key, None)
            if item is None:
                return
            item.asset_id = new_asset_id
            self._items[new_asset_id] = item
            await self._save()

    async def update(self, key: str, **changes) -> None:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return
            for k, v in changes.items():
                setattr(item, k, v)
            await self._save()

    async def mark_skipped(self, asset_id: str) -> bool:
        async with self._lock:
            item = self._items.get(asset_id)
            if item is None:
                return False
            item.status = "skipped"
            await self._save()
            return True

    def get(self, key: str) -> BoughtItem | None:
        return self._items.get(key)

    def all(self) -> list[BoughtItem]:
        return list(self._items.values())

    def by_status(self, status: str) -> list[BoughtItem]:
        return [it for it in self._items.values() if it.status == status]

    def total_invested_in_active(self) -> int:
        """Cents currently tied up in locked/listed items."""
        return sum(
            it.buy_price
            for it in self._items.values()
            if it.status in ("pending_asset", "locked", "listed")
        )

    def realized_profit_since(self, since: datetime | None = None) -> int:
        """Sum of net_profit for sold items, optionally since a given time."""
        total = 0
        for it in self._items.values():
            if it.status != "sold" or it.net_profit is None or not it.sold_at:
                continue
            if since is not None:
                try:
                    sold_dt = datetime.fromisoformat(it.sold_at)
                    if sold_dt < since:
                        continue
                except ValueError:
                    continue
            total += it.net_profit
        return total


# ── Telegram Notifier ─────────────────────────────────────────────────────────


class TelegramNotifier:
    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self, token: str, chat_id: str, session: aiohttp.ClientSession
    ) -> None:
        self._token = token
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
                    logger.warning(
                        "Telegram notification failed (HTTP %d): %s",
                        resp.status,
                        body[:200],
                    )
        except Exception as exc:
            logger.warning("Telegram send error: %s", exc)

    async def send_startup(self, config: "BotConfig", balance: int) -> None:
        text = (
            f"🤖 <b>CSFloat Bot Başladı</b>\n\n"
            f"⚙️ Margin: <b>{config.min_profit_margin * 100:.0f}%</b>  |  "
            f"Vol≥{config.min_volume}  |  Fair≥{config.min_fair_value}¢\n"
            f"💰 Bakiye: <b>${balance / 100:.2f}</b>\n"
            f"📡 Poll: {config.poll_interval}s  |  {config.poll_pages} sayfa"
        )
        await self._post(text)

    async def send_buy(self, listing: dict, balance_left: int) -> None:
        name = listing["item_name"]
        price = listing["price"]
        fair = listing.get("fair_value") or 0
        float_val = listing.get("float_value")
        discount = (1 - price / fair) * 100 if fair else 0.0
        float_str = f"{float_val:.6f}" if float_val is not None else "n/a"
        badge = _strategy_badge(listing.get("strategy"))
        reason = listing.get("buy_reason", "")
        fair_label = "Baseline" if listing.get("strategy") == "momentum" else "Fair"
        text = (
            f"🛒 <b>SATIN ALINDI</b> · {badge}\n\n"
            f"📦 {name}\n"
            f"💰 Fiyat: <b>${price / 100:.2f}</b>  |  {fair_label}: ${fair / 100:.2f}  |  <b>-{discount:.1f}%</b>\n"
            f"🔬 Float: {float_str}\n"
            f"🧮 Neden: {reason}\n"
            f"💼 Kalan bakiye: <b>${balance_left / 100:.2f}</b>"
        )
        await self._post(text)

    async def send_match(self, listing: dict) -> None:
        name = listing["item_name"]
        price = listing["price"]
        fair = listing.get("fair_value") or 0
        float_val = listing.get("float_value")
        discount = (1 - price / fair) * 100 if fair else 0.0
        float_str = f"{float_val:.6f}" if float_val is not None else "n/a"
        vol = listing.get("volume", 0)
        is_mr = listing.get("strategy") == "momentum"
        badge = _strategy_badge(listing.get("strategy"))
        fair_label = "Baseline" if is_mr else "Fair"
        reason = listing.get("buy_reason", "")
        reason_line = f"\n🧮 Neden: {reason}" if reason else ""
        text = (
            f"🎯 <b>FIRSAT BULUNDU</b> · {badge}\n\n"
            f"📦 {name}\n"
            f"💰 Fiyat: <b>${price / 100:.2f}</b>  |  {fair_label}: ${fair / 100:.2f}  |  <b>-{discount:.1f}%</b>\n"
            f"🔬 Float: {float_str}  |  Vol: {vol}{reason_line}\n"
            f"⏳ Satın alınıyor..."
        )
        await self._post(text)

    async def send_momentum_summary(
        self, scanned: int, signals: int, bought: int, used: int, budget: int
    ) -> None:
        text = (
            f"📊 <b>MEAN-REVERSION TARAMA</b>\n\n"
            f"🔍 Taranan: {scanned}  |  Sinyal: {signals}  |  Alım: {bought}\n"
            f"💼 Momentum kasa: <b>${used / 100:.2f}</b> / ${budget / 100:.2f}"
        )
        await self._post(text)

    async def send_buy_failed(self, listing: dict, reason: str) -> None:
        name = listing["item_name"]
        price = listing["price"]
        text = (
            f"❌ <b>SATIN ALINAMADI</b>\n\n"
            f"📦 {name}\n"
            f"💰 Fiyat: ${price / 100:.2f}\n"
            f"⚠️ {reason}"
        )
        await self._post(text)

    async def send_error(self, message: str) -> None:
        text = f"🚨 <b>BOT HATASI</b>\n\n{message}"
        await self._post(text)

    async def send_listed(self, item: BoughtItem, price: int, phase: int) -> None:
        expected_net = int(price * 0.98) - item.buy_price
        badge = _strategy_badge(item.strategy)
        float_line = ""
        if item.float_value is not None:
            pos = _float_tier_position(item.float_value, item.item_name)
            premium_str = " (+prim)" if (pos is not None and pos < 0.33) else ""
            float_line = f"\n🔬 Float: {item.float_value:.6f}{premium_str}"
        # Satış fiyatının nereden geldiğini açıkla
        if item.strategy == "momentum" and item.target_price:
            logic_line = (
                f"\n🎯 Hedef: baseline ${item.target_price / 100:.2f} (toparlamada sat)"
            )
        else:
            logic_line = ""
        text = (
            f"📤 <b>LİSTELENDİ</b> (Faz {phase}) · {badge}\n\n"
            f"📦 {item.item_name}\n"
            f"💰 Alım: ${item.buy_price / 100:.2f}  →  Liste: <b>${price / 100:.2f}</b>\n"
            f"📈 Beklenen net kâr: <b>${expected_net / 100:+.2f}</b>"
            f"{logic_line}"
            f"{float_line}"
        )
        await self._post(text)

    async def send_repriced(
        self, item: BoughtItem, old_price: int, new_price: int, phase: int
    ) -> None:
        text = (
            f"🔄 <b>YENİDEN FİYATLANDI</b> (Faz {phase}) · {_strategy_badge(item.strategy)}\n\n"
            f"📦 {item.item_name}\n"
            f"💰 ${old_price / 100:.2f} → <b>${new_price / 100:.2f}</b>"
        )
        await self._post(text)

    async def send_sold(self, item: BoughtItem) -> None:
        net = item.net_profit or 0
        sign = "+" if net >= 0 else ""
        emoji = "✅" if net >= 0 else "⚠️"
        text = (
            f"{emoji} <b>SATILDI</b> · {_strategy_badge(item.strategy)}\n\n"
            f"📦 {item.item_name}\n"
            f"💰 Alım: ${item.buy_price / 100:.2f}  →  Satış: ${(item.list_price or 0) / 100:.2f}\n"
            f"📊 Net kâr: <b>{sign}${net / 100:.2f}</b>"
        )
        await self._post(text)

    async def send_stuck(self, item: BoughtItem, days_listed: int) -> None:
        text = (
            f"⚠️ <b>MANUEL İNCELEME GEREK</b>\n\n"
            f"📦 {item.item_name}\n"
            f"{days_listed} gündür satılmıyor. Bot artık dokunmuyor.\n"
            f"Liste fiyatı: ${(item.list_price or 0) / 100:.2f}  |  Alım: ${item.buy_price / 100:.2f}"
        )
        await self._post(text)

    async def send_idle(
        self, balance: int, pending: int, locked: int, listed: int
    ) -> None:
        text = (
            f"💤 <b>Bot Boşta</b>\n\n"
            f"Bakiye yeterli alım için yetersiz, locked/listed item'lar bekliyor.\n\n"
            f"💰 Bakiye: ${balance / 100:.2f}\n"
            f"⏳ Locked: {locked}  |  📤 Listed: {listed}  |  ⌛ Pending: {pending}\n\n"
            f"<i>Bot açık kalabilir veya kapatılabilir — fark etmez.</i>"
        )
        await self._post(text)

    async def send_asset_resolve_failed(self, item: BoughtItem) -> None:
        text = (
            f"⚠️ <b>ASSET ID BULUNAMADI</b>\n\n"
            f"📦 {item.item_name}\n"
            f"Bot envanterden bu item'ı eşleştiremedi. Manuel kontrol et.\n"
            f"contract_id: <code>{item.contract_id_buy}</code>"
        )
        await self._post(text)

    async def send_pending_resolved(self, item: BoughtItem, age_hours: float) -> None:
        text = (
            f"🤝 <b>TRADE KABUL EDİLDİ</b>\n\n"
            f"📦 {item.item_name}\n"
            f"⏱️ {age_hours:.1f} saat sonra satıcı kabul etti\n"
            f"⏳ Trade hold sayacı başladı, ~8 gün sonra otomatik listelenir."
        )
        await self._post(text)


# ── Telegram Command Handler ──────────────────────────────────────────────────


class TelegramCommandHandler:
    _UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"
    _SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"

    PARAMS: dict[str, tuple[str, type, str]] = {
        "margin": ("min_profit_margin", float, "ör. 0.20"),
        "volume": ("min_volume", int, "ör. 50"),
        "fair": ("min_fair_value", int, "ör. 20"),
        "min_price": ("min_item_price", int, "ör. 10 (¢)"),
        "max_price": ("max_item_price", int, "ör. 1000 (¢)"),
        "reserve": ("min_balance_reserve", int, "ör. 500 (¢ = $5.00)"),
        "sniper": ("sniper_enabled", _to_bool, "on/off"),
        "momentum": ("mr_enabled", _to_bool, "on/off"),
        "mr_budget": ("mr_budget_cents", int, "ör. 5000 (¢ = $50)"),
        "poll": ("poll_interval", float, "ör. 15 (s)"),
        "poll_idle": ("poll_interval_idle", float, "ör. 90 (s, sniper kapalıyken)"),
    }

    def __init__(
        self,
        token: str,
        chat_id: str,
        config: "BotConfig",
        session: aiohttp.ClientSession,
        tracker: "BoughtItemsTracker | None" = None,
        portfolio: "PortfolioManager | None" = None,
        executor: "ExecutionModule | None" = None,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._config = config
        self._session = session
        self._enabled = bool(token and chat_id)
        self._offset = 0
        self._tracker = tracker
        self._portfolio = portfolio
        self._executor = executor

    async def listen(self) -> None:
        if not self._enabled:
            return
        logger.info("Telegram command handler started")
        while True:
            try:
                updates = await self._get_updates()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("Command handler poll error: %s", exc)
                await asyncio.sleep(5)
                continue
            for update in updates:
                try:
                    await self._handle_update(update)
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.warning("Command handle error: %s", exc)

    async def _get_updates(self) -> list[dict]:
        url = (
            self._UPDATES_URL.format(token=self._token)
            + f"?timeout=30&offset={self._offset}"
        )
        timeout = aiohttp.ClientTimeout(total=40)
        async with self._session.get(url, timeout=timeout) as resp:
            data = await resp.json()
        updates: list[dict] = data.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    async def _handle_update(self, update: dict) -> None:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        if str(msg.get("chat", {}).get("id")) != self._chat_id:
            return
        text = msg.get("text", "").strip()
        if not text:
            return
        if text == "/status" or text.startswith("/status@"):
            await self._send_status()
        elif text == "/help" or text.startswith("/help@"):
            await self._send_help()
        elif text.startswith("/set "):
            await self._handle_set(text[5:].strip())
        elif text == "/portfolio" or text.startswith("/portfolio@"):
            await self._send_portfolio()
        elif text.startswith("/profit"):
            await self._send_profit(text)
        elif text == "/catchup" or text.startswith("/catchup@"):
            await self._handle_catchup()
        elif text.startswith("/skip "):
            await self._handle_skip(text[6:].strip())

    async def _handle_set(self, args: str) -> None:
        parts = args.split()
        if len(parts) != 2:
            await self._reply(
                "❌ Kullanım: <code>/set &lt;parametre&gt; &lt;değer&gt;</code>"
            )
            return
        key, val_str = parts[0].lower(), parts[1]
        if key not in self.PARAMS:
            keys = ", ".join(self.PARAMS)
            await self._reply(f"❌ Bilinmeyen parametre: <b>{key}</b>\nGeçerli: {keys}")
            return
        attr, typ, hint = self.PARAMS[key]
        try:
            new_val = typ(val_str)
        except ValueError:
            await self._reply(f"❌ Geçersiz değer: <b>{val_str}</b> ({hint})")
            return
        old_val = getattr(self._config, attr)
        setattr(self._config, attr, new_val)
        logger.info("Config updated via Telegram: %s %s → %s", key, old_val, new_val)
        await self._reply(f"✅ <b>{key}</b> güncellendi: {old_val} → {new_val}")

    async def _send_status(self) -> None:
        c = self._config
        text = (
            f"📊 <b>Bot Durumu</b>\n\n"
            f"<code>margin    = {c.min_profit_margin:.0%}</code>\n"
            f"<code>volume    = {c.min_volume}</code>\n"
            f"<code>fair      = {c.min_fair_value}¢</code>\n"
            f"<code>min_price = {c.min_item_price}¢  (${c.min_item_price / 100:.2f})</code>\n"
            f"<code>max_price = {c.max_item_price}¢  (${c.max_item_price / 100:.2f})</code>\n"
            f"<code>reserve   = {c.min_balance_reserve}¢  (${c.min_balance_reserve / 100:.2f})</code>\n"
            f"<code>sniper    = {'ON' if c.sniper_enabled else 'OFF'}</code>\n"
            f"<code>momentum  = {'ON' if c.mr_enabled else 'OFF'}  (budget ${c.mr_budget_cents / 100:.2f})</code>"
        )
        await self._reply(text)

    async def _send_help(self) -> None:
        text = (
            "📖 <b>Komutlar</b>\n\n"
            "<b>Durum:</b>\n"
            "/status — filtre değerleri\n"
            "/portfolio — locked/listed item'lar + yatırım\n"
            "/profit [7d|30d|all] — net kâr\n\n"
            "<b>Aksiyon:</b>\n"
            "/catchup — listing+satış+repricing döngüsünü hemen tetikle\n"
            "/skip &lt;asset_id&gt; — botun bir item'a dokunmasını engelle\n\n"
            "<b>Filtre güncelleme:</b>\n"
            "/set margin 0.20\n"
            "/set volume 50\n"
            "/set fair 20\n"
            "/set min_price 10\n"
            "/set max_price 1000\n"
            "/set reserve 500\n\n"
            "<b>Strateji aç/kapa:</b>\n"
            "/set sniper off — eski 'en ucuzun altını kap' stratejisi\n"
            "/set momentum on — mean-reversion stratejisi\n"
            "/set mr_budget 5000 — momentum kasası (¢)\n\n"
            "<i>Fiyatlar ¢ (cent) cinsinden — $1.00 = 100</i>"
        )
        await self._reply(text)

    async def _send_portfolio(self) -> None:
        if self._tracker is None:
            await self._reply("⚠️ Tracker hazır değil")
            return
        pending = self._tracker.by_status("pending_asset")
        locked = self._tracker.by_status("locked")
        listed = self._tracker.by_status("listed")
        invested = self._tracker.total_invested_in_active()
        expected_net = sum(
            int((it.list_price or 0) * (1 - self._config.csfloat_seller_fee))
            - it.buy_price
            for it in listed
        )

        lines: list[str] = [
            f"📊 <b>Portfolio</b>\n",
            f"💼 Aktif yatırım: <b>${invested / 100:.2f}</b>",
            f"⌛ Pending: {len(pending)}  |  ⏳ Locked: {len(locked)}  |  📤 Listed: {len(listed)}",
            f"📈 Beklenen net kâr (listedeki): <b>${expected_net / 100:+.2f}</b>",
        ]

        if locked:
            lines.append("\n<b>🔒 Kilitli (trade hold)</b>")
            for it in locked[:5]:
                unlock_str = it.trade_unlock_at[:10] if it.trade_unlock_at else "?"
                lines.append(
                    f"  • {it.item_name[:40]} — ${it.buy_price / 100:.2f} (açılır: {unlock_str})"
                )
            if len(locked) > 5:
                lines.append(f"  <i>… +{len(locked) - 5} adet</i>")

        if listed:
            lines.append("\n<b>📤 Listede</b>")
            for it in listed[:5]:
                lines.append(
                    f"  • {it.item_name[:40]} — alım ${it.buy_price / 100:.2f} → liste ${(it.list_price or 0) / 100:.2f} (Faz {it.current_phase or 1})"
                )
            if len(listed) > 5:
                lines.append(f"  <i>… +{len(listed) - 5} adet</i>")

        await self._reply("\n".join(lines))

    async def _send_profit(self, text: str) -> None:
        if self._tracker is None:
            await self._reply("⚠️ Tracker hazır değil")
            return
        arg = text.replace("/profit", "", 1).strip().lower() or "all"
        now = datetime.now(timezone.utc)
        since: datetime | None
        label: str
        if arg in ("7d", "7", "week"):
            since = now - timedelta(days=7)
            label = "Son 7 gün"
        elif arg in ("30d", "30", "month"):
            since = now - timedelta(days=30)
            label = "Son 30 gün"
        elif arg in ("all", "tum", "tüm"):
            since = None
            label = "Tüm zamanlar"
        else:
            await self._reply("❌ Kullanım: <code>/profit 7d|30d|all</code>")
            return

        net = self._tracker.realized_profit_since(since)
        sold_count = sum(
            1
            for it in self._tracker.by_status("sold")
            if since is None
            or (
                it.sold_at
                and PortfolioManager._parse_iso(it.sold_at)
                and PortfolioManager._parse_iso(it.sold_at) >= since
            )
        )
        sign = "+" if net >= 0 else ""
        await self._reply(
            f"💰 <b>{label}</b>\n\n"
            f"Net kâr: <b>{sign}${net / 100:.2f}</b>\n"
            f"Satılan item: {sold_count}"
        )

    async def _handle_catchup(self) -> None:
        if self._portfolio is None:
            await self._reply("⚠️ Portfolio manager hazır değil")
            return
        await self._reply("⏳ Catch-up döngüsü çalışıyor…")
        try:
            summary = await self._portfolio.run_catch_up()
        except Exception as exc:
            await self._reply(f"❌ Hata: <code>{exc}</code>")
            return
        await self._reply(
            f"✅ <b>Catch-up tamam</b>\n\n"
            f"🤝 Trade kabul edildi: {summary.get('resolved', 0)}\n"
            f"📤 Listelendi: {summary['listed']}\n"
            f"✅ Satıldı: {summary['sold']}\n"
            f"🔄 Yeniden fiyatlandı: {summary['repriced']}"
        )

    async def _handle_skip(self, asset_id: str) -> None:
        if self._tracker is None:
            await self._reply("⚠️ Tracker hazır değil")
            return
        if not asset_id:
            await self._reply("❌ Kullanım: <code>/skip &lt;asset_id&gt;</code>")
            return
        ok = await self._tracker.mark_skipped(asset_id)
        if ok:
            await self._reply(
                f"✅ <code>{asset_id}</code> skipped — bot artık bu item'a dokunmayacak"
            )
        else:
            await self._reply(f"❌ asset_id bulunamadı: <code>{asset_id}</code>")

    async def _reply(self, text: str) -> None:
        url = self._SEND_URL.format(token=self._token)
        try:
            async with self._session.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Telegram reply failed (HTTP %d): %s", resp.status, body[:200]
                    )
        except Exception as exc:
            logger.warning("Telegram reply error: %s", exc)


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
            # Skip auction listings — their current bid looks like a huge discount
            if raw.get("type") == "auction" or raw.get("is_auction") is True:
                return None

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
            keychain = int(ref.get("keychain_price") or 0)
            effective_base = (
                (int(base) + keychain) if base and int(base) > 0 else keychain
            )
            fair_value: int | None = None
            if effective_base > 0:
                fair_value = effective_base
            elif predicted and int(predicted) > 0:
                fair_value = int(predicted)

            base_price_int: int = (
                effective_base if effective_base > 0 else (fair_value or price)
            )
            volume: int = int(ref.get("quantity") or 0)

            return {
                "listing_id": listing_id,
                "item_name": item_name,
                "price": price,
                "float_value": float_value,
                "fair_value": fair_value,
                "base_price": base_price_int,
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

        if item_name.startswith(
            (
                "Sticker |",
                "Patch |",
                "Sealed Graffiti |",
                "Graffiti |",
                "Music Kit |",
                "Pin |",
            )
        ):
            return False, f"excluded category: {item_name.split(' |')[0]}"

        # 2. Price range
        if price < self.config.min_item_price:
            return False, f"price {price}¢ below minimum {self.config.min_item_price}¢"
        if price > self.config.max_item_price:
            return False, f"price {price}¢ above cap {self.config.max_item_price}¢"

        # 3. Fair value availability
        if fair_value is None or fair_value < self.config.min_fair_value:
            return (
                False,
                f"fair value {fair_value}¢ below minimum {self.config.min_fair_value}¢",
            )

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
        self._steam_id: str = ""
        # Callback invoked after a successful buy with (listing dict). Wired by TradingBot.
        self.on_buy_success: Callable[[dict], Awaitable[None]] | None = None

    @property
    def steam_id(self) -> str:
        return self._steam_id

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
                    "balance field not found in /api/v1/me — available keys: %s",
                    list(data.keys()),
                )
                return False
            self._balance = int(balance)
            self._balance_loaded = True
            self._last_balance_refresh = asyncio.get_event_loop().time()
            # Extract steam_id (needed for /users/{steam_id}/stall)
            user = data.get("user") or data
            sid = (
                user.get("steam_id")
                or user.get("steamid")
                or user.get("steamId")
                or user.get("id")
            )
            if sid and not self._steam_id:
                self._steam_id = str(sid)
                logger.info("Steam ID resolved: %s", self._steam_id)
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

    def _build_headers(self, include_auth: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {
            "Cookie": self.config.cookies,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": self._UA,
            "Referer": "https://csfloat.com/",
            "Origin": "https://csfloat.com",
        }
        if include_auth:
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
        if (
            not self._balance_loaded
            or (now - self._last_balance_refresh) > self.config.balance_refresh_interval
        ):
            await self.refresh_balance()

        # Balance gate: skip if insufficient funds or would breach reserve
        if self._balance_loaded:
            if self._balance < price:
                logger.warning(
                    "Insufficient balance: have $%.2f, need $%.2f for %r — skipping",
                    self._balance / 100,
                    price / 100,
                    listing["item_name"],
                )
                return False
            reserve = self.config.min_balance_reserve
            if (self._balance - price) < reserve:
                logger.warning(
                    "Reserve guard: buying %r ($%.2f) would drop balance to $%.2f < reserve $%.2f — skipping",
                    listing["item_name"],
                    price / 100,
                    (self._balance - price) / 100,
                    reserve / 100,
                )
                return False

        url = self.config.buy_url
        try:
            async with self.session.post(
                url,
                headers=self._build_headers(include_auth=False),
                json={"total_price": price, "contract_ids": [listing_id]},
            ) as resp:
                return await self._handle_response(resp, listing)
        except aiohttp.ClientError as exc:
            logger.error("HTTP client error buying listing_id=%s: %s", listing_id, exc)
            return False
        except Exception as exc:
            logger.error(
                "Unexpected error buying listing_id=%s: %s",
                listing_id,
                exc,
                exc_info=True,
            )
            return False

    async def _handle_response(
        self, resp: aiohttp.ClientResponse, listing: dict
    ) -> bool:
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
                f"{listing['float_value']:.6f}"
                if listing.get("float_value") is not None
                else "n/a",
                listing_id,
                self._balance / 100,
            )
            await self._notifier.send_buy(listing, self._balance)
            if self.on_buy_success is not None:
                try:
                    await self.on_buy_success(listing)
                except Exception as exc:
                    logger.error(
                        "on_buy_success callback failed: %s", exc, exc_info=True
                    )
            return True

        body = await resp.text()

        if status == 400:
            try:
                import json as _json

                err = _json.loads(body)
                code = err.get("code")
                msg = err.get("message", body)
            except Exception:
                code, msg = None, body
            if code == 4:  # "already sold" — kapıldı
                logger.info(
                    "Already sold: %r  listing_id=%s", listing["item_name"], listing_id
                )
                await self._notifier.send_buy_failed(listing, "Başkası kaptı")
            else:
                logger.warning(
                    "Buy rejected (400 code=%s): %s  listing_id=%s",
                    code,
                    msg,
                    listing_id,
                )
                await self._notifier.send_buy_failed(listing, f"Reddedildi: {msg}")
            return False

        if status == 429:
            logger.warning(
                "Rate limited (HTTP 429). Pausing %.0fs. listing_id=%s",
                self.config.rate_limit_pause,
                listing_id,
            )
            self._rate_limited_until = (
                asyncio.get_event_loop().time() + self.config.rate_limit_pause
            )
            return False

        if status in (401, 403):
            logger.critical(
                "AUTH FAILURE (HTTP %d) — credentials expired. Update .env and restart. body=%s",
                status,
                body[:300],
            )
            await self._notifier.send_error(
                f"🔐 Satın alma auth hatası (HTTP {status})\n"
                ".env dosyasını güncelle ve botu yeniden başlat."
            )
            return False

        reason = f"HTTP {status}"
        logger.warning(
            "Unexpected HTTP %d for listing_id=%s. body=%s",
            status,
            listing_id,
            body[:300],
        )
        await self._notifier.send_buy_failed(listing, reason)
        return False


# ── Inventory + Stall API client ──────────────────────────────────────────────


class CsfloatClient:
    """
    Thin wrapper around the CSFloat endpoints we need for portfolio management.
    Reuses the same User-Agent/Cookie/Referer pattern as ExecutionModule so that
    every authenticated request looks identical to the browser.
    """

    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def __init__(self, config: BotConfig, session: aiohttp.ClientSession) -> None:
        self.config = config
        self.session = session

    def _headers(self, json_body: bool = False) -> dict[str, str]:
        h: dict[str, str] = {
            "Cookie": self.config.cookies,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": self._UA,
            "Referer": "https://csfloat.com/",
            "Origin": "https://csfloat.com",
        }
        if json_body:
            h["Content-Type"] = "application/json"
        auth = self.config.auth_header_value()
        if auth:
            h["Authorization"] = auth
        return h

    async def fetch_inventory(self) -> list[dict] | None:
        """GET /me/inventory. Returns list of items, or None on auth/network failure."""
        try:
            async with self.session.get(
                self.config.inventory_url, headers=self._headers()
            ) as resp:
                if resp.status in (401, 403):
                    logger.critical("Inventory AUTH FAILURE (HTTP %d)", resp.status)
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Inventory fetch HTTP %d: %s", resp.status, body[:200]
                    )
                    return None
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning("Inventory fetch network error: %s", exc)
            return None

        # CSFloat returns either a bare list or {"data": [...]} — handle both
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or data.get("items") or data.get("inventory") or []
        return []

    async def fetch_stall(self, steam_id: str) -> list[dict] | None:
        """GET /users/{steam_id}/stall — listings the user currently has up for sale."""
        if not steam_id:
            logger.warning("Stall fetch: steam_id not yet resolved, skipping")
            return None
        url = self.config.stall_url_template.format(steam_id=steam_id)
        try:
            async with self.session.get(url, headers=self._headers()) as resp:
                if resp.status in (401, 403):
                    logger.critical("Stall AUTH FAILURE (HTTP %d)", resp.status)
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Stall fetch HTTP %d: %s", resp.status, body[:200])
                    return None
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning("Stall fetch network error: %s", exc)
            return None

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or data.get("listings") or []
        return []

    async def fetch_lowest_listing_price(self, item_name: str) -> int | None:
        """Look up the cheapest current buy_now listing for an item_name (in cents)."""
        url = (
            "https://csfloat.com/api/v1/listings"
            f"?sort_by=lowest_price&limit=10&market_hash_name="
            f"{url_quote(item_name, safe='')}"
        )
        try:
            async with self.session.get(url, headers=self._headers()) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
        except aiohttp.ClientError:
            return None

        listings = (
            payload
            if isinstance(payload, list)
            else (payload.get("data") or payload.get("listings") or [])
        )
        prices: list[int] = []
        for raw in listings:
            try:
                if raw.get("type") == "auction" or raw.get("is_auction") is True:
                    continue
                if str(raw.get("item", {}).get("market_hash_name")) != item_name:
                    continue
                prices.append(int(raw["price"]))
            except (KeyError, TypeError, ValueError):
                continue
        return min(prices) if prices else None

    async def fetch_cheapest_listing(self, item_name: str) -> dict | None:
        """Return the raw cheapest buy_now listing dict for ``item_name``.

        Unlike :meth:`fetch_lowest_listing_price` this returns the full listing
        (id, price, item, reference.quantity), so the momentum strategy can both
        snapshot the price+volume AND buy that exact listing.
        """
        url = (
            "https://csfloat.com/api/v1/listings"
            f"?sort_by=lowest_price&limit=10&market_hash_name="
            f"{url_quote(item_name, safe='')}"
        )
        try:
            async with self.session.get(url, headers=self._headers()) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
        except aiohttp.ClientError:
            return None

        listings = (
            payload
            if isinstance(payload, list)
            else (payload.get("data") or payload.get("listings") or [])
        )
        cheapest: dict | None = None
        for raw in listings:
            try:
                if raw.get("type") == "auction" or raw.get("is_auction") is True:
                    continue
                if str(raw.get("item", {}).get("market_hash_name")) != item_name:
                    continue
                price = int(raw["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if cheapest is None or price < int(cheapest["price"]):
                cheapest = raw
        return cheapest

    async def create_listing(self, asset_id: str, price: int) -> tuple[str | None, str]:
        """
        POST /api/v1/listings  body: {asset_id, type=buy_now, price, ...}
        Returns (listing_id, error_msg). On success error_msg is "".
        Endpoint format is a best guess — first run logs the full request body for
        verification.
        """
        body = {
            "asset_id": str(asset_id),
            "type": "buy_now",
            "price": int(price),
            "description": "",
            "private": False,
        }
        logger.info("CREATE LISTING request body=%s", body)
        try:
            async with self.session.post(
                self.config.listings_create_url,
                headers=self._headers(json_body=True),
                json=body,
            ) as resp:
                text = await resp.text()
                if resp.status in (200, 201):
                    try:
                        data = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        data = {}
                    listing_id = (
                        data.get("id")
                        or data.get("listing_id")
                        or (data.get("listing") or {}).get("id")
                    )
                    logger.info(
                        "CREATE LISTING ok status=%d body=%s", resp.status, text[:300]
                    )
                    return (str(listing_id) if listing_id else None), ""
                logger.warning(
                    "CREATE LISTING failed status=%d body=%s", resp.status, text[:300]
                )
                return None, f"HTTP {resp.status}: {text[:200]}"
        except aiohttp.ClientError as exc:
            return None, f"network error: {exc}"

    async def delete_listing(self, listing_id: str) -> bool:
        """DELETE /api/v1/listings/{id}. Returns True on success."""
        url = f"{self.config.listings_create_url}/{listing_id}"
        try:
            async with self.session.delete(url, headers=self._headers()) as resp:
                text = await resp.text()
                if resp.status in (200, 204):
                    logger.info(
                        "DELETE LISTING ok id=%s status=%d", listing_id, resp.status
                    )
                    return True
                logger.warning(
                    "DELETE LISTING failed id=%s status=%d body=%s",
                    listing_id,
                    resp.status,
                    text[:200],
                )
                return False
        except aiohttp.ClientError as exc:
            logger.warning("DELETE LISTING network error id=%s: %s", listing_id, exc)
            return False

    async def fetch_contract(self, contract_id: str) -> dict | None:
        """GET /api/v1/listings/{id} — fetch a purchased listing to extract asset_id."""
        url = f"{self.config.listings_create_url}/{contract_id}"
        try:
            async with self.session.get(url, headers=self._headers()) as resp:
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "fetch_contract HTTP %d for %s: %s",
                        resp.status,
                        contract_id,
                        body[:200],
                    )
                    return None
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning("fetch_contract network error %s: %s", contract_id, exc)
            return None


# ── Portfolio Manager (post-buy lifecycle: asset resolve → list → reprice → sold) ──


class PortfolioManager:
    """
    Owns the bot-bought items lifecycle:
      1. After buy: resolve asset_id from inventory, set trade_unlock_at
      2. When unlocked: list at Phase 1 price (base × 1.08)
      3. After day 3: reprice to Phase 2 (base × 1.05)
      4. After day 7: reprice to Phase 3 (undercut lowest by 1¢, floor at buy × 1.03)
      5. After day 14: alert user for manual review
      6. Continuously detect sold items via stall diff
    """

    def __init__(
        self,
        config: BotConfig,
        tracker: BoughtItemsTracker,
        client: CsfloatClient,
        notifier: TelegramNotifier,
        executor: "ExecutionModule",
    ) -> None:
        self.config = config
        self.tracker = tracker
        self.client = client
        self._notifier = notifier
        self._executor = executor

    # ── Step 1: asset_id resolution ───────────────────────────────────────────

    async def on_buy_success(self, listing: dict) -> None:
        """Called by ExecutionModule after a successful buy. Fire-and-forget."""
        contract_id = listing["listing_id"]
        item_name = listing["item_name"]
        buy_price = listing["price"]
        _bp = listing.get("base_price") or 0
        _fv = listing.get("fair_value") or 0
        base_price = max(_bp, _fv) if max(_bp, _fv) > buy_price // 2 else buy_price
        float_value = listing.get("float_value")
        strategy = listing.get("strategy", "sniper")
        target_price = listing.get("target_price")
        pending_key = await self.tracker.add_pending(
            contract_id,
            item_name,
            buy_price,
            base_price,
            float_value=float_value,
            strategy=strategy,
            target_price=target_price,
        )
        asyncio.create_task(
            self._resolve_asset_for(pending_key, contract_id, item_name, buy_price)
        )

    async def _resolve_asset_for(
        self, pending_key: str, contract_id: str, item_name: str, buy_price: int
    ) -> None:
        """Poll inventory until we find the freshly-bought item (or give up)."""
        await asyncio.sleep(self.config.inventory_resolve_delay)
        seen_before = await self._inventory_asset_ids()  # snapshot before we re-poll
        for attempt in range(self.config.inventory_resolve_retries):
            inv = await self.client.fetch_inventory()
            if inv is None:
                await asyncio.sleep(self.config.inventory_resolve_delay)
                continue
            match = self._match_inventory_item(inv, item_name, exclude_ids=seen_before)
            if match is not None:
                asset_id, unlock_iso = match
                await self.tracker.resolve_asset(pending_key, asset_id, unlock_iso)
                logger.info(
                    "Resolved asset_id=%s for %r (unlock=%s)",
                    asset_id,
                    item_name,
                    unlock_iso,
                )
                return
            logger.info(
                "Asset resolve attempt %d/%d: item %r not in inventory yet",
                attempt + 1,
                self.config.inventory_resolve_retries,
                item_name,
            )
            await asyncio.sleep(self.config.inventory_resolve_delay)

        # Fast-path tükendi — satıcı henüz trade'i kabul etmemiş olabilir.
        # Slow-path için pending_trade'e geç, background loop saatler boyunca tekrar dener.
        logger.info(
            "Fast-path exhausted for %r → pending_trade "
            "(seller likely hasn't accepted the trade offer yet; background loop will retry)",
            item_name,
        )
        await self.tracker.update(pending_key, status="pending_trade")

    async def _inventory_asset_ids(self) -> set[str]:
        inv = await self.client.fetch_inventory()
        if not inv:
            return set()
        out: set[str] = set()
        for it in inv:
            aid = self._extract_asset_id(it)
            if aid:
                out.add(aid)
        return out

    @staticmethod
    def _extract_asset_id(raw: dict) -> str | None:
        """Inventory item shape varies — try common field names."""
        for key in ("asset_id", "assetid", "id"):
            v = raw.get(key)
            if v:
                return str(v)
        item = raw.get("item") or {}
        for key in ("asset_id", "assetid", "id"):
            v = item.get(key)
            if v:
                return str(v)
        return None

    @staticmethod
    def _extract_item_name(raw: dict) -> str:
        for key in ("market_hash_name", "name"):
            v = raw.get(key)
            if v:
                return str(v)
        item = raw.get("item") or {}
        for key in ("market_hash_name", "name"):
            v = item.get(key)
            if v:
                return str(v)
        return ""

    @staticmethod
    def _extract_float(raw: dict) -> float | None:
        for key in ("float_value", "float", "floatvalue"):
            v = raw.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        item = raw.get("item") or {}
        for key in ("float_value", "float", "floatvalue"):
            v = item.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _extract_tradable_at(raw: dict) -> str | None:
        for key in ("tradable_at", "tradableAfter", "trade_unlock", "trade_lock_until"):
            v = raw.get(key)
            if v:
                return str(v)
        item = raw.get("item") or {}
        for key in ("tradable_at", "tradableAfter", "trade_unlock"):
            v = item.get(key)
            if v:
                return str(v)
        return None

    def _match_inventory_item(
        self, inventory: list[dict], item_name: str, exclude_ids: set[str]
    ) -> tuple[str, str] | None:
        """
        Find the inventory item matching `item_name` that wasn't in `exclude_ids`
        before the buy. Returns (asset_id, trade_unlock_iso) or None.
        """
        for raw in inventory:
            asset_id = self._extract_asset_id(raw)
            name = self._extract_item_name(raw)
            if not asset_id or asset_id in exclude_ids:
                continue
            if name != item_name:
                continue
            # Compute unlock: max(now + buffer, tradable_at + 1d)
            buffer_unlock = datetime.now(timezone.utc) + timedelta(
                days=self.config.trade_hold_buffer_days
            )
            unlock_dt = buffer_unlock
            tradable_raw = self._extract_tradable_at(raw)
            if tradable_raw:
                try:
                    parsed = datetime.fromisoformat(tradable_raw.replace("Z", "+00:00"))
                    unlock_dt = max(unlock_dt, parsed + timedelta(days=1))
                except ValueError:
                    pass
            return asset_id, unlock_dt.isoformat()
        return None

    # ── Step 2-5: listing + repricing ─────────────────────────────────────────

    @staticmethod
    def _parse_iso(s: str) -> datetime | None:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _phase_for(self, listed_at_iso: str) -> int:
        listed_dt = self._parse_iso(listed_at_iso) or datetime.now(timezone.utc)
        age_days = (datetime.now(timezone.utc) - listed_dt).days
        if age_days < self.config.phase_1_duration_days:
            return 1
        if (
            age_days
            < self.config.phase_1_duration_days + self.config.phase_2_duration_days
        ):
            return 2
        total = (
            self.config.phase_1_duration_days
            + self.config.phase_2_duration_days
            + self.config.phase_3_duration_days
        )
        if age_days < total:
            return 3
        return 4  # stuck — needs manual review

    def _effective_base(self, item: BoughtItem) -> int:
        """Float tier içinde düşükse base_price'ı prim oranıyla artır."""
        base = item.base_price_at_buy
        if item.float_value is not None:
            pos = _float_tier_position(item.float_value, item.item_name)
            if pos is not None and pos < self.config.float_low_threshold:
                base = int(base * self.config.float_low_premium)
        return base

    async def _price_for_phase(self, item: BoughtItem, phase: int) -> int:
        floor = int(item.buy_price * self.config.min_listing_markup)
        # Momentum item'ları base_price markup ile değil, baseline'a (target_price)
        # dönüşle satılır. Phase 3'te yine en ucuzdan undercut devreye girer.
        if item.strategy == "momentum" and item.target_price:
            if phase in (1, 2):
                return max(floor, item.target_price)
            if phase == 3:
                lowest = await self.client.fetch_lowest_listing_price(item.item_name)
                if lowest is None:
                    return max(floor, item.target_price)
                return max(floor, lowest - self.config.phase_3_undercut_cents)
            return floor
        base = self._effective_base(item)
        if phase == 1:
            return max(floor, int(base * self.config.phase_1_markup))
        if phase == 2:
            return max(floor, int(base * self.config.phase_2_markup))
        if phase == 3:
            lowest = await self.client.fetch_lowest_listing_price(item.item_name)
            if lowest is None:
                return max(floor, int(base * self.config.phase_2_markup))
            return max(floor, lowest - self.config.phase_3_undercut_cents)
        return floor  # phase 4 — shouldn't be called, but safe fallback

    def _find_in_inventory(self, inventory: list[dict], item: BoughtItem) -> str | None:
        """Return ``item``'s live asset_id in the current inventory, or None.

        Steam reassigns an item's asset_id when its trade-hold expires, so the id
        captured at buy time goes stale and CSFloat's listing API 403s with
        "not found in your inventory". We re-match by market_hash_name (and float
        when known, to disambiguate duplicates), skipping ids already claimed by
        other tracked items. If the stored id is still present we keep it.
        """
        claimed = {
            it.asset_id
            for it in self.tracker.all()
            if it.asset_id and it.asset_id != item.asset_id
        }
        best_id: str | None = None
        best_delta: float | None = None
        fallback_id: str | None = None
        for raw in inventory:
            aid = self._extract_asset_id(raw)
            if not aid or aid in claimed:
                continue
            if self._extract_item_name(raw) != item.item_name:
                continue
            if item.asset_id and aid == item.asset_id:
                return aid  # stored id still valid — nothing to do
            if item.float_value is not None:
                fv = self._extract_float(raw)
                if fv is not None:
                    delta = abs(fv - item.float_value)
                    if best_delta is None or delta < best_delta:
                        best_id, best_delta = aid, delta
                    continue
            if fallback_id is None:
                fallback_id = aid
        return best_id or fallback_id

    async def _sync_asset_id(
        self, item: BoughtItem, inventory: list[dict] | None
    ) -> str | None:
        """Resolve the current asset_id for ``item``, re-keying the tracker if it changed.

        Returns the live asset_id to list with, or None if the item is no longer
        in inventory (already listed, sold, or refunded) — in which case the
        caller should skip rather than retry a dead id.
        """
        if inventory is None:
            return item.asset_id  # couldn't fetch — fall back to stored id
        live = self._find_in_inventory(inventory, item)
        if live is None:
            return None
        if live != item.asset_id:
            logger.info(
                "asset_id changed for %r: %s → %s (trade-hold expired, re-keying)",
                item.item_name,
                item.asset_id,
                live,
            )
            old_key = item.asset_id
            await self.tracker.rekey_asset(old_key, live)
            item.asset_id = live
        return live

    async def list_unlocked(self) -> int:
        """List any locked items whose trade hold has expired. Returns count listed."""
        now = datetime.now(timezone.utc)
        ready = [
            item
            for item in self.tracker.by_status("locked")
            if (dt := self._parse_iso(item.trade_unlock_at)) is not None and dt <= now
        ]
        if not ready:
            return 0
        # Re-resolve asset_ids against the live inventory once per pass: Steam
        # reassigns ids when the hold lifts, so the id stored at buy time is stale.
        inventory = await self.client.fetch_inventory()
        count = 0
        for item in ready:
            asset_id = await self._sync_asset_id(item, inventory)
            if asset_id is None:
                logger.warning(
                    "List skip %r: not found in inventory (stale id=%s) — "
                    "already listed/sold/refunded?",
                    item.item_name,
                    item.asset_id,
                )
                await self.tracker.update(
                    item.asset_id,
                    error_count=item.error_count + 1,
                    last_error="item not in inventory",
                )
                continue
            price = await self._price_for_phase(item, phase=1)
            eff_base = self._effective_base(item)
            if eff_base != item.base_price_at_buy:
                pos = _float_tier_position(item.float_value, item.item_name)
                logger.info(
                    "Float premium applied: %r float=%.4f pos=%.2f base %d→%d",
                    item.item_name,
                    item.float_value,
                    pos or 0.0,
                    item.base_price_at_buy,
                    eff_base,
                )
            listing_id, err = await self.client.create_listing(item.asset_id, price)
            if listing_id is None:
                logger.warning("Failed to list %r: %s", item.item_name, err)
                await self.tracker.update(
                    item.asset_id,
                    error_count=item.error_count + 1,
                    last_error=err,
                )
                continue
            await self.tracker.update(
                item.asset_id,
                status="listed",
                listing_id_sell=listing_id,
                list_price=price,
                listed_at=now.isoformat(),
                current_phase=1,
                error_count=0,
                last_error="",
            )
            await self._notifier.send_listed(item, price, phase=1)
            count += 1
        return count

    async def reprice_phases(self) -> int:
        """Walk listed items, transition between phases, alert at phase 4. Returns count repriced."""
        count = 0
        for item in self.tracker.by_status("listed"):
            if not item.listed_at:
                continue
            current = item.current_phase or 1
            target = self._phase_for(item.listed_at)
            if target == current:
                continue
            if target == 4:
                age = (
                    datetime.now(timezone.utc)
                    - (self._parse_iso(item.listed_at) or datetime.now(timezone.utc))
                ).days
                await self._notifier.send_stuck(item, days_listed=age)
                await self.tracker.update(item.asset_id, current_phase=4)
                continue
            new_price = await self._price_for_phase(item, target)
            if new_price == item.list_price:
                await self.tracker.update(item.asset_id, current_phase=target)
                continue
            # Delete + re-create (CSFloat doesn't support price update on existing listing universally)
            if item.listing_id_sell:
                await self.client.delete_listing(item.listing_id_sell)
            # Delisting returns the item to inventory, possibly under a new asset_id.
            inventory = await self.client.fetch_inventory()
            if await self._sync_asset_id(item, inventory) is None:
                logger.warning(
                    "Reprice skip %r: not found in inventory after delist (id=%s)",
                    item.item_name,
                    item.asset_id,
                )
                await self.tracker.update(
                    item.asset_id,
                    error_count=item.error_count + 1,
                    last_error="item not in inventory after delist",
                )
                continue
            new_id, err = await self.client.create_listing(item.asset_id, new_price)
            if new_id is None:
                logger.warning("Reprice failed for %r: %s", item.item_name, err)
                await self.tracker.update(
                    item.asset_id,
                    error_count=item.error_count + 1,
                    last_error=err,
                )
                continue
            old_price = item.list_price or 0
            await self.tracker.update(
                item.asset_id,
                listing_id_sell=new_id,
                list_price=new_price,
                current_phase=target,
            )
            await self._notifier.send_repriced(item, old_price, new_price, target)
            count += 1
        return count

    # ── Step 6: sales detection ───────────────────────────────────────────────

    async def detect_sales(self) -> int:
        """Diff stall against tracker — anything missing has been sold. Returns count."""
        stall = await self.client.fetch_stall(self._executor.steam_id)
        if stall is None:
            return 0
        active_listing_ids: set[str] = set()
        for raw in stall:
            lid = raw.get("id") or raw.get("listing_id")
            if lid:
                active_listing_ids.add(str(lid))

        count = 0
        for item in self.tracker.by_status("listed"):
            if not item.listing_id_sell:
                continue
            if item.listing_id_sell in active_listing_ids:
                continue
            # Listing gone from stall → sold
            list_price = item.list_price or 0
            net = (
                int(list_price * (1 - self.config.csfloat_seller_fee)) - item.buy_price
            )
            await self.tracker.update(
                item.asset_id,
                status="sold",
                sold_at=datetime.now(timezone.utc).isoformat(),
                net_profit=net,
            )
            updated = self.tracker.get(item.asset_id)
            if updated is not None:
                await self._notifier.send_sold(updated)
            count += 1
        return count

    # ── Orchestration ─────────────────────────────────────────────────────────

    async def retry_pending_trades(self) -> int:
        """
        Slow-path resolver: try to resolve pending_trade items via inventory.
        Sellers can take minutes-to-hours to accept the trade offer; once they do,
        the item appears in our Steam inventory.
        After max_age_hours (default 24h), give up — CSFloat auto-refunds expired trades.
        Returns count of items resolved this pass.
        """
        pending = [it for it in self.tracker.all() if it.status == "pending_trade"]
        if not pending:
            return 0
        inv = await self.client.fetch_inventory()
        if inv is None:
            return 0
        logger.info(
            "retry_pending_trades: %d pending, inventory size=%d",
            len(pending),
            len(inv),
        )
        # Exclude asset_ids already claimed by other tracker entries to avoid double-assignment
        claimed = {it.asset_id for it in self.tracker.all() if it.asset_id}
        resolved = 0
        now = datetime.now(timezone.utc)
        for item in pending:
            pending_key = f"pending:{item.contract_id_buy}"
            bought_dt = self._parse_iso(item.bought_at) or now
            age_hours = (now - bought_dt).total_seconds() / 3600

            asset_id: str | None = None
            unlock_iso: str | None = None

            # 1) Try contract API — works even during trade hold
            contract = await self.client.fetch_contract(item.contract_id_buy)
            if contract:
                raw_id = (
                    contract.get("asset_id")
                    or contract.get("item", {}).get("asset_id")
                    or contract.get("item", {}).get("assetid")
                )
                if raw_id and str(raw_id) not in claimed:
                    asset_id = str(raw_id)
                    # trade_unlock_at: prefer API field, fall back to buy_time + hold_days
                    raw_unlock = contract.get("item", {}).get(
                        "tradable_at"
                    ) or contract.get("tradable_at")
                    if raw_unlock:
                        try:
                            parsed = datetime.fromisoformat(
                                str(raw_unlock).replace("Z", "+00:00")
                            )
                            unlock_iso = (parsed + timedelta(days=1)).isoformat()
                        except ValueError:
                            pass
                    if unlock_iso is None:
                        unlock_iso = (
                            bought_dt
                            + timedelta(days=self.config.trade_hold_buffer_days)
                        ).isoformat()

            # 2) Fall back to inventory match (works once trade hold clears)
            if asset_id is None:
                match = self._match_inventory_item(
                    inv, item.item_name, exclude_ids=claimed
                )
                if match is not None:
                    asset_id, unlock_iso = match

            if asset_id is not None and unlock_iso is not None:
                await self.tracker.resolve_asset(pending_key, asset_id, unlock_iso)
                claimed.add(asset_id)
                resolved_item = self.tracker.get(asset_id)
                if resolved_item is not None:
                    await self._notifier.send_pending_resolved(resolved_item, age_hours)
                resolved += 1
                continue

            # Not found anywhere — give up only if past max age
            if age_hours > self.config.pending_trade_max_age_hours:
                await self.tracker.update(
                    pending_key,
                    status="failed",
                    last_error=f"trade not accepted after {age_hours:.1f}h",
                )
                await self._notifier.send_asset_resolve_failed(item)
        return resolved

    async def run_catch_up(self) -> dict:
        """One-shot pass over all workers. Used at startup and on /catchup."""
        logger.info("Portfolio catch-up: starting full cycle")
        resolved = await self.retry_pending_trades()
        listed = await self.list_unlocked()
        sold = await self.detect_sales()
        repriced = await self.reprice_phases()
        summary = {
            "resolved": resolved,
            "listed": listed,
            "sold": sold,
            "repriced": repriced,
        }
        logger.info("Portfolio catch-up complete: %s", summary)
        return summary

    async def listing_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.listing_worker_interval)
                await self.list_unlocked()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("listing_loop error: %s", exc, exc_info=True)

    async def sales_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.sales_worker_interval)
                await self.detect_sales()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("sales_loop error: %s", exc, exc_info=True)

    async def repricing_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.repricing_worker_interval)
                await self.reprice_phases()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("repricing_loop error: %s", exc, exc_info=True)

    async def pending_trade_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.pending_trade_retry_interval)
                await self.retry_pending_trades()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("pending_trade_loop error: %s", exc, exc_info=True)


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
                # Sniper kapalıyken poller sadece watchlist'i besliyor — hızlı
                # dönmesine gerek yok; sıkı rate-limit'i momentum'a bırak.
                interval = self.config.poll_interval
                if not self.config.sniper_enabled:
                    interval = max(interval, self.config.poll_interval_idle)
                await asyncio.sleep(interval)

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
                        pause,
                        self._consecutive_429s,
                    )
                    if self._consecutive_429s == 1:
                        await self._notifier.send_error(
                            f"⏳ Rate limit (429) — {pause:.0f}s bekleniyor "
                            f"(art arda {self._consecutive_429s}. kez)"
                        )
                    await asyncio.sleep(pause)
                    return
                if resp.status in (401, 403):
                    logger.critical(
                        "Poll AUTH FAILURE (HTTP %d) — update .env and restart",
                        resp.status,
                    )
                    await self._notifier.send_error(
                        f"🔐 Auth başarısız (HTTP {resp.status})\n"
                        ".env dosyasını güncelle ve botu yeniden başlat."
                    )
                    return
                resp.raise_for_status()
                payload = await resp.json()
                rl_remaining = resp.headers.get("X-RateLimit-Remaining")
                rl_reset = resp.headers.get("X-RateLimit-Reset")

            # Jitter after every request to mimic human timing
            await asyncio.sleep(random.uniform(0.8, 1.8))

            # Adaptive throttle: spread remaining quota evenly across the reset window
            if rl_remaining is not None and rl_reset is not None:
                try:
                    remaining = int(rl_remaining)
                    reset_in = max(1.0, float(rl_reset) - time.time())
                    if remaining > 0:
                        ideal_interval = reset_in / remaining
                        extra = ideal_interval - self.config.poll_interval
                        if extra > 0:
                            logger.info(
                                "Adaptive throttle: %d req left, reset in %.0fs → poll interval %.0fs",
                                remaining,
                                reset_in,
                                ideal_interval,
                            )
                            await asyncio.sleep(extra)
                except (ValueError, TypeError):
                    pass

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
                    logger.error(
                        "Error processing listing %s: %s", lid, exc, exc_info=True
                    )

            # Stop paging: no cursor returned, or page 1+ had nothing new (caught up)
            if not next_cursor or (not found_new and page_num > 0):
                break
            cursor = next_cursor

        self._consecutive_429s = 0

        if not self._initialized:
            self._initialized = True
            logger.info(
                "Poller initialised — %d existing listing(s) seeded (%d page(s))",
                total_seeded,
                page_num + 1,
            )
        elif new_count:
            logger.debug(
                "Poll: %d new listing(s) evaluated across %d page(s)",
                new_count,
                page_num + 1,
            )

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
        self._notifier: TelegramNotifier | None = None
        self._cmd_handler: TelegramCommandHandler | None = None
        self._cmd_task: asyncio.Task | None = None
        self._tracker: BoughtItemsTracker | None = None
        self._portfolio: PortfolioManager | None = None
        self._pricedb: PriceDB | None = None
        self._momentum: MomentumStrategy | None = None
        self._watch_seen: set[str] = set()
        self._background_tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            self._notifier = TelegramNotifier(
                self.config.telegram_bot_token,
                self.config.telegram_chat_id,
                session,
            )
            self._executor = ExecutionModule(self.config, session, self._notifier)
            await self._executor.refresh_balance()
            await self._notifier.send_startup(self.config, self._executor._balance)

            # Portfolio layer: tracker + API client + manager
            data_dir = Path(__file__).parent / self.config.data_dir
            self._tracker = BoughtItemsTracker(data_dir)
            client = CsfloatClient(self.config, session)
            self._portfolio = PortfolioManager(
                self.config, self._tracker, client, self._notifier, self._executor
            )
            self._executor.on_buy_success = self._portfolio.on_buy_success

            # Mean-reversion strateji katmanı (sniper'ın yanında, ayrı bütçe)
            self._pricedb = PriceDB(data_dir / "price_history.db")
            self._momentum = MomentumStrategy(
                self.config,
                session,
                self._pricedb,
                client,
                self.engine,
                self._executor,
                self._tracker,
                self._notifier,
            )

            # Startup catch-up: list unlocked, detect sold, transition phases
            try:
                summary = await self._portfolio.run_catch_up()
                if summary["listed"] or summary["sold"] or summary["repriced"]:
                    logger.info(
                        "Startup catch-up: listed=%d sold=%d repriced=%d",
                        summary["listed"],
                        summary["sold"],
                        summary["repriced"],
                    )
            except Exception as exc:
                logger.error("Startup catch-up failed: %s", exc, exc_info=True)

            self._cmd_handler = TelegramCommandHandler(
                self.config.telegram_bot_token,
                self.config.telegram_chat_id,
                self.config,
                session,
                tracker=self._tracker,
                portfolio=self._portfolio,
                executor=self._executor,
            )
            self._cmd_task = asyncio.create_task(self._cmd_handler.listen())

            # Long-running portfolio workers (pending_trade, listing, sales, repricing)
            self._background_tasks = [
                asyncio.create_task(self._portfolio.pending_trade_loop()),
                asyncio.create_task(self._portfolio.listing_loop()),
                asyncio.create_task(self._portfolio.sales_loop()),
                asyncio.create_task(self._portfolio.repricing_loop()),
            ]
            # Momentum loop'u her zaman başlar; flag'i çalışma anında kontrol
            # eder, böylece /set momentum on/off canlı etki eder.
            self._background_tasks.append(
                asyncio.create_task(self._momentum.run_loop())
            )

            self._poller = MarketPoller(
                self.config, session, self._on_listing, self._notifier
            )
            try:
                await self._poller.run()
            except Exception as exc:
                await self._notifier.send_error(
                    f"Beklenmeyen hata, bot durdu:\n<code>{exc}</code>"
                )
                raise
            finally:
                for t in self._background_tasks:
                    t.cancel()
                self._cmd_task.cancel()
                for t in self._background_tasks + [self._cmd_task]:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

    async def _on_listing(self, raw: dict) -> None:
        listing = self.engine.extract_listing_data(raw)
        if listing is None:
            return

        # Mean-reversion watchlist'i akan stream'den pasifçe besle (yeterli hacimliyse)
        if self._pricedb is not None and self.config.mr_enabled:
            name = listing["item_name"]
            vol = listing.get("volume", 0)
            if vol >= self.config.mr_min_volume and name not in self._watch_seen:
                self._watch_seen.add(name)
                await self._pricedb.add_to_watchlist(name, vol)

        # Sniper kapalıysa burada dur (watchlist beslemesi yukarıda yine de çalıştı)
        if not self.config.sniper_enabled:
            return

        # Bakiye-bazlı ön filtre: alamayacağımız item'ı sessizce atla
        if self._executor and self._executor._balance_loaded:
            affordable = self._executor._balance - self.config.min_balance_reserve
            if listing["price"] > affordable:
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
        assert self._notifier is not None
        await self._notifier.send_match(listing)
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
