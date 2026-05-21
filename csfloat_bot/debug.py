"""
Diagnostic script — bakiyeyi göster, cursor ile çok sayfa tara, her listinge karar ver.
Çalıştır:  poetry run python debug.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
import aiohttp

load_dotenv()

COOKIES = os.getenv("CSFLOAT_COOKIES", "")
TOKEN   = os.getenv("CSFLOAT_BEARER_TOKEN", "")

from bot import (
    MIN_PROFIT_MARGIN, MIN_VOLUME, MIN_FAIR_VALUE,
    MIN_ITEM_PRICE, MAX_ITEM_PRICE, BLACKLIST,
    DecisionEngine, BotConfig,
)

BASE_LISTINGS = "https://csfloat.com/api/v1/listings?sort_by=most_recent&limit=50"
ME_URL        = "https://csfloat.com/api/v1/me"
DEBUG_PAGES   = 3   # kaç sayfa çekileceği (3 × 50 = 150 listing)


def make_headers() -> dict[str, str]:
    h: dict[str, str] = {"Cookie": COOKIES, "Accept": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}" if TOKEN.startswith("eyJ") else TOKEN
    return h


async def fetch_balance(session: aiohttp.ClientSession) -> str:
    try:
        async with session.get(ME_URL, headers=make_headers()) as resp:
            if resp.status != 200:
                return f"HTTP {resp.status}"
            data = await resp.json()
        balance = data.get("balance") or (data.get("user") or {}).get("balance")
        if balance is None:
            return f"alan bulunamadı — mevcut anahtarlar: {list(data.keys())}"
        return f"${int(balance)/100:.2f}  ({balance}¢)"
    except Exception as exc:
        return f"hata: {exc}"


async def fetch_listings(session: aiohttp.ClientSession, pages: int) -> list[dict]:
    all_listings: list[dict] = []
    cursor: str | None = None

    for page_num in range(pages):
        url = BASE_LISTINGS + (f"&cursor={cursor}" if cursor else "")
        async with session.get(url, headers=make_headers()) as resp:
            if resp.status != 200:
                print(f"  Sayfa {page_num+1} HTTP hatası: {resp.status}")
                break
            payload = await resp.json()

        if isinstance(payload, list):
            listings, cursor = payload, None
        else:
            listings = payload.get("data") or payload.get("listings") or []
            cursor = payload.get("cursor")

        all_listings.extend(listings)

        if not cursor:
            break  # son sayfa

    return all_listings


async def main() -> None:
    if not COOKIES:
        print("HATA: .env dosyasında CSFLOAT_COOKIES yok")
        sys.exit(1)

    # ── Auth bilgisi ──────────────────────────────────────────────────────────
    if TOKEN:
        auth_type = f"token ({TOKEN[:8]}...)"
    else:
        auth_type = "cookie-only"

    async with aiohttp.ClientSession() as session:

        # ── Bakiye ───────────────────────────────────────────────────────────
        balance_str = await fetch_balance(session)
        print(f"Auth    : {auth_type}")
        print(f"Bakiye  : {balance_str}")
        print(f"Filtreler: margin≥{MIN_PROFIT_MARGIN*100:.0f}%  vol≥{MIN_VOLUME}  "
              f"fair≥{MIN_FAIR_VALUE}¢  price {MIN_ITEM_PRICE}-{MAX_ITEM_PRICE}¢  "
              f"blacklist={len(BLACKLIST)}")
        print()

        # ── Listing çek ───────────────────────────────────────────────────────
        print(f"{DEBUG_PAGES} sayfa çekiliyor ({DEBUG_PAGES*50} listing hedef)...")
        listings = await fetch_listings(session, DEBUG_PAGES)

    print(f"Toplam listing: {len(listings)}\n")

    cfg = BotConfig(
        bearer_token=TOKEN, cookies=COOKIES,
        min_profit_margin=MIN_PROFIT_MARGIN, min_volume=MIN_VOLUME,
        min_fair_value=MIN_FAIR_VALUE, min_item_price=MIN_ITEM_PRICE,
        max_item_price=MAX_ITEM_PRICE,
    )
    engine = DecisionEngine(cfg)

    buy_candidates = []

    print(f"{'FİYAT':>7}  {'PRED':>6}  {'VOL':>5}  KARAR / ÜRÜN")
    print("─" * 110)

    for raw in listings:
        listing = engine.extract_listing_data(raw)
        if listing is None:
            continue
        ok, reason = engine.should_buy(listing)
        fval = listing.get("fair_value") or 0
        vol  = listing.get("volume", 0)

        if ok:
            verdict = f"✅ {reason}"
            buy_candidates.append(listing)
        else:
            verdict = f"❌ {reason}"

        print(
            f"{listing['price']:>7}¢  {fval:>6}¢  {vol:>5}  "
            f"{verdict:<55}  {listing['item_name']}"
        )

    # ── Özet ──────────────────────────────────────────────────────────────────
    print()
    print("─" * 110)
    if buy_candidates:
        print(f"⚡  {len(buy_candidates)} alım fırsatı ({len(listings)} listing tarandı):")
        for l in buy_candidates:
            disc = (1 - l["price"] / l["fair_value"]) * 100
            print(f"     {l['item_name']:<55}  {l['price']}¢  (fair={l['fair_value']}¢  -{disc:.1f}%)")
    else:
        print(f"Bu {len(listings)} listing'de kriterleri karşılayan fırsat yok.")
        print()
        print("Kalibrasyon önerileri:")
        print(f"  MIN_PROFIT_MARGIN = {MIN_PROFIT_MARGIN}  →  0.10 dene")
        print(f"  MIN_VOLUME        = {MIN_VOLUME}          →  30 dene")
        print(f"  MAX_ITEM_PRICE    = {MAX_ITEM_PRICE}      →  artır")


asyncio.run(main())
