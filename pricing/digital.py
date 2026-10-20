"""European digital options, replicated with vanilla legs on the line's smile.

A digital paying ``N`` if it finishes in the money is priced as a portfolio
of vanillas, each a ``Pricable`` on the same market, smile model and bumps:

* term payout (``N`` of the term currency): a centred call (put) spread of
  width ``w`` around the strike, ``N / w`` of each leg — the smile-consistent
  digital, ``-dC/dK`` (``dP/dK``), skew included;
* base payout (``N`` of the base currency): the asset-or-nothing identity,
  ``N`` of the vanilla at the strike plus ``K`` times the term digital
  (minus the vanilla for a put).

Every value bump-and-reprice produces — MTM, every Greek on both smile
families, theta, the extra Greeks — is linear in the price, so the digital's
value is the weighted sum of its legs' values, with no separate digital
formula. Only the premium (the digital's own), the vol shown (the smile vol
at the strike) and the strike are not sums.
"""
from dataclasses import dataclass
from typing import Optional

from helper_functions.premium_quotes import PREMIUM_QUOTES
from pricing.pricing import Pricable, price_option, signed_premium_cashflow

#: Spread width as a fraction of the strike: the centred spread's error is
#: second order in it, and the legs stay far from cancelling to noise.
RELATIVE_WIDTH = 1e-4
#: Leg fields that are not summed: the digital sets them itself.
_NOT_SUMMED = {
    "strike", "notional", "vol_pct", "premium", "pnl", "pnl_base", "pnl_term",
    "auto_premium", "premium_auto", "currency_pair", "expiry", "premium_ccy", "mtm_ccy",
    "pnl_ccy", "mtm_base_ccy", "mtm_term_ccy", "pnl_base_ccy", "pnl_term_ccy", "theta_ccy",
    "greek_problems",
    *(prefix + quote for quote in PREMIUM_QUOTES for prefix in ("mtm_", "premium_", "pnl_")),
}


def legs(strike, call_put, payout_base, width):
    """``[(strike, weight)]`` per unit of payout, in vanilla base notionals of
    the line's call/put; a negative weight is the opposite side."""
    low, high = strike - width / 2.0, strike + width / 2.0
    if str(call_put).strip().upper() == "CALL":
        spread = [(low, 1.0 / width), (high, -1.0 / width)]
        vanilla = 1.0
    else:
        spread = [(high, 1.0 / width), (low, -1.0 / width)]
        vanilla = -1.0
    if not payout_base:
        return spread
    return [(strike, vanilla), *((k, strike * weight) for k, weight in spread)]


def leg_rows(row, payout_base):
    """A digital line (a dict or Series with a numeric strike; its notional
    is the payout) as vanilla rows, for calculators that sum rows (the spot
    ladder). The premium rides on the first leg on the line's own side; the
    other legs carry none."""
    strike, payout = float(row["strike"]), float(row["notional"])
    buy_sell = str(row["buy_sell"]).strip().upper()
    other_side = {"BUY": "SELL", "SELL": "BUY"}[buy_sell]
    rows, premium_placed = [], False
    for leg_strike, weight in legs(strike, row["call_put"], payout_base,
                                   RELATIVE_WIDTH * strike):
        leg = dict(row)
        leg.update(strike=leg_strike, notional=abs(weight) * payout, option_type="vanilla",
                   buy_sell=buy_sell if weight > 0 else other_side, payout_ccy=None)
        if weight > 0 and not premium_placed:
            premium_placed = True
        else:
            leg["premium"] = 0.0
        rows.append(leg)
    return rows


@dataclass
class DigitalPricable:
    """One European digital line; ``row`` as for ``Pricable`` (numeric
    strike, base or term ``payout_ccy`` code). ``row["notional"]`` is the
    payout amount, in ``payout_ccy``. ``premium`` as for ``Pricable``:
    ``None`` asks for the auto premium."""

    row: object
    premium: Optional[float]
    premium_ccy: str
    payout_ccy: str

    def price(self, market, pricing_model, atm_type, delta_base, mods=None,
              next_market=None, next_valuation=None, extras=(), value_market=None):
        row = self.row
        pair = str(row.get("currency_pair", "")).strip().upper()
        if len(pair) != 6:
            raise ValueError("Currency pair must contain two 3-letter ISO codes")
        payout_ccy = str(self.payout_ccy).strip().upper()
        if payout_ccy not in (pair[:3], pair[3:]):
            raise ValueError(f"Payout currency must be {pair[:3]} or {pair[3:]}")
        strike, payout = float(row["strike"]), float(row["notional"])
        if not strike > 0 or not payout >= 0:
            raise ValueError("Strike must be positive and the payout non-negative")
        buy_sell = str(row["buy_sell"]).strip().upper()
        auto_premium = self.premium is None
        signed_premium = signed_premium_cashflow(0.0 if auto_premium else self.premium, buy_sell)
        other_side = {"BUY": "SELL", "SELL": "BUY"}[buy_sell]

        kwargs = {"extras": tuple(extras)} if extras else {}
        if value_market is not None:
            kwargs["value_market"] = value_market
        total, problems = {}, {}
        for leg_strike, weight in legs(strike, row["call_put"], payout_ccy == pair[:3],
                                       RELATIVE_WIDTH * strike):
            leg = row.copy()
            leg["strike"], leg["notional"] = leg_strike, abs(weight) * payout
            leg["buy_sell"] = buy_sell if weight > 0 else other_side
            result = Pricable(row=leg, premium=0.0, premium_ccy=self.premium_ccy).price(
                market, pricing_model, atm_type, delta_base, mods,
                next_market=next_market, next_valuation=next_valuation, **kwargs)
            problems.update(result.get("greek_problems") or {})
            for key, value in result.items():
                if key in _NOT_SUMMED or isinstance(value, (str, bool)):
                    continue
                if key in total and total[key] is None:
                    continue
                total[key] = None if value is None else total.get(key, 0.0) + float(value)
            constant = {k: result[k] for k in ("currency_pair", "expiry", "premium_ccy", "mtm_ccy",
                                               "pnl_ccy", "mtm_base_ccy", "mtm_term_ccy",
                                               "pnl_base_ccy", "pnl_term_ccy", "theta_ccy")}

        # The digital's own premium, vol and strike.
        mtm = total["mtm"]
        if auto_premium:
            dealt = total.get("value_at_vol", mtm)
            signed_premium = signed_premium_cashflow(abs(dealt), buy_sell)
        spot = float(market.s)
        base_premium = self.premium_ccy == pair[:3]
        premium_base = signed_premium if base_premium else signed_premium / spot
        premium_term = signed_premium * spot if base_premium else signed_premium
        unit = row.copy()
        unit["notional"] = 1.0
        _, vol, _ = price_option(market, unit, pricing_model, atm_type, delta_base)
        result = {
            **({"greek_problems": problems} if extras or problems else {}),
            **({"premium_auto": True, "auto_premium": abs(signed_premium)} if auto_premium else {}),
            **constant,
            **total,
            "strike": strike,
            "notional": payout,
            "payout_ccy": payout_ccy,
            "vol_pct": float(vol),
            "premium": signed_premium,
            "pnl": signed_premium + mtm,
            "pnl_base": premium_base + total["mtm_base"],
            "pnl_term": premium_term + total["mtm_term"],
        }
        return result
