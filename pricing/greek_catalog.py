"""Every Greek the app can report, in one list shared by all tabs.

Each Greek comes in two columns: ``fixed_<name>``, shown as sticky-delta (the
ATM/RR/BF quotes stay unchanged as spot moves, so the smile moves with spot in
delta terms), and ``<name>``, shown as smile (the smile moved by the betas).
The column keys keep ``fixed_`` so saved files and choices still read. The eight ``BASE_GREEKS`` are always computed;
the ``EXTRA_GREEKS`` only when a report asks for them
(``pricing.greeks.calculate_extra_greeks``), since each costs extra pricing:

* decay (``<greek>_decay``): the Greek on the next business day, on the
  rolled market theta uses, minus today's; same units and delta reporting
  as the Greek itself. Theta has no decay column;
* cross (``gamma_vol``): change in gamma per vol point, all quoted vols
  scaled together (``Greeks.cross_greeks``);
* weighted (``weighted_gamma``, ``weighted_vega``): gamma and vega times the
  expiry weight ``0.25 / T**0.4`` (``Greeks.weighted_greeks.scale_t``);
* theta parts (``theta_static``, ``theta_vol_roll``, ``theta_carry``): theta
  split into its static day decay (vol, forward and rates held: what
  Bloomberg reports as theta), the strike vol's overnight move and the
  forward points' and rates' roll. Each family's parts add up to its theta:
  sticky theta holds the quotes and the forward, so its carry is zero.
  Hedge forwards' roll is carry. Premium currency, as theta.
* rates (``rho``, ``phi``): domestic/term and foreign/base rate sensitivities
  per +10bp, in base currency, including the parity-consistent forward move.
  Smile betas respond to the equivalent log-forward move, not actual spot.

``CONVENTION_GREEKS`` (``spot_delta``, ``fwd_delta``) come with every priced
line at no extra cost (``pricing.greeks.convention_deltas``) but are shown
only when picked. ``delta`` itself is spot-equivalent in every total; a
line table shows it in the pair's convention (spot or forward, marked S/F).
"""
from collections import OrderedDict, namedtuple

Greek = namedtuple("Greek", "name label unit group")

BASE_GREEKS = ("delta", "gamma", "vega", "vanna", "volga", "theta", "rega", "sega")
DECAY_OF = ("delta", "gamma", "vega", "vanna", "volga", "rega", "sega")
CONVENTION_GREEKS = ("spot_delta", "fwd_delta")

_BASE = (
    Greek("delta", "Delta", "base ccy", "first order"),
    Greek("gamma", "Gamma", "base ccy per 1% spot", "first order"),
    Greek("vega", "Vega", "base ccy per vol point", "first order"),
    Greek("vanna", "Vanna", "base ccy", "first order"),
    Greek("volga", "Volga", "base ccy", "first order"),
    Greek("theta", "Theta", "premium ccy per day", "first order"),
    Greek("rega", "Rega", "base ccy per 10bp RR", "first order"),
    Greek("sega", "Sega", "base ccy per 10bp BF", "first order"),
    Greek("spot_delta", "Spot delta", "base ccy", "first order"),
    Greek("fwd_delta", "Forward delta", "base ccy forward", "first order"),
)
_DECAY_LABELS = {"delta": "Delta decay (charm)", "gamma": "Gamma decay (color)",
                 "vega": "Vega decay (veta)", "vanna": "Vanna decay", "volga": "Volga decay",
                 "rega": "Rega decay", "sega": "Sega decay"}
_DECAY = tuple(Greek(f"{name}_decay", _DECAY_LABELS[name], "change per business day", "decay")
               for name in DECAY_OF)
_CROSS = (Greek("gamma_vol", "Gamma per vol", "gamma change per vol point", "cross"),)
_WEIGHTED = (Greek("weighted_gamma", "Weighted gamma", "gamma x 0.25/T^0.4", "weighted"),
             Greek("weighted_vega", "Weighted vega", "vega x 0.25/T^0.4", "weighted"))
_THETA_PARTS = (Greek("theta_static", "Theta static", "premium ccy per day", "theta"),
                Greek("theta_vol_roll", "Theta vol roll", "premium ccy per day", "theta"),
                Greek("theta_carry", "Theta carry", "premium ccy per day", "theta"))
_RATES = (Greek("rho", "Rho (domestic)", "base ccy per 10bp term rate", "rates"),
          Greek("phi", "Phi (foreign)", "base ccy per 10bp base rate", "rates"))

CATALOG = OrderedDict((greek.name, greek)
                      for greek in (*_BASE, *_DECAY, *_CROSS, *_WEIGHTED, *_THETA_PARTS, *_RATES))
EXTRA_GREEKS = tuple(name for name in CATALOG
                     if name not in BASE_GREEKS and name not in CONVENTION_GREEKS)
DECAY_GREEKS = tuple(greek.name for greek in _DECAY)
WEIGHTED_GREEKS = tuple(greek.name for greek in _WEIGHTED)
THETA_PARTS = tuple(greek.name for greek in _THETA_PARTS)
#: Greeks in the premium currency (theta and its parts); the rest are base.
PREMIUM_CCY_GREEKS = ("theta", *THETA_PARTS)
GROUPS = ("first order", "decay", "cross", "weighted", "theta", "rates")
#: Display tags of the two families (``fixed_<name>`` and ``<name>``), after
#: the Greek so a narrow column still shows which Greek it is.
STICKY_DELTA = "sticky"
SMILE = "smile"


def columns(names):
    """``fixed_<name>``, ``<name>`` for each Greek, in order."""
    return [column for name in names for column in (f"fixed_{name}", name)]


def known(names):
    """``names`` that are in the catalogue, in catalogue order, no repeats."""
    wanted = set(names or ())
    return [name for name in CATALOG if name in wanted]


def extras(names):
    """The extra (on-demand) Greeks among ``names``."""
    return [name for name in known(names) if name in EXTRA_GREEKS]


#: Greeks measured against a direction of the pair's own: its spot (delta,
#: vanna), its risk reversal (rega) or its rates (rho, phi). The same move
#: means different things across pairs (EURUSD up is USD down, USDJPY up is
#: USD up), so these never add up across pairs. A decay of one is one too.
DIRECTIONAL = frozenset({"delta", "spot_delta", "fwd_delta", "pair_delta", "vanna", "rega",
                         "rho", "phi"})


def is_directional(column):
    """True for a column reporting a directional Greek (see ``DIRECTIONAL``),
    in any family, unit or decay: ``vanna_usd``, ``fixed_rega``,
    ``vanna_decay_usd``, ``delta_decay``."""
    name = column[6:] if column.startswith("fixed_") else column
    name = name[:-4] if name.endswith("_usd") else name
    name = name[:-6] if name.endswith("_decay") else name
    return name in DIRECTIONAL


def greek_of(column):
    """The Greek a column reports (``fixed_vega_decay_usd`` -> ``vega_decay``),
    or ``None`` for a column that is not a Greek."""
    name = column[6:] if column.startswith("fixed_") else column
    name = name[:-4] if name.endswith("_usd") else name
    return name if name in CATALOG else None


def label(column, unit=None):
    """Display label for a Greek column: ``fixed_vega`` -> ``Vega (sticky)``
    (sticky-delta), ``vega`` -> ``Vega (smile)``, ``vega_usd`` ->
    ``Vega (smile, USD)``; ``unit`` adds to the tag. Other columns unchanged."""
    usd = column.endswith("_usd")
    name = column[:-4] if usd else column
    fixed = name.startswith("fixed_")
    greek = CATALOG.get(name[6:] if fixed else name)
    if greek is None:
        return column
    tags = [STICKY_DELTA if fixed else SMILE, *(["USD"] if usd else []),
            *([unit] if unit else [])]
    return f"{greek.label} ({', '.join(tags)})"
