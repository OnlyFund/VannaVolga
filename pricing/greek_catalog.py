"""Every Greek the app can report, in one list shared by all tabs.

Each Greek comes in two columns: ``fixed_<name>`` (fixed volatility) and
``<name>`` (smile-adjusted). The eight ``BASE_GREEKS`` are always computed;
the ``EXTRA_GREEKS`` only when a report asks for them
(``pricing.greeks.calculate_extra_greeks``), since each costs extra pricing:

* decay (``<greek>_decay``): the Greek on the next business day, on the
  rolled market theta uses, minus today's; same units and delta reporting
  as the Greek itself. Theta has no decay column;
* cross (``gamma_vol``): change in gamma per vol point, all quoted vols
  scaled together (``Greeks.cross_greeks``);
* weighted (``weighted_gamma``, ``weighted_vega``): gamma and vega times the
  expiry weight ``0.25 / T**0.4`` (``Greeks.weighted_greeks.scale_t``).

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

CATALOG = OrderedDict((greek.name, greek) for greek in (*_BASE, *_DECAY, *_CROSS, *_WEIGHTED))
EXTRA_GREEKS = tuple(name for name in CATALOG
                     if name not in BASE_GREEKS and name not in CONVENTION_GREEKS)
DECAY_GREEKS = tuple(greek.name for greek in _DECAY)
WEIGHTED_GREEKS = tuple(greek.name for greek in _WEIGHTED)
GROUPS = ("first order", "decay", "cross", "weighted")


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


def greek_of(column):
    """The Greek a column reports (``fixed_vega_decay_usd`` -> ``vega_decay``),
    or ``None`` for a column that is not a Greek."""
    name = column[6:] if column.startswith("fixed_") else column
    name = name[:-4] if name.endswith("_usd") else name
    return name if name in CATALOG else None


def label(column):
    """Display label for a Greek column (``fixed_vega`` -> ``Fixed vega``)."""
    fixed = column.startswith("fixed_")
    greek = CATALOG.get(column[6:] if fixed else column)
    if greek is None:
        return column
    return ("Fixed " if fixed else "Smile ") + greek.label[0].lower() + greek.label[1:]
