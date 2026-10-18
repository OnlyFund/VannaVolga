"""Option types: what kind of option a line or trade is.

Vanilla is the default: a line or trade without a type, including every
trade booked before types existed, is a vanilla. Only the types in
``PRICED`` are priced; any other type is reported as not priced yet and is
never priced as a vanilla. Types with a binary payout pay their notional in
the pair's base or term currency (``PAYOUT_TYPES``; base by default).
"""

VANILLA = "vanilla"
DIGITAL = "digital"
KNOCK_OUT = "knock_out"
KNOCK_IN = "knock_in"
ONE_TOUCH = "one_touch"
NO_TOUCH = "no_touch"
DOUBLE_NO_TOUCH = "double_no_touch"

#: Every type, in display order, with its label.
LABELS = {
    VANILLA: "Vanilla",
    DIGITAL: "Digital",
    KNOCK_OUT: "Knock-out",
    KNOCK_IN: "Knock-in",
    ONE_TOUCH: "One-touch",
    NO_TOUCH: "No-touch",
    DOUBLE_NO_TOUCH: "Double no-touch",
}
OPTION_TYPES = tuple(LABELS)
#: Types the app can price (digitals: ``pricing.digital``).
PRICED = frozenset({VANILLA, DIGITAL})
#: Types whose payout is a fixed amount in the base or term currency.
PAYOUT_TYPES = frozenset({DIGITAL, ONE_TOUCH, NO_TOUCH, DOUBLE_NO_TOUCH})

_BY_NAME = {**{key: key for key in LABELS},
            **{label.lower(): key for key, label in LABELS.items()}}


def _blank(value):
    if value is None:
        return True
    if isinstance(value, float) and value != value:          # NaN
        return True
    return not str(value).strip() or str(value).strip().lower() in ("nan", "none")


def option_type_of(value):
    """The type key for ``value`` (a key or a label, any case); blank is
    vanilla. Raises ``ValueError`` for an unknown type."""
    if _blank(value):
        return VANILLA
    text = str(value).strip().lower().replace("_", " ")
    key = _BY_NAME.get(text) or _BY_NAME.get(text.replace(" ", "_"))
    if key is None:
        raise ValueError(f"Option type must be one of {', '.join(LABELS.values())}, "
                         f"not {value!r}")
    return key


def label(option_type):
    """The display label of a type key (or the value itself if unknown)."""
    return LABELS.get(option_type, str(option_type))


def check_priced(value):
    """The type key for ``value``; raises ``ValueError`` when the app cannot
    price that type yet."""
    key = option_type_of(value)
    if key not in PRICED:
        raise ValueError(f"{label(key)} options are not priced yet")
    return key
