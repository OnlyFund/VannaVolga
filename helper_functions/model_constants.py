import sys
from math import sqrt

# -------------------------------
# Machine precision and limits
# -------------------------------
DBL_EPS = sys.float_info.epsilon
DBL_MIN = sys.float_info.min
DBL_MAX = sys.float_info.max

SQRT_DBL_EPS = sqrt(DBL_EPS)
FORTH_ROOT_DBL_EPS = sqrt(SQRT_DBL_EPS)
EIGHT_ROOT_DBL_EPS = sqrt(FORTH_ROOT_DBL_EPS)
SIXTEEN_ROOT_DBL_EPS = sqrt(EIGHT_ROOT_DBL_EPS)
SQRT_DBL_MIN = sqrt(DBL_MIN)
SQRT_DBL_MAX = sqrt(DBL_MAX)

VOL_VALUE_TO_SIGNAL_PX_IS_BELOW_MIN = -DBL_MAX
VOL_VALUE_TO_SIGNAL_PX_IS_ABOVE_MAX = DBL_MAX

# -------------------------------
# Fundamental constants
# -------------------------------
ONE_OVER_SQRT_TWO = 1 / sqrt(2)
SQRT_TWO_PI = sqrt(2 * 3.14159265358979323846264338327950288)
INV_SQRT_TWO_PI = 1 / SQRT_TWO_PI
ONE_OVER_SQRT_TWO_PI = INV_SQRT_TWO_PI
TWO_PI = 2 * 3.14159265358979323846264338327950288

SQRT_PI_OVER_TWO = sqrt(3.14159265358979323846264338327950288 / 2)
SQRT_THREE = sqrt(3)
SQRT_ONE_OVER_THREE = 1 / SQRT_THREE
TWO_PI_OVER_SQRT_TWENTY_SEVEN = TWO_PI / sqrt(27)
PI_OVER_SIX = 3.14159265358979323846264338327950288 / 6
