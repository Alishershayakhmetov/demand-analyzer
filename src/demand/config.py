from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

STORE = "CA_1"
STATE = "CA"
N_PER_STRATUM = 100
SEED = 42

# M5 has 1969 daily observations (2011-01-29 .. 2016-06-19), all with known labels.
LAST_DAY = 1969
HORIZON = 28

# A fold trains on d <= cutoff and forecasts d = cutoff+1 .. cutoff+HORIZON.
# The last fold is the test fold and is evaluated only once, at the very end.
DEV_CUTOFFS = [1829, 1857, 1885]
VAL_CUTOFF = 1913
TEST_CUTOFF = LAST_DAY - HORIZON
SELECTION_CUTOFF = DEV_CUTOFFS[0]
