import time

# ------------------------------------------------------------------- #
# --------------------      Random Helpers       -------------------- #
# ------------------------------------------------------------------- #

TIMER_PRINT = True

class Timer:
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        
        if TIMER_PRINT:
            print(f"Duration {self.label}: {self.duration:0.5f} seconds")


# ------------------------------------------------------------------- #
# --------------------    JAX Precompile LUT     -------------------- #
# ------------------------------------------------------------------- #

VALID_SYSTEMS = ["discrete", "continuous"]


def system_check(system):
    """ """
    assert system in VALID_SYSTEMS, f"Invalid system '{system}', valid sytems: {VALID_SYSTEMS}"


# ------------------------------------------------------------------- #
# --------------------            End            -------------------- #
# ------------------------------------------------------------------- #
