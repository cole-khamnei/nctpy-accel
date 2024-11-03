import time
import numpy as np

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


def check_system(system):
    """ """
    assert system in VALID_SYSTEMS, f"Invalid system '{system}', valid sytems: {VALID_SYSTEMS}"


# ------------------------------------------------------------------- #
# --------------------         NCT Utils         -------------------- #
# ------------------------------------------------------------------- #


def symmetric_matrix_norm(A, c=1, system="continuous"):
    """ """
    check_system(system)
    w, _ = np.linalg.eigh(A)
    l = np.abs(w).max()

    # Matrix normalization for discrete-time systems
    A_norm = A / (c + l)
    if system == 'continuous':
        # for continuous-time systems
        A_norm = A_norm - np.eye(A.shape[0])

    return A_norm


# ------------------------------------------------------------------- #
# --------------------            End            -------------------- #
# ------------------------------------------------------------------- #
