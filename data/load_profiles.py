# =============================================================================
# data/load_profiles.py — Current Load Profiles
# =============================================================================
# Defines current demand profiles applied to the battery pack each sim tick.
# All profiles return current in Amperes at a given time step.
#
# Sign convention (standard BMS):
#   Positive current (+A) = Discharge (current leaving the pack)
#   Negative current (-A) = Charge   (current entering the pack)
#
# Usage:
#   from data.load_profiles import get_profile
#   profile = get_profile("udds")
#   current = profile.get_current(t=120)   # current at t=120s
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from config import PACK_NOMINAL_CAPACITY_AH

# 1C current for this pack (scales with capacity)
I_1C = PACK_NOMINAL_CAPACITY_AH   # 16.0 A for 8S8P 16Ah pack


# =============================================================================
# BASE CLASS
# =============================================================================

class LoadProfile:
    """Base class for all load profiles."""

    name        : str = "base"
    description : str = ""

    def get_current(self, t: float) -> float:
        """
        Returns demanded current [A] at simulation time t [seconds].
        Override in subclasses.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"<LoadProfile: {self.name}>"


# =============================================================================
# PROFILE 1 — Constant Current Discharge
# =============================================================================

class ConstantCurrentProfile(LoadProfile):
    """
    Steady discharge or charge at a fixed C-rate.
    Most common for baseline testing and benchmarking.

    Args:
        c_rate : C-rate multiplier (e.g. 1.0 = 1C, 0.5 = C/2, -1.0 = charge)
    """
    name        = "constant_current"
    description = "Fixed C-rate discharge or charge"

    def __init__(self, c_rate: float = 1.0):
        self.c_rate  = c_rate
        self.current = c_rate * I_1C       # [A]

    def get_current(self, t: float) -> float:
        return self.current


# =============================================================================
# PROFILE 2 — Step Profile (HPPC-style)
# =============================================================================

class StepProfile(LoadProfile):
    """
    Hybrid Pulse Power Characterization (HPPC) style profile.
    Used for internal resistance estimation — alternates between
    discharge pulses and rest periods.

    Cycle: 10s discharge → 40s rest → 10s regen → 40s rest
    """
    name        = "step_hppc"
    description = "HPPC pulse profile for resistance estimation"

    CYCLE_S     = 100       # Total cycle duration [s]
    PULSE_A     = 2.0 * I_1C   # 2C pulse current [A]

    def get_current(self, t: float) -> float:
        phase = t % self.CYCLE_S
        if phase < 10:
            return self.PULSE_A          # Discharge pulse
        elif phase < 50:
            return 0.0                   # Rest
        elif phase < 60:
            return -self.PULSE_A * 0.75  # Regen pulse (75% of discharge)
        else:
            return 0.0                   # Rest


# =============================================================================
# PROFILE 3 — UDDS (Urban Dynamometer Driving Schedule)
# =============================================================================

class UDDSProfile(LoadProfile):
    """
    Urban city driving cycle approximation.
    Models stop-and-go traffic: acceleration bursts, coasting, braking regen.
    One UDDS cycle ~ 1369 seconds.

    Simplified as a sinusoidal + noise approximation of the real UDDS trace.
    """
    name        = "udds"
    description = "Urban driving cycle — city stop-and-go"

    CYCLE_S     = 1369      # UDDS cycle duration [s]

    def get_current(self, t: float) -> float:
        phase = t % self.CYCLE_S

        # Base sinusoidal drive demand
        base = I_1C * 0.8 * np.sin(2 * np.pi * phase / 120)

        # Acceleration spikes every ~30s
        spike = I_1C * 1.5 * np.exp(-((phase % 30 - 5) ** 2) / 8)

        # Regen braking events (negative current) every ~45s
        regen = -I_1C * 0.6 * np.exp(-((phase % 45 - 40) ** 2) / 6)

        # Idle periods (near zero current) every ~90s for ~10s
        idle_mask = 1.0 if (phase % 90) > 10 else 0.05

        current = (base + spike + regen) * idle_mask

        # Clamp to safe pack limits
        return float(np.clip(current, -I_1C * 1.0, I_1C * 2.5))


# =============================================================================
# PROFILE 4 — CC-CV Charge (Constant Current → Constant Voltage)
# =============================================================================

class CCCVChargeProfile(LoadProfile):
    """
    Standard CC-CV charging protocol:
      Phase 1 (CC): Charge at 0.5C until pack reaches max voltage
      Phase 2 (CV): Taper current to hold voltage; ends when I < 0.05C

    The sim_loop handles voltage-based phase switching.
    This profile just returns the CC phase current — sim_loop
    overrides with CV taper logic when near full charge.
    """
    name        = "cc_cv_charge"
    description = "Standard CC-CV charge profile"

    def __init__(self, c_rate: float = 0.5):
        self.c_rate  = c_rate
        self.current = -c_rate * I_1C    # Negative = charging

    def get_current(self, t: float) -> float:
        return self.current


# =============================================================================
# PROFILE 5 — Random Walk (Realistic Variable Load)
# =============================================================================

class RandomWalkProfile(LoadProfile):
    """
    Stochastic variable load simulating real-world unpredictable usage.
    Uses seeded random walk so results are reproducible.

    Useful for SOC/SOH algorithm stress testing.
    """
    name        = "random_walk"
    description = "Reproducible stochastic variable load"

    def __init__(self, seed: int = 42, volatility: float = 0.3):
        self.seed       = seed
        self.volatility = volatility
        self._build_trace()

    def _build_trace(self, duration_s: int = 7200):
        """Pre-generate a 2-hour random walk current trace."""
        rng     = np.random.default_rng(self.seed)
        steps   = rng.normal(0, self.volatility * I_1C, duration_s)
        trace   = np.cumsum(steps)
        # Centre around 0.5C discharge baseline, clip to safe range
        trace   = trace - np.mean(trace) + 0.5 * I_1C
        self._trace = np.clip(trace, -I_1C, 2.0 * I_1C)

    def get_current(self, t: float) -> float:
        idx = int(t) % len(self._trace)
        return float(self._trace[idx])


# =============================================================================
# PROFILE REGISTRY & FACTORY
# =============================================================================

_PROFILE_REGISTRY = {
    "constant_1c"    : lambda: ConstantCurrentProfile(c_rate=1.0),
    "constant_2c"    : lambda: ConstantCurrentProfile(c_rate=2.0),
    "constant_half_c": lambda: ConstantCurrentProfile(c_rate=0.5),
    "step_hppc"      : lambda: StepProfile(),
    "udds"           : lambda: UDDSProfile(),
    "cc_cv_charge"   : lambda: CCCVChargeProfile(c_rate=0.5),
    "random_walk"    : lambda: RandomWalkProfile(),
}


def get_profile(name: str) -> LoadProfile:
    """
    Factory function — returns a LoadProfile instance by name.

    Available profiles:
        'constant_1c'     — 1C constant discharge
        'constant_2c'     — 2C constant discharge
        'constant_half_c' — C/2 constant discharge
        'step_hppc'       — HPPC pulse profile
        'udds'            — Urban driving cycle
        'cc_cv_charge'    — CC-CV charge at 0.5C
        'random_walk'     — Stochastic variable load

    Args:
        name : Profile key string (see above)

    Returns:
        LoadProfile instance

    Raises:
        ValueError if name not found
    """
    if name not in _PROFILE_REGISTRY:
        raise ValueError(
            f"Unknown profile '{name}'. "
            f"Available: {list(_PROFILE_REGISTRY.keys())}"
        )
    return _PROFILE_REGISTRY[name]()


def list_profiles() -> list:
    """Returns list of all available profile names."""
    return list(_PROFILE_REGISTRY.keys())


# =============================================================================
# DIAGNOSTICS — run directly to preview profile shapes
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    profiles_to_plot = ["constant_1c", "step_hppc", "udds", "random_walk", "cc_cv_charge"]
    t = np.arange(0, 1369, 1)

    fig, axes = plt.subplots(len(profiles_to_plot), 1, figsize=(12, 10), sharex=False)
    fig.suptitle("BMS Simulator — Load Profiles", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, profiles_to_plot):
        profile = get_profile(name)
        currents = [profile.get_current(ti) for ti in t]
        ax.plot(t, currents, linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(f"{name}  |  {profile.description}", fontsize=9)
        ax.set_ylabel("Current [A]")
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/load_profiles_preview.png", dpi=120)
    plt.show()
    print("\nAvailable profiles:", list_profiles())