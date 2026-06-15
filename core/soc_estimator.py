# =============================================================================
# core/soc_estimator.py — SOC Estimation Algorithms
# =============================================================================
# Three SOC estimation algorithms running per cell:
#
#   1. Coulomb Counting (CC)
#      Simple current integration. Fast, but drifts over time due to
#      sensor noise and no correction mechanism.
#
#   2. OCV-Based Lookup
#      Maps resting terminal voltage → SOC via OCV-SOC table.
#      Accurate only at rest (no load). Used for initialisation.
#
#   3. Extended Kalman Filter (EKF)  ← primary estimator
#      Fuses Coulomb Counting (prediction) with voltage measurement
#      (correction) to eliminate drift. Handles ECM nonlinearity via
#      linearisation at each step.
#
#      State vector  : x = [SOC]
#      Input         : u = I (current)
#      Measurement   : z = Vt (terminal voltage)
#
#      Predict:
#        x_k|k-1 = x_k-1 - (I * dt) / Q
#        P_k|k-1 = P_k-1 + Q_noise
#
#      Update:
#        H_k  = dOCV/dSOC   (Jacobian from OCV table)
#        K_k  = P_k|k-1 * H_k' / (H_k * P_k|k-1 * H_k' + R_noise)
#        x_k  = x_k|k-1 + K_k * (Vt_measured - Vt_predicted)
#        P_k  = (1 - K_k * H_k) * P_k|k-1
#
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dataclasses import dataclass
from typing import List
from data.ocv_soc_table import get_ocv, get_soc_from_ocv, get_docv_dsoc
from config import (
    NUM_CELLS,
    SOC_INITIAL,
    SIM_TIMESTEP_S,
    COULOMB_EFFICIENCY,
    EKF_PROCESS_NOISE_Q,
    EKF_MEAS_NOISE_R,
    EKF_INIT_COVARIANCE_P,
)


# =============================================================================
# SOC ESTIMATE — per-cell result each tick
# =============================================================================

@dataclass
class SOCEstimate:
    """SOC estimates from all three algorithms for one cell."""
    cell_id     : int   = 0
    soc_cc      : float = 0.0    # Coulomb Counting estimate
    soc_ocv     : float = 0.0    # OCV lookup estimate (valid at rest only)
    soc_ekf     : float = 0.0    # EKF estimate (primary)
    ekf_variance: float = 0.0    # EKF error covariance (confidence measure)
    kalman_gain : float = 0.0    # Last Kalman gain (diagnostic)
    innovation  : float = 0.0    # Vt_measured - Vt_predicted (residual)


# =============================================================================
# ALGORITHM 1 — COULOMB COUNTING
# =============================================================================

class CoulombCounter:
    """
    Integrates current over time to estimate SOC.

    ΔSOC = -(η * I * dt) / Q_nominal
    η    = Coulombic efficiency (accounts for charge losses)

    Pros: Simple, low compute
    Cons: Drifts with sensor offset, no self-correction
    """

    def __init__(self, soc_init: float = SOC_INITIAL,
                 capacity_ah: float = 2.0):
        self.soc         = float(np.clip(soc_init, 0.0, 1.0))
        self.capacity_as = capacity_ah * 3600.0
        self.dt          = SIM_TIMESTEP_S
        self.efficiency  = COULOMB_EFFICIENCY

    def update(self, current: float) -> float:
        """
        Update SOC estimate with new current measurement.

        Args:
            current : Cell current [A] (+ = discharge, - = charge)

        Returns:
            Updated SOC [0.0 → 1.0]
        """
        # Apply efficiency only during charging (current < 0)
        eta = self.efficiency if current < 0 else 1.0
        delta_soc = -(eta * current * self.dt) / self.capacity_as
        self.soc  = float(np.clip(self.soc + delta_soc, 0.0, 1.0))
        return self.soc

    def reset(self, soc_init: float):
        self.soc = float(np.clip(soc_init, 0.0, 1.0))


# =============================================================================
# ALGORITHM 2 — OCV LOOKUP
# =============================================================================

class OCVLookup:
    """
    Maps terminal voltage to SOC via OCV-SOC table.
    Only valid when current ≈ 0 (battery at rest, no polarisation).
    Used at:
      - Startup initialisation
      - After long rest periods (>30 min)
      - As a correction checkpoint for CC drift
    """

    REST_CURRENT_THRESHOLD_A = 0.05   # Below this, battery is "at rest"

    def estimate(self, voltage: float, current: float,
                 temp_c: float = 25.0) -> float:
        """
        Estimate SOC from terminal voltage.

        Args:
            voltage : Terminal voltage [V]
            current : Current [A] — used to gate validity
            temp_c  : Cell temperature [°C]

        Returns:
            SOC estimate [0.0 → 1.0], or None if not at rest
        """
        if abs(current) > self.REST_CURRENT_THRESHOLD_A:
            return None     # Not valid under load
        return get_soc_from_ocv(voltage, temp_c=temp_c)

    def estimate_forced(self, voltage: float, temp_c: float = 25.0) -> float:
        """Force OCV-based estimate regardless of current (startup use)."""
        return get_soc_from_ocv(voltage, temp_c=temp_c)


# =============================================================================
# ALGORITHM 3 — EXTENDED KALMAN FILTER (EKF)
# =============================================================================

class ExtendedKalmanFilter:
    """
    EKF-based SOC estimator for a single Li-ion cell.

    State    : x = SOC  (scalar, 1D state)
    Input    : u = I    (current [A])
    Measured : z = Vt   (terminal voltage [V])

    Process model (Coulomb Counting):
        x_k = x_k-1 - (I * dt) / Q

    Observation model (ECM terminal voltage):
        z_k = OCV(x_k) - I*R0 - V_RC

    Jacobian of observation model w.r.t. SOC:
        H = dOCV/dSOC  (from OCV table numerical derivative)

    Tuning:
        Q_noise : Process noise — how much we trust CC prediction
                  Higher Q → faster response but more noisy
        R_noise : Measurement noise — how much we trust voltage sensor
                  Higher R → smoother but slower correction
    """

    def __init__(
        self,
        soc_init     : float = SOC_INITIAL,
        capacity_ah  : float = 2.0,
        r0           : float = 0.025,
        process_noise: float = EKF_PROCESS_NOISE_Q,
        meas_noise   : float = EKF_MEAS_NOISE_R,
        init_cov     : float = EKF_INIT_COVARIANCE_P,
    ):
        self.capacity_as  = capacity_ah * 3600.0
        self.r0           = r0
        self.dt           = SIM_TIMESTEP_S

        # EKF noise matrices (scalar since 1D state)
        self.Q = process_noise    # Process noise covariance
        self.R = meas_noise       # Measurement noise covariance

        # State and covariance
        self.x = float(np.clip(soc_init, 0.0, 1.0))   # SOC estimate
        self.P = init_cov                               # Error covariance

        # Diagnostics
        self.kalman_gain = 0.0
        self.innovation  = 0.0

    def update(
        self,
        current   : float,
        voltage_measured: float,
        v_rc      : float = 0.0,
        temp_c    : float = 25.0,
    ) -> float:
        """
        Run one EKF predict-update cycle.

        Args:
            current          : Cell current [A] (+ discharge, - charge)
            voltage_measured : Measured terminal voltage [V]
            v_rc             : RC pair voltage from ECM [V]
            temp_c           : Cell temperature [°C]

        Returns:
            Updated SOC estimate [0.0 → 1.0]
        """

        # ── PREDICT STEP ──────────────────────────────────────────────────
        # State prediction via Coulomb Counting
        x_pred = self.x - (current * self.dt) / self.capacity_as
        x_pred = float(np.clip(x_pred, 0.0, 1.0))

        # Covariance prediction
        # F = dx_k/dx_k-1 = 1 (linear state transition)
        P_pred = self.P + self.Q

        # ── UPDATE STEP ───────────────────────────────────────────────────
        # Predicted terminal voltage from ECM model
        ocv_pred    = get_ocv(x_pred, temp_c=temp_c)
        v_predicted = ocv_pred - current * self.r0 - v_rc

        # Innovation (measurement residual)
        self.innovation = voltage_measured - v_predicted

        # Observation Jacobian H = dVt/dSOC = dOCV/dSOC
        H = get_docv_dsoc(x_pred)

        # Innovation covariance
        S = H * P_pred * H + self.R

        # Kalman gain
        self.kalman_gain = (P_pred * H) / S

        # State update
        self.x = x_pred + self.kalman_gain * self.innovation
        self.x = float(np.clip(self.x, 0.0, 1.0))

        # Covariance update (Joseph form for numerical stability)
        self.P = (1.0 - self.kalman_gain * H) * P_pred

        return self.x

    def reset(self, soc_init: float, init_cov: float = EKF_INIT_COVARIANCE_P):
        self.x = float(np.clip(soc_init, 0.0, 1.0))
        self.P = init_cov

    def update_capacity(self, new_capacity_ah: float):
        """Called by SOH estimator when capacity degrades."""
        self.capacity_as = new_capacity_ah * 3600.0

    def update_r0(self, new_r0: float):
        """Called by SOH estimator when resistance grows."""
        self.r0 = new_r0


# =============================================================================
# PACK-LEVEL SOC ESTIMATOR — manages one estimator set per cell
# =============================================================================

class PackSOCEstimator:
    """
    Manages SOC estimation for all 64 cells in the pack.
    Runs CC, OCV, and EKF for each cell every tick.

    Used by sim_loop.py — call update() each tick with pack state.
    """

    def __init__(self, pack_model):
        """
        Args:
            pack_model : PackModel instance (to read cell parameters)
        """
        self.n_cells = NUM_CELLS
        self.ocv_lookup = OCVLookup()

        # One CC + EKF per cell, initialised from cell parameters
        self.cc_estimators  : List[CoulombCounter]      = []
        self.ekf_estimators : List[ExtendedKalmanFilter] = []

        for cell in pack_model.all_cells:
            # Initialise CC from cell's internal SOC
            self.cc_estimators.append(
                CoulombCounter(
                    soc_init    = cell.soc,
                    capacity_ah = cell.capacity_ah,
                )
            )
            # Initialise EKF — use OCV-based SOC as starting estimate
            soc_ocv_init = get_soc_from_ocv(cell.get_ocv())
            self.ekf_estimators.append(
                ExtendedKalmanFilter(
                    soc_init    = soc_ocv_init,
                    capacity_ah = cell.capacity_ah,
                    r0          = cell.r0,
                )
            )

    def update(self, cell_states) -> List[SOCEstimate]:
        """
        Update all SOC estimators from latest cell states.

        Args:
            cell_states : List of CellState from PackModel.step()

        Returns:
            List of SOCEstimate (one per cell)
        """
        estimates = []

        for i, state in enumerate(cell_states):
            cc  = self.cc_estimators[i]
            ekf = self.ekf_estimators[i]

            # Coulomb Counting update
            soc_cc = cc.update(current=state.current)

            # OCV lookup (valid only at rest)
            soc_ocv_raw = self.ocv_lookup.estimate(
                voltage=state.voltage,
                current=state.current,
                temp_c=state.temperature,
            )
            soc_ocv = soc_ocv_raw if soc_ocv_raw is not None else ekf.x

            # EKF update
            soc_ekf = ekf.update(
                current          = state.current,
                voltage_measured = state.voltage,
                v_rc             = state.v_rc,
                temp_c           = state.temperature,
            )

            estimates.append(SOCEstimate(
                cell_id      = state.cell_id,
                soc_cc       = soc_cc,
                soc_ocv      = soc_ocv,
                soc_ekf      = soc_ekf,
                ekf_variance = ekf.P,
                kalman_gain  = ekf.kalman_gain,
                innovation   = ekf.innovation,
            ))

        return estimates

    def get_pack_soc(self, estimates: List[SOCEstimate]) -> float:
        """Mean EKF SOC across all cells — pack-level SOC."""
        return float(np.mean([e.soc_ekf for e in estimates]))

    def reinitialise_from_ocv(self, pack_model):
        """
        Hard-reset all estimators using OCV-based SOC.
        Call after a long rest period for drift correction.
        """
        for i, cell in enumerate(pack_model.all_cells):
            soc_ocv = get_soc_from_ocv(cell.get_ocv(), temp_c=cell.temperature)
            self.cc_estimators[i].reset(soc_ocv)
            self.ekf_estimators[i].reset(soc_ocv)

    def notify_soh_update(self, cell_id: int,
                          new_capacity_ah: float, new_r0: float):
        """
        Propagate SOH-driven parameter changes into the EKF and CC.
        Called by soh_estimator.py when capacity or resistance changes.
        """
        self.ekf_estimators[cell_id].update_capacity(new_capacity_ah)
        self.ekf_estimators[cell_id].update_r0(new_r0)
        self.cc_estimators[cell_id].capacity_as = new_capacity_ah * 3600.0


# =============================================================================
# DIAGNOSTICS — compare CC vs EKF on a single cell with injected noise
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from core.cell_model import CellModel
    from data.load_profiles import get_profile

    print("=" * 55)
    print("SOC Estimator — CC vs EKF Comparison (with noise)")
    print("=" * 55)

    cell    = CellModel(cell_id=0, soc_init=1.0)
    profile = get_profile("constant_1c")
    I_cell  = 2.0   # 1C per cell

    cc  = CoulombCounter(soc_init=1.0, capacity_ah=cell.capacity_ah)
    ekf = ExtendedKalmanFilter(soc_init=1.0, capacity_ah=cell.capacity_ah, r0=cell.r0)

    rng = np.random.default_rng(0)

    times, true_soc, cc_soc, ekf_soc, innovations = [], [], [], [], []

    t = 0
    while cell.soc > 0.05 and t < 7200:
        # True cell step
        state = cell.step(current=I_cell)

        # Add voltage sensor noise (±5mV)
        v_noisy = state.voltage + rng.normal(0, 0.005)

        # Add current sensor noise (±10mA)
        i_noisy = state.current + rng.normal(0, 0.01)

        soc_cc  = cc.update(i_noisy)
        soc_ekf = ekf.update(i_noisy, v_noisy, v_rc=state.v_rc,
                             temp_c=state.temperature)

        times.append(t)
        true_soc.append(state.soc * 100)
        cc_soc.append(soc_cc * 100)
        ekf_soc.append(soc_ekf * 100)
        innovations.append(ekf.innovation * 1000)   # mV

        t += SIM_TIMESTEP_S

    # Errors
    cc_err  = np.abs(np.array(cc_soc)  - np.array(true_soc))
    ekf_err = np.abs(np.array(ekf_soc) - np.array(true_soc))
    print(f"  CC  MAE : {np.mean(cc_err):.4f}%  |  RMSE: {np.sqrt(np.mean(cc_err**2)):.4f}%")
    print(f"  EKF MAE : {np.mean(ekf_err):.4f}%  |  RMSE: {np.sqrt(np.mean(ekf_err**2)):.4f}%")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("SOC Estimation — CC vs EKF (with sensor noise)", fontweight="bold")

    axes[0].plot(times, true_soc, "k-",  linewidth=2.0, label="True SOC")
    axes[0].plot(times, cc_soc,  "b--", linewidth=1.2, label="Coulomb Counting")
    axes[0].plot(times, ekf_soc, "r-",  linewidth=1.2, label="EKF")
    axes[0].set_ylabel("SOC [%]")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, cc_err,  "b--", linewidth=1.2, label="CC error")
    axes[1].plot(times, ekf_err, "r-",  linewidth=1.2, label="EKF error")
    axes[1].set_ylabel("Absolute Error [%]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, innovations, color="purple", linewidth=1.0)
    axes[2].axhline(0, color="gray", linewidth=0.6, linestyle="--")
    axes[2].set_ylabel("EKF Innovation [mV]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/soc_estimator_comparison.png", dpi=120)
    plt.show()