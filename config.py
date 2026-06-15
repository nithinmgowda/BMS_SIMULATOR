# =============================================================================
# config.py — BMS Simulator Constants & Pack Configuration
# =============================================================================
# All physical constants, pack parameters, algorithm tuning values, and
# protection thresholds live here. Never hardcode magic numbers elsewhere.
# =============================================================================

# -----------------------------------------------------------------------------
# CELL PARAMETERS (NMC 18650, 2 Ah nominal)
# -----------------------------------------------------------------------------

CELL_NOMINAL_CAPACITY_AH = 2.0          # Rated capacity [Ah]
CELL_NOMINAL_VOLTAGE_V   = 3.6          # Nominal voltage [V]
CELL_MAX_VOLTAGE_V       = 4.2          # Max charge voltage [V]
CELL_MIN_VOLTAGE_V       = 2.5          # Min discharge cutoff [V]
CELL_MAX_TEMP_C          = 60.0         # Max safe temperature [°C]
CELL_MIN_TEMP_C          = -20.0        # Min safe temperature [°C]

# Equivalent Circuit Model (1RC Thevenin)
CELL_R0_OHMS             = 0.025        # Internal (series) resistance [Ω]
CELL_R1_OHMS             = 0.010        # RC pair resistance [Ω]
CELL_C1_FARADS           = 3000.0       # RC pair capacitance [F]

# Cell-to-cell manufacturing variation (Gaussian spread, applied at init)
CELL_CAPACITY_VARIATION  = 0.02         # ±2% capacity spread
CELL_RESISTANCE_VARIATION= 0.05         # ±5% resistance spread

# -----------------------------------------------------------------------------
# THERMAL MODEL
# -----------------------------------------------------------------------------

CELL_THERMAL_MASS_J_K    = 30.0         # Thermal mass [J/K]
CELL_THERMAL_RESISTANCE  = 5.0          # Cell-to-ambient thermal resistance [K/W]
AMBIENT_TEMP_C           = 25.0         # Default ambient temperature [°C]

# -----------------------------------------------------------------------------
# PACK CONFIGURATION (8S1P)
# -----------------------------------------------------------------------------

NUM_CELLS_SERIES         = 8            # Cells in series
NUM_CELLS_PARALLEL       = 8            # Cells in parallel
NUM_CELLS                = NUM_CELLS_SERIES * NUM_CELLS_PARALLEL  # 64 total cells

PACK_NOMINAL_VOLTAGE_V   = CELL_NOMINAL_VOLTAGE_V * NUM_CELLS_SERIES    # 28.8 V
PACK_MAX_VOLTAGE_V       = CELL_MAX_VOLTAGE_V     * NUM_CELLS_SERIES    # 33.6 V
PACK_MIN_VOLTAGE_V       = CELL_MIN_VOLTAGE_V     * NUM_CELLS_SERIES    # 20.0 V
PACK_NOMINAL_CAPACITY_AH = CELL_NOMINAL_CAPACITY_AH * NUM_CELLS_PARALLEL  # 16.0 Ah

# -----------------------------------------------------------------------------
# SOC ESTIMATION
# -----------------------------------------------------------------------------

SOC_INITIAL              = 1.0          # Starting SOC (100%)
SOC_MIN                  = 0.0          # Lower bound
SOC_MAX                  = 1.0          # Upper bound

# Extended Kalman Filter tuning
EKF_PROCESS_NOISE_Q      = 1e-5         # Process noise covariance (SOC drift)
EKF_MEAS_NOISE_R         = 1e-3         # Measurement noise covariance (voltage sensor)
EKF_INIT_COVARIANCE_P    = 0.01         # Initial state estimation uncertainty

# Coulomb counting
COULOMB_EFFICIENCY       = 0.99         # Charge efficiency (accounts for losses)

# -----------------------------------------------------------------------------
# SOH ESTIMATION
# -----------------------------------------------------------------------------

SOH_INITIAL              = 1.0          # Fresh battery (100%)
EOL_SOH_THRESHOLD        = 0.70         # End-of-life at 70% capacity remaining

# Empirical degradation model coefficients
# Capacity fade: Q(n) = Q0 * exp(-ALPHA * n^BETA)
SOH_ALPHA                = 2.5e-4       # Degradation rate constant
SOH_BETA                 = 0.50         # Degradation exponent (sub-linear fade)

# Resistance growth: R(n) = R0 * (1 + GAMMA * n)
SOH_GAMMA                = 5.0e-4       # Resistance growth rate per cycle

# -----------------------------------------------------------------------------
# CELL BALANCING
# -----------------------------------------------------------------------------

BALANCING_MODE           = "passive"    # "passive" or "active"

# Passive balancing
PASSIVE_BALANCE_RESISTOR = 10.0         # Bleed resistor [Ω]
PASSIVE_BALANCE_THRESHOLD_V = 0.020     # Trigger balancing if ΔV > 20 mV

# Active balancing
ACTIVE_BALANCE_CURRENT_A = 0.1          # Transfer current between cells [A]
ACTIVE_BALANCE_THRESHOLD_V = 0.010      # Trigger if ΔV > 10 mV (tighter)
ACTIVE_BALANCE_EFFICIENCY = 0.92        # DC-DC converter efficiency

# -----------------------------------------------------------------------------
# FAULT PROTECTION THRESHOLDS
# -----------------------------------------------------------------------------

FAULT_OV_VOLTAGE_V       = 4.25         # Over-voltage per cell [V]
FAULT_UV_VOLTAGE_V       = 2.45         # Under-voltage per cell [V]
FAULT_OC_CHARGE_A        = 32.0         # Over-current during charge [A]  (4A × 8P)
FAULT_OC_DISCHARGE_A     = 64.0         # Over-current during discharge [A] (8A × 8P)
FAULT_OT_CELL_C          = 55.0         # Over-temperature cell [°C]
FAULT_OT_AMBIENT_C       = 45.0         # Over-temperature ambient [°C]
FAULT_IMBALANCE_V        = 0.200        # Cell imbalance alert threshold [V]

# -----------------------------------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------------------------------

SIM_TIMESTEP_S           = 1.0          # Simulation time step [seconds]
SIM_DEFAULT_DURATION_S   = 3600         # Default run: 1 hour
SIM_SPEED_MULTIPLIER     = 1            # 1x real-time (increase to fast-forward)

# -----------------------------------------------------------------------------
# LOGGING & OUTPUT
# -----------------------------------------------------------------------------

LOG_ENABLED              = True
LOG_INTERVAL_S           = 10           # Log every N simulation seconds
LOG_OUTPUT_PATH          = "logs/sim_output.csv"
