"""Analytical energy model (§8). All outputs are MODEL-BASED ESTIMATES.

  E_per_decision = E_sense + E_infer + E_comm(if alarm)
  E_hour         = duty_cycle * decisions_per_hour * E_per_decision + E_sleep

E_infer is derived from simulated cycles x datasheet energy/cycle for a named MCU.
For the cascade (H2), E_hour depends on the stage-0 gate wake-rate: only when the gate
fires does the stage-1 CNN run. That wake-rate is the lever the cascade paper pulls.
"""


def infer_energy_uj(sim, mcu):
    """Energy of one inference, in microjoules, from simulated cycles."""
    e_per_cycle_nj = mcu.get("energy_nj_per_cycle")
    if e_per_cycle_nj is None:
        # derive from active power / freq if given: P(mW)/f(MHz) = nJ/cycle
        e_per_cycle_nj = mcu["active_mw"] / mcu["freq_mhz"]
    return sim["sim_cycles"] * e_per_cycle_nj / 1000.0     # nJ -> uJ


def energy_report(sim, mcu, duty_cycle=0.01, decisions_per_hour=3600,
                  gate_wake_rate=None, e_sense_uj=5.0, e_comm_uj=0.0):
    """Return per-decision and per-hour energy estimates (uJ, mJ).

    gate_wake_rate: if set (cascade), stage-1 inference energy is multiplied by it
    (fraction of decisions where the gate fires); the gate's own cost is added.
    """
    e_infer = infer_energy_uj(sim, mcu)
    if gate_wake_rate is not None:
        e_gate = mcu.get("gate_uj", 0.5)
        e_decision = e_sense_uj + e_gate + gate_wake_rate * e_infer
    else:
        e_decision = e_sense_uj + e_infer
    sleep_uw = mcu.get("sleep_uw", 20.0)
    e_sleep_hour_mj = sleep_uw * 1e-6 * 3600 * 1e3         # W*s -> mJ  (uW->W:1e-6)
    e_active_hour_mj = duty_cycle * decisions_per_hour * e_decision / 1000.0
    return {
        "e_infer_uj": round(e_infer, 3),
        "e_decision_uj": round(e_decision, 3),
        "e_hour_mj": round(e_active_hour_mj + e_sleep_hour_mj, 3),
        "duty_cycle": duty_cycle,
        "gate_wake_rate": gate_wake_rate,
        "note": "model-based estimate (not measured)",
    }
