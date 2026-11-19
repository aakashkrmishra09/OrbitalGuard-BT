# OrbitalGuard-BT

**Pre-Execution Safety Validator & Simplex FDIR Simulation Harness for Autonomous Space Robotics**

[![CI](https://github.com/aakashkrmishra09/OrbitalGuard-BT/actions/workflows/ci.yml/badge.svg)](https://github.com/aakashkrmishra09/OrbitalGuard-BT/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)

🔗 **Live Interactive Fault Lab:** [https://orbitalguard-bt.onrender.com/](https://orbitalguard-bt.onrender.com/)  
📖 **Interactive OpenAPI Docs:** [https://orbitalguard-bt.onrender.com/docs](https://orbitalguard-bt.onrender.com/docs)

---

## Overview

In orbital and lunar robotics operations (such as proximity inspection, on-orbit servicing, and surface manipulation), communication latency and harsh thermal/optical environments require autonomous systems to validate commands locally and respond safely to telemetry faults without waiting for ground-in-the-loop intervention.

**OrbitalGuard-BT** is a full-stack verification prototype demonstrating a **Simplex Neuro-Symbolic FDIR (Fault Detection, Isolation, and Recovery) architecture**:
1. **Deterministic Pre-Execution Safety Guard:** Validates kinematic joint envelopes, slew-rate caps, minimum battery State-of-Charge (SoC), and keep-out sectors via strict Pydantic contracts *before* any execution tick starts.
2. **Behaviour Tree FDIR Controller:** Separates transient sensor degradation (such as solar glare, handled via bounded `HOLD_REACQUIRE` retries) from critical hardware/link faults (telemetry blackouts, thermal excursions, actuator overcurrent), which deterministically latch into `SAFE_HOLD`.
3. **Advisory Simplex ML Anomaly Detector (GMM + Conformal Prediction):** Monitors coupled multi-sensor telemetry inside nominal physical bounds to intercept subtle early degradation—such as joint bearing friction drift—via `PREEMPTIVE_HOLD` *before* catastrophic thresholds are breached.

> **Engineering Scope Note:** OrbitalGuard-BT is a software verification and simulation prototype built for architectural demonstration and testing; it is not flight-qualified avionics software.

---

## Simplex Safety Architecture

Aerospace Verification & Validation (V&V) workflows require deterministic, auditable guarantees on safety-critical paths. OrbitalGuard-BT enforces a strict separation of authority between hard physical invariants and statistical machine learning:

```text
                       ┌──────────────────────────────────────┐
                       │      Incoming Mission Command        │
                       └──────────────────┬───────────────────┘
                                          ▼
                       ┌──────────────────────────────────────┐
                       │   Pre-Execution Safety Guard         │
                       │ (Joint Limits, Slew Rate, SoC, KOZ)  │
                       └──────────────────┬───────────────────┘
                                  ┌───────┴───────┐
                             [REJECTED]      [APPROVED]
                                  │               ▼
                                  │   ┌───────────────────────┐
                                  │   │   Telemetry Stream    │
                                  │   └───────────┬───────────┘
                                  │               ▼
                                  │   ┌───────────────────────┐
                                  │   │ LAYER 1: Deterministic│ ──► Critical Fault ──► SAFE_HOLD (Latched)
                                  │   │ Hard Invariant Guard  │ ──► Low Vision     ──► HOLD_REACQUIRE
                                  │   └───────────┬───────────┘
                                  │          [Healthy]
                                  │               ▼
                                  │   ┌───────────────────────┐
                                  │   │ LAYER 2: Advisory ML  │ ──► 2-of-3 Anomaly ──► PREEMPTIVE_HOLD
                                  │   │ GMM + Conformal Score │
                                  │   └───────────┬───────────┘
                                  │          [Nominal]
                                  │               ▼
                                  │   ┌───────────────────────┐
                                  └──►│  NOMINAL_INSPECTION   │
                                      │     -> COMPLETED      │
                                      └───────────────────────┘