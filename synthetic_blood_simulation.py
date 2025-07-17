"""
Synthetic Blood Supply Chain Simulation

This discrete-event simulation models the production, storage, and deployment of 
synthetic blood products (HBOCs and PFCs) in emergency and resource-limited settings 
clinical settings.

Paper: "Simulation-Based Optimization of Synthetic Blood Production and 
       Deployment in Emergency and Resource-Limited Clinical Settings"
Conference: MODSIM World 2025
Lead Researcher: Soraya Hani
Code Developer: Ancuta Margondai
Institution: University of Central Florida

Usage: python synthetic_blood_simulation.py

Dependencies: simpy, numpy, matplotlib, pandas
"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import time
import csv
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Parameters (from literature)
HBOC_PROD_TIME = (5, 7)  # days [Jahr, 2022]

# Parameters (from literature)
HBOC_PROD_TIME = (5, 7)  # days [Jahr, 2022]
PFC_PROD_TIME = (3, 5)  # days [Kim et al., 2024; Vichare & Janjic, 2025]
HBOC_BATCH_SIZE = (500, 1000)  # liters [Jahr, 2022]
PFC_BATCH_SIZE = (100, 300)  # liters [Kim et al., 2024]
HBOC_UNIT_SIZE = 0.25  # liters/unit [Jahr, 2022]
PROD_FAILURE_RATE = 0.075  # 7.5% [Khan et al., 2020; Kim et al., 2024]
HBOC_SHELF_LIFE = 2 * 365 * 24  # hours [Chang, 2014]
PFC_SHELF_LIFE = 1.5 * 365 * 24  # hours [Riess, 2001]
LAND_DELIVERY_TIME = (2, 6)  # hours [Roberts et al., 2018]
DRONE_DELIVERY_TIME = 0.5  # hours [Roberts et al., 2018]
DRONE_FAILURE_RATE = 0.018  # [Glick et al., 2020]
WEATHER_FAILURE_RATE = 0.023  # [Glick et al., 2020]
TRAUMA_DEMAND = (4, 10)  # units [Holcomb et al., 2005]
HBOC_AUTOXIDATION_RATE = 0.22  # per hour [Estep, 2025]
HBOC_MI_RISK = 0.0201  # per patient [Estep, 2025]
PFC_CARPA_RISK = 0.01  # estimated [Kim et al., 2024]
HBOC_COST = 10000  # $/unit [Estrada et al., 2025]
DRONE_COST = 75000  # $ [Glick et al., 2020]
SIM_TIME = 90 * 24  # 90 days

# Part 2: Sensitivity Analysis Parameters (Reduced for Validation)
THRESHOLDS_HBOC = [8]  # units
THRESHOLDS_PFC = [16]  # liters
DRONE_SPLITS = [0.75]
ARRIVAL_RATES_PART2 = [4.8, 6]  # hours (include conflict scenario)
PFC_DEG_RATES = [1e-05]  # per hour [Vichare & Janjic, 2025; Freire et al., 2005]
PFC_COSTS = [2000]  # $/liter [Vichare & Janjic, 2025]
ITERATIONS = 100

# Part 3: Real-World Parameters
FACILITY_PROFILES = {
    "urban_hospital": {"location": "urban", "delay_factor": 1.0, "failure_boost": 0.00, "arrival_rate": 6},  # ~4 patients/day
    "conflict_clinic": {"location": "conflict", "delay_factor": 2.5, "failure_boost": 0.05, "arrival_rate": 2.4}  # ~10 patients/day
}
NUM_DRONES = 2  # Shared drones for Part 3
DISASTER_MODE = True  # Enabled for surge demand simulation
DISASTER_ARRIVAL_RATE = 2  # hours (12 patients/day during surge)
DISASTER_DURATION = 24  # hours

# Part 4: Case Study Parameters (Conflict Zone Hospital from Elsayed et al., 2022)
CASE_STUDY_PROFILE = {
    "conflict_hospital": {
        "location": "conflict",
        "delay_factor": 2.5,  # WHO, 2021
        "failure_boost": 0.05,
        "arrival_rate": 1.692  # ~35.4 units/day, avg 7 units/patient = ~5 patients/day = 1 patient every 1.692 hours
    }
}
DRONE_SPLITS_PART4 = [0.9]  # Policy 1: Increase drone split to 90%
INITIAL_HBOC_PART4 = 200  # Policy 2: Pre-position 200 HBOC units
INITIAL_PFC_PART4 = 400  # Policy 2: Pre-position 400 PFC liters
NUM_DRONES_PART4 = 1  # Single facility setup

# Markov Chain (updated with sourced data: Smith et al., 2023)
markov_states = ["Operational", "Delayed", "Emergency", "Failure"]
base_transition_matrix = np.array([
    [0.70, 0.15, 0.10, 0.05],  # Operational
    [0.20, 0.60, 0.15, 0.05],  # Delayed
    [0.05, 0.20, 0.60, 0.15],  # Emergency
    [0.00, 0.10, 0.20, 0.70]  # Failure
])

# Location Profiles (updated conflict delay factor: WHO, 2021)
base_location_profiles = {
    "urban": {"location": "urban", "delay_factor": 1.0, "failure_boost": 0.00},
    "rural": {"location": "rural", "delay_factor": 1.2, "failure_boost": 0.01},
    "remote": {"location": "remote", "delay_factor": 1.5, "failure_boost": 0.03},
    "conflict": {"location": "conflict", "delay_factor": 2.5, "failure_boost": 0.05}
}

# Metrics
metrics = {}

# PFC Equivalence (updated: Spahn et al., 2018)
PFC_PER_HBOC_UNIT = 0.5  # liters of PFC equivalent to 1 HBOC unit (0.25 L)

def perturb_matrix(matrix, perturbation=0.1):
    """Perturb Markov transition matrix."""
    perturbed = matrix.copy()
    for i in range(len(matrix)):
        row = matrix[i] + np.random.uniform(-perturbation, perturbation, size=len(matrix))
        row = np.clip(row, 0, 1)
        row /= row.sum() if row.sum() > 0 else 1
        perturbed[i] = row
    return perturbed

def perturb_profiles(profiles, perturbation=0.2):
    """Perturb location profiles."""
    perturbed = {}
    for loc, params in profiles.items():
        perturbed[loc] = {
            "location": params["location"],
            "delay_factor": max(1.0, params["delay_factor"] * (1 + random.uniform(-perturbation, perturbation))),
            "failure_boost": max(0.0, params["failure_boost"] + random.uniform(-perturbation * 0.05, perturbation * 0.05))
        }
    return perturbed

def system_status_update(env, transition_matrix):
    """Update system status via Markov chain."""
    global current_status
    while True:
        current_status = np.random.choice(range(4), p=transition_matrix[current_status])
        metrics["state_transitions"].append((env.now, markov_states[current_status]))
        yield env.timeout(24)

def get_status_multiplier():
    """Return multiplier based on system status."""
    return [1.0, 1.2, 1.5, 2.0][current_status]

def production(env, product, store, prod_time, batch_size, failure_rate, pfc_cost):
    """Simulate production."""
    while True:
        yield env.timeout(random.uniform(*prod_time) * 24 * get_status_multiplier())
        batch = random.uniform(*batch_size)
        if random.random() > failure_rate:
            yield store.put(batch)
            cost = batch * (HBOC_COST / HBOC_UNIT_SIZE if product == "HBOC" else pfc_cost)
            metrics["total_cost"] += cost
        else:
            metrics["wasted_units"][product] += batch

def delivery(env, product, store, hospital, drone, units_needed, drone_split, facility_profile, delay_causes_file):
    """Simulate delivery (updated for multi-facility)."""
    with drone.request() as req:
        yield req
        amount = units_needed * HBOC_UNIT_SIZE if product == "HBOC" else units_needed * PFC_PER_HBOC_UNIT
        if store.level >= amount:
            yield store.get(amount)
            loc_factor = facility_profile["delay_factor"]
            location = facility_profile["location"]
            if random.random() < drone_split:
                delay = DRONE_DELIVERY_TIME * loc_factor * get_status_multiplier()
                failure_rate = DRONE_FAILURE_RATE + WEATHER_FAILURE_RATE + facility_profile["failure_boost"]
            else:
                delay = random.uniform(*LAND_DELIVERY_TIME) * loc_factor * get_status_multiplier()
                failure_rate = WEATHER_FAILURE_RATE + facility_profile["failure_boost"]
            yield env.timeout(delay)
            metrics["delivery_delays"].append(delay)
            cause = {"state": markov_states[current_status], "location": location, "delay": delay, "facility": hospital.name}
            with open(delay_causes_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["state", "location", "delay", "facility"])
                writer.writerow(cause)
            if random.random() > failure_rate:
                yield hospital.put(amount)
            else:
                metrics["stockouts"] += units_needed
        else:
            metrics["stockouts"] += units_needed

def trauma_demand(env, facility_name, facility_profile, hboc_store, pfc_store, hospitals, drones, drone_split, delay_causes_file, location_profiles=None):
    """Simulate trauma demand for a facility (updated for multi-facility and disaster mode)."""
    hospital = hospitals[facility_name]
    base_arrival_rate = facility_profile["arrival_rate"]
    start_time = env.now
    while True:
        if DISASTER_MODE and (env.now - start_time) < DISASTER_DURATION:
            arrival_rate = DISASTER_ARRIVAL_RATE
        else:
            arrival_rate = base_arrival_rate
        yield env.timeout(random.expovariate(1 / arrival_rate))
        units_needed = random.randint(*TRAUMA_DEMAND)
        metrics["total_demand"] += units_needed
        drone = random.choice(drones)
        if location_profiles:
            single_profile = random.choice(list(location_profiles.values()))
        else:
            single_profile = facility_profile
        if hboc_store.level >= units_needed * HBOC_UNIT_SIZE:
            env.process(delivery(env, "HBOC", hboc_store, hospital, drone, units_needed, drone_split, single_profile, delay_causes_file))
            if random.random() < HBOC_MI_RISK:
                metrics["mi_incidents"] += 1
        elif pfc_store.level >= units_needed * PFC_PER_HBOC_UNIT:
            env.process(delivery(env, "PFC", pfc_store, hospital, drone, units_needed, drone_split, single_profile, delay_causes_file))
            if random.random() < PFC_CARPA_RISK:
                metrics["carpa_incidents"] += 1
        else:
            metrics["stockouts"] += units_needed

def degradation(env, product, store, degradation_rate):
    """Simulate degradation."""
    while True:
        yield env.timeout(24)
        if store.level > 0:
            loss = store.level * degradation_rate * 24
            yield store.get(loss)
            metrics["wasted_units"][product] += loss

def inventory_management(env, product, store, hospitals, drones, threshold, pfc_cost):
    """Manage inventory (updated for multi-facility)."""
    while True:
        threshold_val = threshold * HBOC_UNIT_SIZE if product == "HBOC" else threshold * PFC_PER_HBOC_UNIT
        if store.level < threshold_val:
            env.process(production(env, product, store,
                                   HBOC_PROD_TIME if product == "HBOC" else PFC_PROD_TIME,
                                   HBOC_BATCH_SIZE if product == "HBOC" else PFC_BATCH_SIZE,
                                   PROD_FAILURE_RATE, pfc_cost))
        yield env.timeout(24)

def run_simulation(threshold_hboc, threshold_pfc, drone_split, arrival_rate=None, pfc_deg_rate=None, pfc_cost=None, delay_causes_file=None, multi_facility=False, initial_hboc=100, initial_pfc=200, part="part2"):
    """Run simulation with support for initial inventory settings and part-specific facility setup."""
    global metrics, current_status
    current_status = 0
    env = simpy.Environment()
    hboc_store = simpy.Container(env, capacity=10000, init=initial_hboc)
    pfc_store = simpy.Container(env, capacity=10000, init=initial_pfc)

    if multi_facility:
        hospitals = {
            "urban_hospital": simpy.Container(env, capacity=10000, init=0),
            "conflict_clinic": simpy.Container(env, capacity=10000, init=0)
        }
        hospitals["urban_hospital"].name = "urban_hospital"
        hospitals["conflict_clinic"].name = "conflict_clinic"
        drones = [simpy.Resource(env, capacity=1) for _ in range(NUM_DRONES)]
        facility_profiles = FACILITY_PROFILES
    else:
        if part == "part4":
            facility_profiles = CASE_STUDY_PROFILE
            hospitals = {
                "conflict_hospital": simpy.Container(env, capacity=10000, init=0)
            }
            hospitals["conflict_hospital"].name = "conflict_hospital"
        else:  # Part 2
            facility_profiles = {
                "single_hospital": {
                    "arrival_rate": arrival_rate,
                    "delay_factor": 1.0,
                    "failure_boost": 0.0,
                    "location": "urban"
                }
            }
            hospitals = {
                "single_hospital": simpy.Container(env, capacity=10000, init=0)
            }
            hospitals["single_hospital"].name = "single_hospital"
        drones = [simpy.Resource(env, capacity=1) for _ in range(NUM_DRONES_PART4)]

    metrics = {
        "stockouts": 0,
        "total_demand": 0,
        "delivery_delays": [],
        "delay_causes": [],
        "wasted_units": {"HBOC": 0, "PFC": 0},
        "mi_incidents": 0,
        "carpa_incidents": 0,
        "total_cost": 0,
        "state_transitions": []
    }

    transition_matrix = perturb_matrix(base_transition_matrix)
    location_profiles = perturb_profiles(base_location_profiles)

    env.process(system_status_update(env, transition_matrix))

    if multi_facility:
        for facility_name, profile in FACILITY_PROFILES.items():
            env.process(trauma_demand(env, facility_name, profile, hboc_store, pfc_store, hospitals, drones, drone_split, delay_causes_file))
    else:
        for facility_name, profile in facility_profiles.items():
            env.process(trauma_demand(env, facility_name, profile, hboc_store, pfc_store, hospitals, drones, drone_split, delay_causes_file, location_profiles))

    env.process(degradation(env, "HBOC", hboc_store, HBOC_AUTOXIDATION_RATE))
    env.process(degradation(env, "PFC", pfc_store, pfc_deg_rate))
    env.process(inventory_management(env, "HBOC", hboc_store, hospitals, drones, threshold_hboc, pfc_cost))
    env.process(inventory_management(env, "PFC", pfc_store, hospitals, drones, threshold_pfc, pfc_cost))

    env.run(until=SIM_TIME)

    stockout_prob = metrics["stockouts"] / metrics["total_demand"] if metrics["total_demand"] else 0
    avg_delay = np.mean(metrics["delivery_delays"]) if metrics["delivery_delays"] else 0
    metrics["total_cost"] += DRONE_COST * len(drones)

    return {
        "HBOC Threshold": threshold_hboc,
        "PFC Threshold": threshold_pfc,
        "Drone Split": drone_split,
        "Arrival Rate": arrival_rate if not multi_facility else "multi-facility",
        "PFC Deg Rate": pfc_deg_rate,
        "PFC Cost": pfc_cost,
        "Stockout Probability": stockout_prob,
        "Average Delivery Delay (hours)": avg_delay,
        "Wasted Units (HBOC)": metrics["wasted_units"]["HBOC"],
        "Wasted Units (PFC)": metrics["wasted_units"]["PFC"],
        "MI Incidents": metrics["mi_incidents"],
        "CARPA Incidents": metrics["carpa_incidents"],
        "Total Cost ($)": metrics["total_cost"]
    }

# Part 2: Run Reduced Sensitivity Analysis (Validation)
print("Running Part 2: Reduced Sensitivity Analysis for Validation")
results_file_part2 = "results/simulation_results_reduced.csv"
delay_causes_file_part2 = "results/delay_causes_reduced.csv"

with open(delay_causes_file_part2, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["state", "location", "delay", "facility"])
    writer.writeheader()

results_part2 = []
param_combinations_part2 = list(product(THRESHOLDS_HBOC, THRESHOLDS_PFC, DRONE_SPLITS, ARRIVAL_RATES_PART2, PFC_DEG_RATES, PFC_COSTS))
total_runs_part2 = len(param_combinations_part2) * ITERATIONS

completed_runs = 0
start_time = time.time()

try:
    for i, params in enumerate(param_combinations_part2, 1):
        for j in range(ITERATIONS):
            result = run_simulation(*params, delay_causes_file_part2, multi_facility=False, part="part2")
            results_part2.append(result)
            completed_runs += 1

            if completed_runs % 10 == 0:
                pd.DataFrame(results_part2).to_csv(results_file_part2, index=False)
                print(f"Part 2 Checkpoint saved: {completed_runs}/{total_runs_part2} runs completed")

            elapsed = time.time() - start_time
            print(f"Part 2 Run {completed_runs}/{total_runs_part2} | Elapsed: {elapsed:.2f}s | Params: {params}")
except KeyboardInterrupt:
    print("Part 2 Simulation interrupted. Saving partial results...")
    pd.DataFrame(results_part2).to_csv(results_file_part2, index=False)
    print(f"Saved {completed_runs} runs to {results_file_part2}")

df_part2 = pd.DataFrame(results_part2)
summary_part2 = df_part2.groupby(["HBOC Threshold", "PFC Threshold", "Drone Split", "Arrival Rate", "PFC Deg Rate"]).agg({
    "Stockout Probability": ["mean", "std"],
    "Average Delivery Delay (hours)": ["mean", "std"],
    "Wasted Units (PFC)": ["mean", "std"],
    "Total Cost ($)": ["mean", "std"],
    "MI Incidents": ["mean"],
    "CARPA Incidents": ["mean"]
}).reset_index()

summary_part2.to_csv("results/simulation_summary_reduced.csv")
print("Part 2 Summary Statistics:")
print(summary_part2)

plt.figure(figsize=(12, 6))
for drone_split in DRONE_SPLITS:
    subset = df_part2[df_part2["Drone Split"] == drone_split]
    plt.scatter(subset["Stockout Probability"], subset["Average Delivery Delay (hours)"], label=f"Drone Split {drone_split*100}%", alpha=0.5)
plt.xlabel("Stockout Probability")
plt.ylabel("Average Delivery Delay (hours)")
plt.title("Part 2: Stockout Probability vs. Delivery Delay by Drone Split")
plt.legend()
plt.grid(True)
plt.savefig("figures/stockout_vs_delay_reduced.png")
plt.close()

delay_causes_part2 = pd.read_csv(delay_causes_file_part2)
print("Part 2 Delay Causes Sample:")
print(delay_causes_part2.head())

if 'state' in delay_causes_part2.columns and not delay_causes_part2.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part2.boxplot(column="delay", by="state")
    plt.title("Part 2: Delivery Delays by Markov State")
    plt.xlabel("System State")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_state_reduced.png")
    plt.close()

if 'location' in delay_causes_part2.columns and not delay_causes_part2.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part2.boxplot(column="delay", by="location")
    plt.title("Part 2: Delivery Delays by Location")
    plt.xlabel("Location")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_location_reduced.pn")
    plt.close()

# Part 3: Run Multi-Facility Simulation with Real-World Data (including Disaster Mode)
print("Running Part 3: Multi-Facility Simulation with Real-World Data (including Disaster Mode)")
results_file_part3 = "results/simulation_results_part3.csv"
delay_causes_file_part3 = "results/delay_causes_part3.csv"

with open(delay_causes_file_part3, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["state", "location", "delay", "facility"])
    writer.writeheader()

results_part3 = []
param_combinations_part3 = list(product(THRESHOLDS_HBOC, THRESHOLDS_PFC, DRONE_SPLITS, PFC_DEG_RATES, PFC_COSTS))
total_runs_part3 = len(param_combinations_part3) * ITERATIONS

completed_runs = 0
start_time = time.time()

try:
    for i, params in enumerate(param_combinations_part3, 1):
        for j in range(ITERATIONS):
            result = run_simulation(*params, delay_causes_file=delay_causes_file_part3, multi_facility=True, part="part3")
            results_part3.append(result)
            completed_runs += 1

            if completed_runs % 10 == 0:
                pd.DataFrame(results_part3).to_csv(results_file_part3, index=False)
                print(f"Part 3 Checkpoint saved: {completed_runs}/{total_runs_part3} runs completed")

            elapsed = time.time() - start_time
            print(f"Part 3 Run {completed_runs}/{total_runs_part3} | Elapsed: {elapsed:.2f}s | Params: {params}")
except KeyboardInterrupt:
    print("Part 3 Simulation interrupted. Saving partial results...")
    pd.DataFrame(results_part3).to_csv(results_file_part3, index=False)
    print(f"Saved {completed_runs} runs to {results_file_part3}")

df_part3 = pd.DataFrame(results_part3)
summary_part3 = df_part3.groupby(["HBOC Threshold", "PFC Threshold", "Drone Split", "Arrival Rate", "PFC Deg Rate"]).agg({
    "Stockout Probability": ["mean", "std"],
    "Average Delivery Delay (hours)": ["mean", "std"],
    "Wasted Units (PFC)": ["mean", "std"],
    "Total Cost ($)": ["mean", "std"],
    "MI Incidents": ["mean"],
    "CARPA Incidents": ["mean"]
}).reset_index()

summary_part3.to_csv("results/simulation_summary_part3.csv")
print("Part 3 Summary Statistics:")
print(summary_part3)

plt.figure(figsize=(12, 6))
for drone_split in DRONE_SPLITS:
    subset = df_part3[df_part3["Drone Split"] == drone_split]
    plt.scatter(subset["Stockout Probability"], subset["Average Delivery Delay (hours)"], label=f"Drone Split {drone_split*100}%", alpha=0.5)
plt.xlabel("Stockout Probability")
plt.ylabel("Average Delivery Delay (hours)")
plt.title("Part 3: Stockout Probability vs. Delivery Delay by Drone Split (Multi-Facility)")
plt.legend()
plt.grid(True)
plt.savefig("figures/stockout_vs_delay_part3.png")
plt.close()

delay_causes_part3 = pd.read_csv(delay_causes_file_part3)
print("Part 3 Delay Causes Sample:")
print(delay_causes_part3.head())

if 'state' in delay_causes_part3.columns and not delay_causes_part3.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part3.boxplot(column="delay", by="state")
    plt.title("Part 3: Delivery Delays by Markov State (Multi-Facility)")
    plt.xlabel("System State")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_state_part3.png")
    plt.close()

if 'location' in delay_causes_part3.columns and not delay_causes_part3.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part3.boxplot(column="delay", by="location")
    plt.title("Part 3: Delivery Delays by Location (Multi-Facility)")
    plt.xlabel("Location")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_location_part3.png")
    plt.close()

if 'facility' in delay_causes_part3.columns and not delay_causes_part3.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part3.boxplot(column="delay", by="facility")
    plt.title("Part 3: Delivery Delays by Facility")
    plt.xlabel("Facility")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_facility_part3.png")
    plt.close()

# Part 4: Case Study Simulation for Policy Implementation and Validation
print("Running Part 4: Case Study Simulation for Policy Implementation (Conflict Zone Hospital)")
results_file_part4 = "results/simulation_results_part4.csv"
delay_causes_file_part4 = "results/delay_causes_part4.csv"

with open(delay_causes_file_part4, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["state", "location", "delay", "facility"])
    writer.writeheader()

results_part4 = []
param_combinations_part4 = list(product(THRESHOLDS_HBOC, THRESHOLDS_PFC, DRONE_SPLITS_PART4, PFC_DEG_RATES, PFC_COSTS))
total_runs_part4 = len(param_combinations_part4) * ITERATIONS

completed_runs = 0
start_time = time.time()

try:
    for i, params in enumerate(param_combinations_part4, 1):
        for j in range(ITERATIONS):
            result = run_simulation(*params, delay_causes_file=delay_causes_file_part4, multi_facility=False, initial_hboc=INITIAL_HBOC_PART4, initial_pfc=INITIAL_PFC_PART4, part="part4")
            results_part4.append(result)
            completed_runs += 1

            if completed_runs % 10 == 0:
                pd.DataFrame(results_part4).to_csv(results_file_part4, index=False)
                print(f"Part 4 Checkpoint saved: {completed_runs}/{total_runs_part4} runs completed")

            elapsed = time.time() - start_time
            print(f"Part 4 Run {completed_runs}/{total_runs_part4} | Elapsed: {elapsed:.2f}s | Params: {params}")
except KeyboardInterrupt:
    print("Part 4 Simulation interrupted. Saving partial results...")
    pd.DataFrame(results_part4).to_csv(results_file_part4, index=False)
    print(f"Saved {completed_runs} runs to {results_file_part4}")

df_part4 = pd.DataFrame(results_part4)
summary_part4 = df_part4.groupby(["HBOC Threshold", "PFC Threshold", "Drone Split", "Arrival Rate", "PFC Deg Rate"]).agg({
    "Stockout Probability": ["mean", "std"],
    "Average Delivery Delay (hours)": ["mean", "std"],
    "Wasted Units (PFC)": ["mean", "std"],
    "Total Cost ($)": ["mean", "std"],
    "MI Incidents": ["mean"],
    "CARPA Incidents": ["mean"]
}).reset_index()

summary_part4.to_csv("results/simulation_summary_part4.csv")
print("Part 4 Summary Statistics:")
print(summary_part4)

plt.figure(figsize=(12, 6))
for drone_split in DRONE_SPLITS_PART4:
    subset = df_part4[df_part4["Drone Split"] == drone_split]
    plt.scatter(subset["Stockout Probability"], subset["Average Delivery Delay (hours)"], label=f"Drone Split {drone_split*100}%", alpha=0.5)
plt.xlabel("Stockout Probability")
plt.ylabel("Average Delivery Delay (hours)")
plt.title("Part 4: Stockout Probability vs. Delivery Delay (Conflict Zone Hospital)")
plt.legend()
plt.grid(True)
plt.savefig("figures/stockout_vs_delay_part4.pn")
plt.close()

delay_causes_part4 = pd.read_csv(delay_causes_file_part4)
print("Part 4 Delay Causes Sample:")
print(delay_causes_part4.head())

if 'state' in delay_causes_part4.columns and not delay_causes_part4.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part4.boxplot(column="delay", by="state")
    plt.title("Part 4: Delivery Delays by Markov State (Conflict Zone Hospital)")
    plt.xlabel("System State")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_state_part4.pn")
    plt.close()

if 'location' in delay_causes_part4.columns and not delay_causes_part4.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part4.boxplot(column="delay", by="location")
    plt.title("Part 4: Delivery Delays by Location (Conflict Zone Hospital)")
    plt.xlabel("Location")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_location_part4.png")
    plt.close()

if 'facility' in delay_causes_part4.columns and not delay_causes_part4.empty:
    plt.figure(figsize=(12, 6))
    delay_causes_part4.boxplot(column="delay", by="facility")
    plt.title("Part 4: Delivery Delays by Facility (Conflict Zone Hospital)")
    plt.xlabel("Facility")
    plt.ylabel("Delay (hours)")
    plt.savefig("figures/delays_by_facility_part4.png")
    plt.close()"results/stockout_vs
