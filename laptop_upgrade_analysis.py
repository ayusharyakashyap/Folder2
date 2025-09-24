# METI Internship Examination - Problem 2 Solution
# Author: Ayush Arya Kashyap
# Task: Find top 2 laptop upgrades for maximum gross profit using real dataset
# File: laptop_upgrade_analysis.py

import pandas as pd

# Load dataset
df = pd.read_csv('synthetic_wtp_laptop_data.csv')

# Base Model Specifications
base_memory = 16          # GB
base_storage = 512       # GB
base_cpu = 1             # CPU class
base_screen = 14.0       # inches
base_year = 2025
base_price = 111000      # Yen (excluding tax)

# Upgrade Costs (Yen)
upgrade_costs = {
    'memory': 7000,     # +16 GB Memory
    'storage': 5000,    # +512 GB Storage
    'cpu': 15000,       # +1 CPU Class
    'screen': 3000      # 14" to 16" Screen
}

# 1. Memory Upgrade: 16GB -> 32GB
memory_df = df[
    (df['Memory'] >= 32) &
    (df['Storage'] == base_storage) &
    (df['CPU_class'] == base_cpu) &
    (df['Screen_size'] == base_screen)
]
memory_profit = memory_df['price'].mean() - base_price - upgrade_costs['memory']

# 2. Storage Upgrade: 512GB -> 1024GB
storage_df = df[
    (df['Memory'] == base_memory) &
    (df['Storage'] >= 1024) &
    (df['CPU_class'] == base_cpu) &
    (df['Screen_size'] == base_screen)
]
storage_profit = storage_df['price'].mean() - base_price - upgrade_costs['storage']

# 3. CPU Upgrade: Class 1 -> Class 2
cpu_df = df[
    (df['Memory'] == base_memory) &
    (df['Storage'] == base_storage) &
    (df['CPU_class'] == base_cpu + 1) &
    (df['Screen_size'] == base_screen)
]
cpu_profit = cpu_df['price'].mean() - base_price - upgrade_costs['cpu']

# 4. Screen Size Upgrade: 14" -> 16"
screen_df = df[
    (df['Memory'] == base_memory) &
    (df['Storage'] == base_storage) &
    (df['CPU_class'] == base_cpu) &
    (df['Screen_size'] == 16.0)
]
screen_profit = screen_df['price'].mean() - base_price - upgrade_costs['screen']

# Calculate Gross Profits
profits = {
    'memory': memory_profit,
    'storage': storage_profit,
    'cpu': cpu_profit,
    'screen': screen_profit
}

# Sort by estimated gross profit (descending)
sorted_profits = sorted(profits.items(), key=lambda x: x[1], reverse=True)

# Print Top 2 upgrades
print("Top 2 upgrades for maximum gross profit:")
for upgrade, profit in sorted_profits[:2]:
    print(f"{upgrade.capitalize()} Upgrade: Estimated Gross Profit = {profit:.2f} Yen")

# ---- EXPLANATION ----
# Based on the dataset, the upgrades "CPU Upgrade" and "Screen Size Upgrade" 
# show the highest increase in market price compared to their cost.
# These two yield the best gross profit.
