#!/usr/bin/env python3
"""
Create advanced visualizations for bridge analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import ast
from collections import defaultdict, Counter

# Define ordering for consistent display across all charts
HUMAN_SKILL_ORDER = ["Beginner", "Novice", "Intermediate", "Advanced", "Expert"]
BOT_ORDER = ["ben_beginner", "ben_novice", "ben_intermediate", "ben_advanced", "gib_basic", "gib_advanced", "argine"]

def sort_by_order(data, order_list, key_column):
    """Sort DataFrame by a predefined order list"""
    # Create a categorical type with the specified order
    data[key_column] = pd.Categorical(data[key_column], categories=order_list, ordered=True)
    return data.sort_values(key_column)

def clean_bot_name(bot_name: str) -> str:
    """Clean bot names to keep only the relevant substring."""
    if not isinstance(bot_name, str):
        return str(bot_name)
    
    # Define the bot name patterns we want to keep
    target_names = ["ben_advanced", "ben_beginner", "ben_intermediate", "ben_novice", 
                   "gib_advanced", "gib_basic", "argine"]
    
    # Check if any target name is in the bot name
    for target in target_names:
        if target in bot_name:
            return target
    
    # If no match found, return the original name
    return bot_name

# Setup
FIG_DIR = "bridge_output_run/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
skill_data = pd.read_parquet("bridge_output_run/skill_bucket_profiles.parquet")
bot_data = pd.read_parquet("bridge_output_run/bot_skill_profiles.parquet")
risk_data = pd.read_parquet("bridge_output_run/risk_analysis.parquet")
contract_dist_data = pd.read_parquet("bridge_output_run/contract_level_distribution.parquet")

# Load additional analysis data
thin_contract_data = pd.read_parquet("bridge_output_run/thin_contract_rate.parquet")
strategic_rates_data = pd.read_parquet("bridge_output_run/strategic_rates.parquet")
performance_contract_strain_data = pd.read_parquet("bridge_output_run/performance_by_contract_and_strain.parquet")
play_sharpness_data = pd.read_parquet("bridge_output_run/play_phase_sharpness.parquet")
bot_opening_lead_data = pd.read_parquet("bridge_output_run/bot_opening_lead_match.parquet")

# Load shadow analysis data for different skill levels
shadow_files = [
    "shadow_analysis_expert_corrected.parquet",
    "shadow_analysis_advanced_corrected.parquet", 
    "shadow_analysis_beginner_corrected.parquet",
    "shadow_analysis_intermediate_corrected.parquet",
    "shadow_analysis_novice_corrected.parquet"
]

shadow_datasets = {}
for file in shadow_files:
    try:
        data = pd.read_parquet(file)
        # Extract skill level from filename
        if "advanced" in file:
            skill_level = "Advanced"
        elif "beginner" in file:
            skill_level = "Beginner"
        elif "intermediate" in file:
            skill_level = "Intermediate"
        elif "novice" in file:
            skill_level = "Novice"
        else:
            skill_level = "Expert"
        
        shadow_datasets[skill_level] = data
        print(f"Loaded {file}: {len(data)} games")
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

shadow_data_available = len(shadow_datasets) > 0

# Clean bot names in all datasets
if "bot_name" in bot_data.columns:
    bot_data["bot_name"] = bot_data["bot_name"].apply(clean_bot_name)

if "player_name" in risk_data.columns:
    # Create a mask for bot rows to avoid complex lambda
    bot_mask = risk_data["is_bot"] == True
    risk_data.loc[bot_mask, "player_name"] = risk_data.loc[bot_mask, "player_name"].apply(clean_bot_name)

if "player_name" in contract_dist_data.columns:
    # Create a mask for bot rows to avoid complex lambda  
    bot_mask = contract_dist_data["player_type"] == "Bot"
    contract_dist_data.loc[bot_mask, "player_name"] = contract_dist_data.loc[bot_mask, "player_name"].apply(clean_bot_name)

# Clean bot names in additional datasets
for dataset in [thin_contract_data, strategic_rates_data, performance_contract_strain_data, play_sharpness_data]:
    if "player_name" in dataset.columns and "is_bot" in dataset.columns:
        bot_mask = dataset["is_bot"] == True
        dataset.loc[bot_mask, "player_name"] = dataset.loc[bot_mask, "player_name"].apply(clean_bot_name)

# Clean bot names in opening lead data (different format - no is_bot column)
if "player_name" in bot_opening_lead_data.columns:
    bot_opening_lead_data["player_name"] = bot_opening_lead_data["player_name"].apply(clean_bot_name)

# Sort data by predefined orders
skill_data = sort_by_order(skill_data, HUMAN_SKILL_ORDER, "skill_bucket")
bot_data = sort_by_order(bot_data, BOT_ORDER, "bot_name")

print("Creating advanced visualizations...")

# ============================================================================
# 1) SKILL BUCKET PROFILE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("1. SKILL BUCKET PROFILE COMPARISON")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

metrics_to_compare = [
    ("avg_mp_pct", "Average MP%"),
    ("avg_raw_score", "Average Raw Score"),
    ("game_rate", "Game Contract Rate"),
    ("double_rate", "Doubling Rate"),
    ("thin_contract_rate", "Thin Contract Rate"),
    ("avg_overtricks", "Average Overtricks")
]

print("\nHUMAN SKILL BUCKET PROFILES:")
print("-" * 40)
for skill_bucket in HUMAN_SKILL_ORDER:
    if skill_bucket in skill_data["skill_bucket"].values:
        bucket_data = skill_data[skill_data["skill_bucket"] == skill_bucket].iloc[0]
        print(f"{skill_bucket:>12}: MP%={bucket_data['avg_mp_pct']:.1f}%, Raw Score={bucket_data['avg_raw_score']:.1f}, "
              f"Game Rate={bucket_data['game_rate']:.1f}%, Double Rate={bucket_data['double_rate']:.1f}%")

print("\nBOT PERFORMANCE PROFILES:")
print("-" * 40)
bot_data_clean = bot_data.copy()
bot_data_clean['clean_name'] = bot_data_clean['bot_name'].apply(clean_bot_name)
for _, bot in bot_data_clean.iterrows():
    print(f"{bot['clean_name']:>15}: MP%={bot['avg_mp_pct']:.1f}%, Raw Score={bot['avg_raw_score']:.1f}, "
          f"Game Rate={bot['game_rate']:.1f}%, Double Rate={bot['double_rate']:.1f}%, "
          f"Closest Skill: {bot['closest_skill_bucket']}")

for idx, (metric, title) in enumerate(metrics_to_compare):
    if idx >= len(axes):
        break
    
    # Plot skill buckets and bots together as bars
    if metric in skill_data.columns:
        skill_buckets = skill_data["skill_bucket"].tolist()
        skill_values = skill_data[metric].tolist()
        
        # Combine human and bot data
        all_labels = skill_buckets.copy()
        all_values = skill_values.copy()
        
        # Define distinct colors
        human_colors = ['steelblue', 'green', 'purple', 'orange', 'brown']
        bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
        
        # Create colors for humans
        colors = [human_colors[i % len(human_colors)] for i in range(len(skill_buckets))]
        
        # Add bot values as additional bars
        if metric in bot_data.columns:
            bot_values = bot_data[metric].tolist()
            bot_names = bot_data["bot_name"].tolist()
            
            all_labels.extend(bot_names)
            all_values.extend(bot_values)
            
            # Add bot colors
            bot_bar_colors = [bot_colors[i % len(bot_colors)] for i in range(len(bot_names))]
            colors.extend(bot_bar_colors)
        
        bars = axes[idx].bar(range(len(all_labels)), all_values, 
                            color=colors, alpha=0.7)
        
        axes[idx].set_title(title)
        axes[idx].set_xticks(range(len(all_labels)))
        axes[idx].set_xticklabels(all_labels, rotation=45)
        
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "skill_bucket_comparison.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Skill bucket comparison chart saved to {FIG_DIR}/skill_bucket_comparison.png")
print("   - Shows 6 key performance metrics comparing human skill buckets vs bots")
print("   - Human skill buckets in blue/green tones, bots in red/gold tones")

# ============================================================================
# 2) BOT SKILL MATCHING VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("2. BOT SKILL MATCHING ANALYSIS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Bot skill bucket matching
bot_buckets = bot_data["closest_skill_bucket"].value_counts()
print("\nBOT SKILL BUCKET DISTRIBUTION:")
print("-" * 35)
for bucket, count in bot_buckets.items():
    percentage = (count / len(bot_data)) * 100
    print(f"{bucket:>12}: {count} bots ({percentage:.1f}%)")

axes[0].pie(bot_buckets.values, labels=bot_buckets.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title("Bot Distribution Across Human Skill Buckets")

# Bot vs Human performance comparison - simplified
skill_bucket_order = ["Beginner", "Novice", "Intermediate", "Advanced", "Expert"]
human_values = []
bot_values = []
labels = []

print("\nHUMAN vs BOT MP% COMPARISON BY SKILL LEVEL:")
print("-" * 50)

for bucket in skill_bucket_order:
    if bucket in skill_data["skill_bucket"].values:
        human_val = skill_data[skill_data["skill_bucket"] == bucket]["avg_mp_pct"].iloc[0]
        bots_in_bucket = bot_data[bot_data["closest_skill_bucket"] == bucket]
        bot_val = bots_in_bucket["avg_mp_pct"].mean() if len(bots_in_bucket) > 0 else 0
        
        human_values.append(human_val)
        bot_values.append(bot_val if not np.isnan(bot_val) else 0)
        labels.append(bucket)
        
        bot_count = len(bots_in_bucket)
        diff = bot_val - human_val if not np.isnan(bot_val) else 0
        print(f"{bucket:>12}: Human={human_val:.1f}%, Bot Avg={bot_val:.1f}% ({bot_count} bots), "
              f"Diff={diff:+.1f}%")

x_pos = np.arange(len(labels))
width = 0.35

axes[1].bar(x_pos - width/2, human_values, width, label="Human Average", alpha=0.7)
axes[1].bar(x_pos + width/2, bot_values, width, label="Bot Performance", alpha=0.7)

axes[1].set_xlabel("Skill Bucket")
axes[1].set_ylabel("Average MP%")
axes[1].set_title("Bot vs Human MP% Performance by Skill Level")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels, rotation=45)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "bot_skill_matching.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Bot skill matching chart saved to {FIG_DIR}/bot_skill_matching.png")
print("   - Left: Pie chart showing which human skill buckets bots are assigned to")
print("   - Right: Bar chart comparing human vs bot MP% performance by skill level")

# ============================================================================
# 3) RISK ANALYSIS VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("3. RISK ANALYSIS - DOUBLING BEHAVIOR")
print("="*80)

if len(risk_data) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Combined Risk aversion chart - humans and bots together
    human_risk = risk_data[risk_data["is_bot"] == False]
    bot_risk = risk_data[risk_data["is_bot"] == True]
    
    # Filter out "unknown" skill bucket from humans
    if len(human_risk) > 0:
        human_risk = human_risk[human_risk["skill_bucket"] != "Unknown"]
    
    print(f"\nRISK ANALYSIS DATA SUMMARY:")
    print(f"- Total risk decisions analyzed: {len(risk_data)}")
    print(f"- Human decisions: {len(human_risk)}")
    print(f"- Bot decisions: {len(bot_risk)}")
    
    # Prepare combined data for risk aversion with ordered display
    combined_labels = []
    combined_values = []
    combined_colors = []
    
    # Define color schemes
    human_colors = ['steelblue', 'green', 'purple', 'orange', 'brown']
    bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
    
    print("\nRISK AVERSION SCORES (Higher = More Risk Averse):")
    print("-" * 55)
    
    # Add human data by skill bucket in specified order
    if len(human_risk) > 0:
        risk_by_skill = human_risk.groupby("skill_bucket")["risk_aversion_score"].mean()
        for i, bucket in enumerate(HUMAN_SKILL_ORDER):
            if bucket in risk_by_skill.index:
                combined_labels.append(f"Human\n{bucket}")
                combined_values.append(risk_by_skill[bucket])
                combined_colors.append(human_colors[i % len(human_colors)])
                print(f"Human {bucket:>12}: {risk_by_skill[bucket]:.3f}")
    
    # Add bot data in specified order
    if len(bot_risk) > 0:
        bot_risk_summary = bot_risk.groupby("player_name")["risk_aversion_score"].mean()
        for i, bot_name in enumerate(BOT_ORDER):
            clean_name = clean_bot_name(bot_name)
            if bot_name in bot_risk_summary.index:
                combined_labels.append(f"Bot\n{clean_name}")
                combined_values.append(bot_risk_summary[bot_name])
                combined_colors.append(bot_colors[i % len(bot_colors)])
                print(f"Bot {clean_name:>15}: {bot_risk_summary[bot_name]:.3f}")
    
    if combined_values:
        bars = axes[0].bar(range(len(combined_labels)), combined_values, 
                          color=combined_colors, alpha=0.7)
        axes[0].set_title("Risk Aversion Score - Humans vs Bots")
        axes[0].set_ylabel("Risk Aversion Score")
        axes[0].set_xticks(range(len(combined_labels)))
        axes[0].set_xticklabels(combined_labels, rotation=45)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No risk data available\n(May indicate no missed\ndoubling opportunities)", 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Risk Aversion Score - Humans vs Bots")
    
    # Combined Doubling rate when opponents fail
    combined_labels_double = []
    combined_values_double = []
    combined_colors_double = []
    
    print("\nDOUBLING OPPORTUNITY RATES (When opponents fail):")
    print("-" * 52)
    
    # Add human data by skill bucket in specified order
    if len(human_risk) > 0:
        double_by_skill = human_risk.groupby("skill_bucket")["doubling_opportunity_rate"].mean()
        for i, bucket in enumerate(HUMAN_SKILL_ORDER):
            if bucket in double_by_skill.index:
                combined_labels_double.append(f"Human\n{bucket}")
                combined_values_double.append(double_by_skill[bucket])
                combined_colors_double.append(human_colors[i % len(human_colors)])
                print(f"Human {bucket:>12}: {double_by_skill[bucket]:.1%}")
    
    # Add bot data in specified order
    if len(bot_risk) > 0:
        bot_double_summary = bot_risk.groupby("player_name")["doubling_opportunity_rate"].mean()
        for i, bot_name in enumerate(BOT_ORDER):
            clean_name = clean_bot_name(bot_name)
            if bot_name in bot_double_summary.index:
                combined_labels_double.append(f"Bot\n{clean_name}")
                combined_values_double.append(bot_double_summary[bot_name])
                combined_colors_double.append(bot_colors[i % len(bot_colors)])
                print(f"Bot {clean_name:>15}: {bot_double_summary[bot_name]:.1%}")
    
    if combined_values_double:
        bars = axes[1].bar(range(len(combined_labels_double)), combined_values_double, 
                          color=combined_colors_double, alpha=0.7)
        axes[1].set_title("Doubling Rate When Opponents Fail - Humans vs Bots")
        axes[1].set_ylabel("Doubling Rate")
        axes[1].set_xticks(range(len(combined_labels_double)))
        axes[1].set_xticklabels(combined_labels_double, rotation=45)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No doubling data\navailable", 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Doubling Rate When Opponents Fail - Humans vs Bots")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "risk_analysis.png"), bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\n✅ Risk analysis chart saved to {FIG_DIR}/risk_analysis.png")
    print("   - Left: Risk aversion scores (higher = more conservative)")
    print("   - Right: Rate of doubling when opponents fail (higher = more aggressive)")

else:
    print("\n⚠️  No risk analysis data available - skipping risk visualization")

# ============================================================================
# 4) CONTRACT SUCCESS RATE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("4. CONTRACT SUCCESS RATE COMPARISON")
print("="*80)

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Human success rates by skill
human_data = contract_dist_data[contract_dist_data["player_type"] == "Human"]
bot_data_dist = contract_dist_data[contract_dist_data["player_type"] == "Bot"]

success_metrics = ["part_score_success", "game_success", "small_slam_success"]
success_labels = ["Part Score", "Game", "Small Slam"]

# Combine all data
all_labels = []
all_data = []
all_colors = []

# Define color schemes
human_colors = ['steelblue', 'green', 'purple', 'orange', 'brown']
bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']

# Add human skill buckets in specified order
if len(human_data) > 0:
    for i, bucket in enumerate(HUMAN_SKILL_ORDER):
        bucket_data = human_data[human_data["skill_bucket"] == bucket]
        if len(bucket_data) > 0:
            values = [bucket_data[metric].iloc[0] for metric in success_metrics]
            all_labels.append(f"Human\n{bucket}")
            all_data.append(values)
            all_colors.append(human_colors[i % len(human_colors)])

# Add bots in specified order
if len(bot_data_dist) > 0:
    for i, bot_name in enumerate(BOT_ORDER):
        bot_data_subset = bot_data_dist[bot_data_dist["player_name"] == bot_name]
        if len(bot_data_subset) > 0:
            values = [bot_data_subset[metric].iloc[0] for metric in success_metrics]
            all_labels.append(f"Bot\n{bot_name}")
            all_data.append(values)
            all_colors.append(bot_colors[i % len(bot_colors)])

# Create grouped bar chart
if all_data:
    x = np.arange(len(success_labels))
    width = 0.8 / len(all_data)  # Adjust width based on number of groups
    
    for i, (label, values, color) in enumerate(zip(all_labels, all_data, all_colors)):
        ax.bar(x + i*width - (len(all_data)*width)/2 + width/2, values, 
               width, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Success Rate")
    ax.set_title("Contract Success Rate - Humans vs Bots")
    ax.set_xticks(x)
    ax.set_xticklabels(success_labels)
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "contract_success_analysis.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Contract success analysis saved to {FIG_DIR}/contract_success_analysis.png")
print("   - Shows success rates for Part Score, Game, and Small Slam contracts")
print("   - Compares human skill buckets vs bot performance")

# Print detailed success rate summary
print(f"\nCONTRACT SUCCESS RATES SUMMARY:")
print("-" * 45)
if len(human_data) > 0:
    print("HUMAN SUCCESS RATES BY SKILL LEVEL:")
    print("-" * 37)
    for bucket in HUMAN_SKILL_ORDER:
        bucket_data = human_data[human_data["skill_bucket"] == bucket]
        if len(bucket_data) > 0:
            row = bucket_data.iloc[0]
            print(f"  {bucket:>12}: Part Score={row['part_score_success']:.1%}, "
                  f"Game={row['game_success']:.1%}, Small Slam={row['small_slam_success']:.1%}")

if len(bot_data_dist) > 0:
    print("\nBOT SUCCESS RATES:")
    print("-" * 20)
    for bot_name in BOT_ORDER:
        bot_subset = bot_data_dist[bot_data_dist["player_name"] == bot_name]
        if len(bot_subset) > 0:
            row = bot_subset.iloc[0]
            print(f"  {bot_name:>15}: Part Score={row['part_score_success']:.1%}, "
                  f"Game={row['game_success']:.1%}, Small Slam={row['small_slam_success']:.1%}")

# 5) Contract Pick Rate Analysis - Combined Chart
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Contract pick rate metrics - focusing on the most interesting levels
pick_metrics = ["level_1_rate", "level_3_rate", "level_6_rate"]  # Part score (1), Game (3), Small slam (6)
pick_labels = ["Level 1 (Part Score)", "Level 3 (Game)", "Level 6 (Small Slam)"]

# Combine all data with proper ordering
all_labels = []
all_data = []
all_colors = []

# Define distinct colors (same as other charts)
human_colors = ['steelblue', 'green', 'purple', 'orange', 'brown']
bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']

# Order human skill buckets
human_skill_order = ["Beginner", "Novice", "Intermediate", "Advanced", "Expert"]

# Add human skill buckets in order
if len(human_data) > 0:
    for i, bucket in enumerate(human_skill_order):
        if bucket in human_data["skill_bucket"].values:
            bucket_row = human_data[human_data["skill_bucket"] == bucket]
            values = [bucket_row[metric].iloc[0] for metric in pick_metrics]
            all_labels.append(f"Human\n{bucket}")
            all_data.append(values)
            all_colors.append(human_colors[i % len(human_colors)])

# Order bots
bot_order = ["ben_beginner", "ben_novice", "ben_intermediate", "ben_advanced", 
             "gib_basic", "gib_advanced", "argine"]

# Add bots in order
if len(bot_data_dist) > 0:
    for i, bot_name in enumerate(bot_order):
        if bot_name in bot_data_dist["player_name"].values:
            bot_row = bot_data_dist[bot_data_dist["player_name"] == bot_name]
            values = [bot_row[metric].iloc[0] for metric in pick_metrics]
            all_labels.append(f"Bot\n{bot_name}")
            all_data.append(values)
            all_colors.append(bot_colors[i % len(bot_colors)])

# Create grouped bar chart
if all_data:
    x = np.arange(len(pick_labels))
    width = 0.8 / len(all_data)  # Adjust width based on number of groups
    
    for i, (label, values, color) in enumerate(zip(all_labels, all_data, all_colors)):
        ax.bar(x + i*width - (len(all_data)*width)/2 + width/2, values, 
               width, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel("Contract Level")
    ax.set_ylabel("Pick Rate")
    ax.set_title("Contract Pick Rate - Humans vs Bots")
    ax.set_xticks(x)
    ax.set_xticklabels(pick_labels)
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "contract_pick_rate.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Contract pick rate analysis saved to {FIG_DIR}/contract_pick_rate.png")
print("   - Shows bidding preferences for different contract levels")
print("   - Level 1 = Part Score, Level 3 = Game, Level 6 = Small Slam")

# Print detailed pick rate summary
print(f"\nCONTRACT PICK RATES SUMMARY:")
print("-" * 35)
if len(human_data) > 0:
    print("HUMAN PICK RATES BY SKILL LEVEL:")
    print("-" * 34)
    for bucket in HUMAN_SKILL_ORDER:
        bucket_data = human_data[human_data["skill_bucket"] == bucket]
        if len(bucket_data) > 0:
            row = bucket_data.iloc[0]
            print(f"  {bucket:>12}: Level 1={row['level_1_rate']:.1%}, "
                  f"Level 3={row['level_3_rate']:.1%}, Level 6={row['level_6_rate']:.1%}")

if len(bot_data_dist) > 0:
    print("\nBOT PICK RATES:")
    print("-" * 17)
    for bot_name in BOT_ORDER:
        bot_subset = bot_data_dist[bot_data_dist["player_name"] == bot_name]
        if len(bot_subset) > 0:
            row = bot_subset.iloc[0]
            print(f"  {bot_name:>15}: Level 1={row['level_1_rate']:.1%}, "
                  f"Level 3={row['level_3_rate']:.1%}, Level 6={row['level_6_rate']:.1%}")

# 6) Contract Pick Rate Heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Prepare data for heatmaps with proper ordering
pick_rate_metrics = ["level_1_rate", "level_2_rate", "level_3_rate", "level_4_rate", "level_5_rate", "level_6_rate", "level_7_rate"]
success_rate_metrics = ["part_score_success", "game_success", "small_slam_success", "grand_slam_success"]

pick_rate_labels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
success_rate_labels = ["Part Score", "Game", "Small Slam", "Grand Slam"]

# Combine and order all players/entities
all_entities = []
all_entity_labels = []

# Add humans in order
human_skill_order = ["Beginner", "Novice", "Intermediate", "Advanced", "Expert"]
for bucket in human_skill_order:
    if len(human_data[human_data["skill_bucket"] == bucket]) > 0:
        all_entities.append(("human", bucket))
        all_entity_labels.append(f"Human {bucket}")

# Add bots in order
bot_order = ["ben_beginner", "ben_novice", "ben_intermediate", "ben_advanced", 
             "gib_basic", "gib_advanced", "argine"]
for bot_name in bot_order:
    if len(bot_data_dist[bot_data_dist["player_name"] == bot_name]) > 0:
        all_entities.append(("bot", bot_name))
        all_entity_labels.append(f"Bot {bot_name}")

# Create pick rate matrix
pick_rate_matrix = []
for entity_type, entity_name in all_entities:
    if entity_type == "human":
        row_data = human_data[human_data["skill_bucket"] == entity_name]
    else:  # bot
        row_data = bot_data_dist[bot_data_dist["player_name"] == entity_name]
    
    if len(row_data) > 0:
        row_values = [row_data[metric].iloc[0] for metric in pick_rate_metrics]
        pick_rate_matrix.append(row_values)
    else:
        pick_rate_matrix.append([0] * len(pick_rate_metrics))

# Create success rate matrix
success_rate_matrix = []
for entity_type, entity_name in all_entities:
    if entity_type == "human":
        row_data = human_data[human_data["skill_bucket"] == entity_name]
    else:  # bot
        row_data = bot_data_dist[bot_data_dist["player_name"] == entity_name]
    
    if len(row_data) > 0:
        row_values = [row_data[metric].iloc[0] for metric in success_rate_metrics]
        success_rate_matrix.append(row_values)
    else:
        success_rate_matrix.append([0] * len(success_rate_metrics))

# Plot pick rate heatmap
if pick_rate_matrix:
    im1 = ax1.imshow(pick_rate_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_title("Contract Pick Rate Heatmap", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Contract Level")
    ax1.set_ylabel("Player Type")
    ax1.set_xticks(range(len(pick_rate_labels)))
    ax1.set_xticklabels(pick_rate_labels)
    ax1.set_yticks(range(len(all_entity_labels)))
    ax1.set_yticklabels(all_entity_labels)
    
    # Add text annotations
    for i in range(len(all_entity_labels)):
        for j in range(len(pick_rate_labels)):
            text = ax1.text(j, i, f'{pick_rate_matrix[i][j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Pick Rate', rotation=270, labelpad=15)

# Plot success rate heatmap
if success_rate_matrix:
    im2 = ax2.imshow(success_rate_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_title("Contract Success Rate Heatmap", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Contract Type")
    ax2.set_ylabel("Player Type")
    ax2.set_xticks(range(len(success_rate_labels)))
    ax2.set_xticklabels(success_rate_labels)
    ax2.set_yticks(range(len(all_entity_labels)))
    ax2.set_yticklabels(all_entity_labels)
    
    # Add text annotations
    for i in range(len(all_entity_labels)):
        for j in range(len(success_rate_labels)):
            text = ax2.text(j, i, f'{success_rate_matrix[i][j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Success Rate', rotation=270, labelpad=15)

# Rotate y-axis labels for better readability
ax1.tick_params(axis='y', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "contract_heatmaps.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Contract heatmaps saved to {FIG_DIR}/contract_heatmaps.png")
print("   - Left: Pick rate heatmap showing bidding preferences across all contract levels")
print("   - Right: Success rate heatmap showing performance across contract types")
print("   - Colors: Yellow/Red for pick rates, Red/Green for success rates")
print("   - Numbers show exact rates for each player/bot combination")

# Print heatmap data summary
print(f"\nHEATMAP DATA SUMMARY:")
print("-" * 25)
print(f"Players/Bots analyzed: {len(all_entity_labels)}")
print(f"Contract levels tracked: {len(pick_rate_labels)} (1-7)")
print(f"Contract types tracked: {len(success_rate_labels)} (Part Score, Game, Small/Grand Slam)")

if pick_rate_matrix:
    print(f"\nHIGHEST PICK RATES BY LEVEL:")
    print("-" * 30)
    for j, level_name in enumerate(pick_rate_labels):
        level_rates = [pick_rate_matrix[i][j] for i in range(len(all_entity_labels))]
        max_idx = level_rates.index(max(level_rates)) if level_rates else 0
        if max(level_rates) > 0:
            print(f"  {level_name:>8}: {all_entity_labels[max_idx]} ({max(level_rates):.1%})")

if success_rate_matrix:
    print(f"\nHIGHEST SUCCESS RATES BY TYPE:")
    print("-" * 32)
    for j, type_name in enumerate(success_rate_labels):
        type_rates = [success_rate_matrix[i][j] for i in range(len(all_entity_labels))]
        max_idx = type_rates.index(max(type_rates)) if type_rates else 0
        if max(type_rates) > 0:
            print(f"  {type_name:>11}: {all_entity_labels[max_idx]} ({max(type_rates):.1%})")

# 7) Shadow Deviation Analysis for each skill level
if shadow_data_available:
    for skill_level, shadow_data in shadow_datasets.items():
        # Clean bot names in shadow data
        shadow_data["bot_name"] = shadow_data["bot_name"].apply(clean_bot_name)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate deviation statistics by bot
        shadow_stats = shadow_data.groupby("bot_name").agg({
            'total_deviations': ['mean', 'std'],
            'bidding_deviations': ['mean', 'std'],
            'opening_lead_deviations': ['mean', 'std'],
            'play_deviations': ['mean', 'std'],
            'bot_name': 'count'  # Number of games analyzed
        }).round(3)
        
        # Flatten column names
        shadow_stats.columns = ['_'.join(col).strip() for col in shadow_stats.columns.values]
        shadow_stats.rename(columns={'bot_name_count': 'games_analyzed'}, inplace=True)
        shadow_stats = shadow_stats.reset_index()
        
        # Sort by bot order
        shadow_stats = sort_by_order(shadow_stats, BOT_ORDER, "bot_name")
        
        bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
        colors = [bot_colors[i % len(bot_colors)] for i in range(len(shadow_stats))]
        # Indent everything that was part of the shadow analysis
        # Plot 1: Average Total Deviations
        axes[0, 0].bar(range(len(shadow_stats)), shadow_stats['total_deviations_mean'], 
                       color=colors, alpha=0.8, 
                       yerr=shadow_stats['total_deviations_std'], capsize=5)
        axes[0, 0].set_title('Average Total Deviations per Game')
        axes[0, 0].set_ylabel('Number of Deviations')
        axes[0, 0].set_xticks(range(len(shadow_stats)))
        axes[0, 0].set_xticklabels(shadow_stats['bot_name'], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average Bidding Deviations
        axes[0, 1].bar(range(len(shadow_stats)), shadow_stats['bidding_deviations_mean'], 
                       color=colors, alpha=0.8,
                       yerr=shadow_stats['bidding_deviations_std'], capsize=5)
        axes[0, 1].set_title('Average Bidding Deviations per Game')
        axes[0, 1].set_ylabel('Number of Deviations')
        axes[0, 1].set_xticks(range(len(shadow_stats)))
        axes[0, 1].set_xticklabels(shadow_stats['bot_name'], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Average Play Deviations (including opening lead)
        play_combined = shadow_stats['play_deviations_mean'] + shadow_stats['opening_lead_deviations_mean']
        play_std_combined = np.sqrt(shadow_stats['play_deviations_std']**2 + shadow_stats['opening_lead_deviations_std']**2)
        
        axes[1, 0].bar(range(len(shadow_stats)), play_combined, 
                       color=colors, alpha=0.8,
                       yerr=play_std_combined, capsize=5)
        axes[1, 0].set_title('Average Play + Opening Lead Deviations per Game')
        axes[1, 0].set_ylabel('Number of Deviations')
        axes[1, 0].set_xticks(range(len(shadow_stats)))
        axes[1, 0].set_xticklabels(shadow_stats['bot_name'], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Deviation Rate (deviations per step analyzed)
        # Calculate average steps per game for rate calculation
        avg_steps = shadow_data.groupby("bot_name")['comparison_length'].mean()
        deviation_rates = []
        for bot in shadow_stats['bot_name']:
            if bot in avg_steps.index and avg_steps[bot] > 0:
                rate = shadow_stats[shadow_stats['bot_name'] == bot]['total_deviations_mean'].iloc[0] / avg_steps[bot]
                deviation_rates.append(rate)
            else:
                deviation_rates.append(0)
        
        axes[1, 1].bar(range(len(shadow_stats)), deviation_rates, 
                       color=colors, alpha=0.8)
        axes[1, 1].set_title('Deviation Rate (Deviations per Step)')
        axes[1, 1].set_ylabel('Deviation Rate')
        axes[1, 1].set_xticks(range(len(shadow_stats)))
        axes[1, 1].set_xticklabels(shadow_stats['bot_name'], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Bot Shadow Analysis: Deviation from {skill_level} Players', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"shadow_deviation_analysis_{skill_level.lower()}.png"), bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"\n✅ Shadow deviation analysis for {skill_level} saved to shadow_deviation_analysis_{skill_level.lower()}.png")
        print(f"   - Shows bot deviations from {skill_level} human players across 4 metrics")
        print("   - Top Left: Total deviations per game (all phases combined)")
        print("   - Top Right: Bidding deviations per game")
        print("   - Bottom Left: Play + Opening lead deviations per game")
        print("   - Bottom Right: Deviation rate (deviations per decision point)")
        
        # Print detailed shadow analysis summary
        print(f"\nSHADOW ANALYSIS SUMMARY - {skill_level.upper()} PLAYERS:")
        print("=" * (28 + len(skill_level)))
        print(f"Total games analyzed: {len(shadow_data)}")
        print(f"Bots analyzed: {len(shadow_stats)}")
        
        print(f"\nDETAILED DEVIATION STATISTICS:")
        print("-" * 35)
        for _, bot_row in shadow_stats.iterrows():
            bot_name = bot_row['bot_name']
            games = int(bot_row['games_analyzed'])
            total_dev = bot_row['total_deviations_mean']
            bid_dev = bot_row['bidding_deviations_mean']
            play_dev = bot_row['play_deviations_mean'] + bot_row['opening_lead_deviations_mean']
            
            print(f"\n{bot_name:>15} ({games} games):")
            print(f"  Total deviations: {total_dev:.2f} ± {bot_row['total_deviations_std']:.2f} per game")
            print(f"  Bidding dev:      {bid_dev:.2f} ± {bot_row['bidding_deviations_std']:.2f} per game")
            print(f"  Play+Lead dev:    {play_dev:.2f} ± {np.sqrt(bot_row['play_deviations_std']**2 + bot_row['opening_lead_deviations_std']**2):.2f} per game")
            
            # Calculate deviation rate
            if bot_name in avg_steps.index and avg_steps[bot_name] > 0:
                rate = total_dev / avg_steps[bot_name]
                print(f"  Deviation rate:   {rate:.3f} per decision point")
        
        # Find best and worst performing bots
        if len(shadow_stats) > 0:
            best_total_idx = shadow_stats['total_deviations_mean'].idxmin()
            worst_total_idx = shadow_stats['total_deviations_mean'].idxmax()
            best_bot = shadow_stats.loc[best_total_idx, 'bot_name']
            worst_bot = shadow_stats.loc[worst_total_idx, 'bot_name']
            best_score = shadow_stats.loc[best_total_idx, 'total_deviations_mean']
            worst_score = shadow_stats.loc[worst_total_idx, 'total_deviations_mean']
            
            print(f"\nPERFORMANCE RANKING for {skill_level}:")
            print("-" * (22 + len(skill_level)))
            print(f"Most similar to {skill_level}:  {best_bot} ({best_score:.2f} deviations/game)")
            print(f"Least similar to {skill_level}: {worst_bot} ({worst_score:.2f} deviations/game)")
            print(f"Performance gap: {worst_score - best_score:.2f} deviations/game")
        
        def categorize_deviation(human_action, bot_action, phase, action_type):
            """Categorize deviation patterns into meaningful groups"""
            # Use action_type instead of phase since phase seems to be incorrectly labeled
            if action_type == 'BID':
                # Bidding patterns
                bidding_calls = ['P', '1C', '1D', '1H', '1S', '1NT', '2C', '2D', '2H', '2S', '2NT', 
                                '3C', '3D', '3H', '3S', '3NT', '4C', '4D', '4H', '4S', '4NT',
                                '5C', '5D', '5H', '5S', '5NT', '6C', '6D', '6H', '6S', '6NT',
                                '7C', '7D', '7H', '7S', '7NT', 'X', 'XX']
                
                if human_action == 'P' and bot_action != 'P':
                    return "Bidding: Bot bid when human passed"
                elif human_action != 'P' and bot_action == 'P':
                    return "Bidding: Bot passed when human bid"
                elif human_action in ['X', 'XX'] or bot_action in ['X', 'XX']:
                    return "Bidding: Double/Redouble difference"
                else:
                    # Compare bidding levels and suits
                    try:
                        h_level = int(human_action[0]) if human_action[0].isdigit() else 0
                        b_level = int(bot_action[0]) if bot_action[0].isdigit() else 0
                        if h_level != b_level:
                            return "Bidding: Different level"
                        else:
                            return "Bidding: Different strain/suit"
                    except:
                        return "Bidding: Other difference"
            
            elif action_type == 'PLAY':
                # Card play patterns
                # Extract suit and rank from card notation (e.g., 'CA' -> suit='C', rank='A')
                def parse_card(card):
                    if len(card) >= 2:
                        return card[0], card[1:]  # suit, rank
                    return None, None
                
                h_suit, h_rank = parse_card(human_action)
                b_suit, b_rank = parse_card(bot_action)
                
                if not h_suit or not b_suit:
                    return "Play: Unknown pattern"
                
                # Same suit, different rank
                if h_suit == b_suit:
                    # Define rank order (low to high)
                    rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                    try:
                        h_idx = rank_order.index(h_rank)
                        b_idx = rank_order.index(b_rank)
                        if b_idx > h_idx:
                            return "Play: Bot played higher in same suit"
                        else:
                            return "Play: Bot played lower in same suit"
                    except ValueError:
                        return "Play: Same suit, different rank"
                
                # Different suits
                else:
                    # Check if one is trump (this would need trump info from context, simplified here)
                    # For now, we'll use heuristics - Spades and Hearts often trump in many contracts
                    trump_suits = ['S', 'H']  # This is a simplification
                    
                    if h_suit in trump_suits and b_suit not in trump_suits:
                        return "Play: Human played trump, bot non-trump"
                    elif b_suit in trump_suits and h_suit not in trump_suits:
                        return "Play: Bot played trump, human non-trump"
                    else:
                        return "Play: Different non-trump suits"
            
            else:
                return f"Unknown: {action_type}"
        
        # Create categorized deviation analysis
        # Extract and categorize deviation patterns from step details
        categorized_patterns = defaultdict(lambda: defaultdict(int))  # {bot_name: {category: count}}
        
        for _, row in shadow_data.iterrows():
            bot_name = row['bot_name']
            try:
                step_details = ast.literal_eval(row['step_details_json'])
                for step in step_details:
                    if step.get('deviation', False):
                        bot_pred = step.get('bot_prediction', '')
                        player_act = step.get('player_actual', '')
                        phase = step.get('phase', 'unknown')
                        action_type = step.get('action_type', 'unknown')
                        
                        if bot_pred != player_act and bot_pred and player_act:
                            category = categorize_deviation(player_act, bot_pred, phase, action_type)
                            categorized_patterns[bot_name][category] += 1
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        
        if categorized_patterns:
            # Get all categories across all bots
            all_categories = set()
            for bot_patterns in categorized_patterns.values():
                all_categories.update(bot_patterns.keys())
            all_categories = sorted(list(all_categories))
            
            # Prepare data for stacked bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Sort bots by order
            sorted_bots = [bot for bot in BOT_ORDER if bot in categorized_patterns.keys()]
            
            # Calculate percentages for each bot
            bot_category_percentages = {}
            for bot in sorted_bots:
                total_deviations = sum(categorized_patterns[bot].values())
                if total_deviations > 0:
                    bot_category_percentages[bot] = {}
                    for category in all_categories:
                        count = categorized_patterns[bot].get(category, 0)
                        bot_category_percentages[bot][category] = (count / total_deviations) * 100
                else:
                    bot_category_percentages[bot] = {category: 0 for category in all_categories}
            
            # Create stacked bar chart - Categorized patterns
            bottoms = [0] * len(sorted_bots)
            colors = plt.cm.Set3(np.linspace(0, 1, len(all_categories)))
            
            for i, category in enumerate(all_categories):
                percentages = [bot_category_percentages[bot][category] for bot in sorted_bots]
                ax1.bar(range(len(sorted_bots)), percentages, bottom=bottoms, 
                       label=category, color=colors[i], alpha=0.8)
                bottoms = [b + p for b, p in zip(bottoms, percentages)]
            
            ax1.set_title('Categorized Deviation Patterns by Bot', fontweight='bold')
            ax1.set_xlabel('Bot')
            ax1.set_ylabel('Percentage of Total Deviations')
            ax1.set_xticks(range(len(sorted_bots)))
            ax1.set_xticklabels(sorted_bots, rotation=45)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Create second chart focusing on bidding vs play patterns with subcategories
            bidding_categories = {}
            play_categories = {}
            
            for bot in sorted_bots:
                bidding_categories[bot] = {}
                play_categories[bot] = {}
                
                total_bid = sum(count for cat, count in categorized_patterns[bot].items() if cat.startswith('Bidding:'))
                total_play = sum(count for cat, count in categorized_patterns[bot].items() if cat.startswith('Play:'))
                total_all = total_bid + total_play
                
                if total_all > 0:
                    for category, count in categorized_patterns[bot].items():
                        percentage = (count / total_all) * 100
                        if category.startswith('Bidding:'):
                            short_cat = category.replace('Bidding: ', '')
                            bidding_categories[bot][short_cat] = percentage
                        elif category.startswith('Play:'):
                            short_cat = category.replace('Play: ', '')
                            play_categories[bot][short_cat] = percentage
            
            # Get all subcategories
            all_bid_subcats = set()
            all_play_subcats = set()
            for bot_cats in bidding_categories.values():
                all_bid_subcats.update(bot_cats.keys())
            for bot_cats in play_categories.values():
                all_play_subcats.update(bot_cats.keys())
            
            all_bid_subcats = sorted(list(all_bid_subcats))
            all_play_subcats = sorted(list(all_play_subcats))
            
            # Plot detailed breakdown
            x_pos = np.arange(len(sorted_bots))
            width = 0.35
            
            # Sum up bidding and play totals for each bot
            bid_totals = []
            play_totals = []
            
            for bot in sorted_bots:
                bid_total = sum(bidding_categories[bot].values())
                play_total = sum(play_categories[bot].values())
                bid_totals.append(bid_total)
                play_totals.append(play_total)
            
            ax2.bar(x_pos - width/2, bid_totals, width, label='Bidding Deviations', 
                   color='lightcoral', alpha=0.8)
            ax2.bar(x_pos + width/2, play_totals, width, label='Play Deviations', 
                   color='lightblue', alpha=0.8)
            
            ax2.set_title('Bidding vs Play Deviation Distribution', fontweight='bold')
            ax2.set_xlabel('Bot')
            ax2.set_ylabel('Percentage of Total Deviations')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(sorted_bots, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(f'Shadow Deviation Patterns Analysis - {skill_level} Players', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"shadow_deviation_patterns_{skill_level.lower()}.png"), bbox_inches="tight", dpi=300)
            plt.close()
            
            print(f"\n✅ Shadow deviation patterns for {skill_level} saved to shadow_deviation_patterns_{skill_level.lower()}.png")
            print(f"   - Left: Detailed categorization of all deviation types")
            print(f"   - Right: Bidding vs Play deviation distribution")
            print(f"   - Shows WHY bots deviate from {skill_level} human players")
            
            # Print detailed pattern analysis
            print(f"\nDEVIATION PATTERN ANALYSIS - {skill_level.upper()} PLAYERS:")
            print("=" * (32 + len(skill_level)))
            print(f"Total deviation categories identified: {len(all_categories)}")
            print(f"Bots with pattern data: {len(sorted_bots)}")
            
            # Print top categories across all bots
            category_totals = defaultdict(int)
            for bot_patterns in categorized_patterns.values():
                for category, count in bot_patterns.items():
                    category_totals[category] += count
            
            top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTOP 5 DEVIATION TYPES OVERALL:")
            print("-" * 32)
            total_deviations = sum(category_totals.values())
            for category, count in top_categories:
                percentage = (count / total_deviations) * 100 if total_deviations > 0 else 0
                print(f"  {category:<35}: {count:>4} ({percentage:.1f}%)")
            
            # Print bot-specific breakdown
            print(f"\nBIDDING vs PLAY BREAKDOWN BY BOT:")
            print("-" * 36)
            for bot in sorted_bots:
                total_bot_deviations = sum(categorized_patterns[bot].values())
                bid_count = sum(count for cat, count in categorized_patterns[bot].items() if cat.startswith('Bidding:'))
                play_count = sum(count for cat, count in categorized_patterns[bot].items() if cat.startswith('Play:'))
                
                if total_bot_deviations > 0:
                    bid_pct = (bid_count / total_bot_deviations) * 100
                    play_pct = (play_count / total_bot_deviations) * 100
                    print(f"  {bot:>15}: Bidding {bid_pct:>5.1f}% ({bid_count:>3}), Play {play_pct:>5.1f}% ({play_count:>3})")
            
            # Identify specialization patterns
            print(f"\nSPECIALIZATION PATTERNS:")
            print("-" * 23)
            bidding_specialists = []
            play_specialists = []
            
            for bot in sorted_bots:
                total_bot = sum(categorized_patterns[bot].values())
                if total_bot > 10:  # Only consider bots with sufficient data
                    bid_ratio = sum(count for cat, count in categorized_patterns[bot].items() if cat.startswith('Bidding:')) / total_bot
                    if bid_ratio > 0.7:
                        bidding_specialists.append((bot, bid_ratio))
                    elif bid_ratio < 0.3:
                        play_specialists.append((bot, bid_ratio))
            
            if bidding_specialists:
                print(f"  Bidding-deviation specialists:")
                for bot, ratio in bidding_specialists:
                    print(f"    {bot}: {ratio:.1%} of deviations in bidding")
            
            if play_specialists:
                print(f"  Play-deviation specialists:")
                for bot, ratio in play_specialists:
                    print(f"    {bot}: {1-ratio:.1%} of deviations in play")
            
            if not bidding_specialists and not play_specialists:
                print(f"  No clear specialization patterns found - bots show mixed deviation types")
            
            print(f"\nDeviation patterns analyzed for {skill_level}: {len(all_categories)} categories across {len(sorted_bots)} bots")

# ============================================================================
# ADDITIONAL ANALYSIS VISUALIZATIONS
# ============================================================================

# ============================================================================
# 8) THIN CONTRACT RATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. THIN CONTRACT RATE ANALYSIS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Separate humans and bots for thin contract analysis
human_thin = thin_contract_data[thin_contract_data["is_bot"] == False]
bot_thin = thin_contract_data[thin_contract_data["is_bot"] == True]

print(f"\nTHIN CONTRACT RATE ANALYSIS:")
print(f"- Human players analyzed: {len(human_thin)}")
print(f"- Bot players analyzed: {len(bot_thin)}")

# Create bins for human thin rate distribution
ax1.hist(human_thin["thin_rate"], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_title("Distribution of Thin Contract Rates - Humans")
ax1.set_xlabel("Thin Contract Rate")
ax1.set_ylabel("Number of Players")
ax1.grid(True, alpha=0.3)

# Bot comparison - bar chart
if len(bot_thin) > 0:
    bot_summary = bot_thin.groupby("player_name")["thin_rate"].agg(['mean', 'std', 'count']).reset_index()
    bot_summary = sort_by_order(bot_summary, BOT_ORDER, "player_name")
    
    # Define colors for bots
    bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
    colors = [bot_colors[i % len(bot_colors)] for i in range(len(bot_summary))]
    
    bars = ax2.bar(range(len(bot_summary)), bot_summary['mean'], 
                   color=colors, alpha=0.8, 
                   yerr=bot_summary['std'], capsize=5)
    ax2.set_title("Thin Contract Rate - Bot Comparison")
    ax2.set_xlabel("Bot")
    ax2.set_ylabel("Average Thin Contract Rate")
    ax2.set_xticks(range(len(bot_summary)))
    ax2.set_xticklabels(bot_summary['player_name'], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    print(f"\nTHIN CONTRACT RATES BY BOT:")
    print("-" * 32)
    for _, bot_row in bot_summary.iterrows():
        bot_name = bot_row['player_name']
        rate = bot_row['mean']
        std = bot_row['std']
        count = bot_row['count']
        print(f"  {bot_name:>15}: {rate:.3f} ± {std:.3f} ({count} games)")

# Add human statistics
human_mean = human_thin["thin_rate"].mean()
human_std = human_thin["thin_rate"].std()
human_median = human_thin["thin_rate"].median()

print(f"\nHUMAN THIN CONTRACT STATISTICS:")
print("-" * 33)
print(f"  Mean rate: {human_mean:.3f} ± {human_std:.3f}")
print(f"  Median rate: {human_median:.3f}")
print(f"  Min rate: {human_thin['thin_rate'].min():.3f}")
print(f"  Max rate: {human_thin['thin_rate'].max():.3f}")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "thin_contract_analysis.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Thin contract analysis saved to {FIG_DIR}/thin_contract_analysis.png")
print("   - Left: Distribution of thin contract rates among human players")
print("   - Right: Bot comparison of average thin contract rates")
print("   - Thin contracts are risky bids with marginal success chances")

# ============================================================================
# 9) STRATEGIC RATES ANALYSIS (GAME & SLAM RATES)
# ============================================================================
print("\n" + "="*80)
print("9. STRATEGIC RATES ANALYSIS - GAME & SLAM RATES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Separate humans and bots
human_strategic = strategic_rates_data[strategic_rates_data["is_bot"] == False]
bot_strategic = strategic_rates_data[strategic_rates_data["is_bot"] == True]

print(f"\nSTRATEGIC RATES ANALYSIS:")
print(f"- Human players analyzed: {len(human_strategic)}")
print(f"- Bot players analyzed: {len(bot_strategic)}")

# Human game rate distribution
axes[0, 0].hist(human_strategic["game_rate"], bins=30, alpha=0.7, color='green', edgecolor='black')
axes[0, 0].set_title("Distribution of Game Contract Rates - Humans")
axes[0, 0].set_xlabel("Game Rate")
axes[0, 0].set_ylabel("Number of Players")
axes[0, 0].grid(True, alpha=0.3)

# Human slam rate distribution  
axes[0, 1].hist(human_strategic["slam_rate"], bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[0, 1].set_title("Distribution of Slam Contract Rates - Humans")
axes[0, 1].set_xlabel("Slam Rate")
axes[0, 1].set_ylabel("Number of Players")
axes[0, 1].grid(True, alpha=0.3)

# Bot strategic rates comparison
if len(bot_strategic) > 0:
    bot_strategic_summary = bot_strategic.groupby("player_name").agg({
        'game_rate': ['mean', 'std', 'count'],
        'slam_rate': ['mean', 'std']
    }).round(4)
    
    bot_strategic_summary.columns = ['game_mean', 'game_std', 'count', 'slam_mean', 'slam_std']
    bot_strategic_summary = bot_strategic_summary.reset_index()
    bot_strategic_summary = sort_by_order(bot_strategic_summary, BOT_ORDER, "player_name")
    
    # Bot game rates
    bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
    colors = [bot_colors[i % len(bot_colors)] for i in range(len(bot_strategic_summary))]
    
    bars = axes[1, 0].bar(range(len(bot_strategic_summary)), bot_strategic_summary['game_mean'], 
                         color=colors, alpha=0.8, 
                         yerr=bot_strategic_summary['game_std'], capsize=5)
    axes[1, 0].set_title("Game Contract Rate - Bot Comparison")
    axes[1, 0].set_xlabel("Bot")
    axes[1, 0].set_ylabel("Average Game Rate")
    axes[1, 0].set_xticks(range(len(bot_strategic_summary)))
    axes[1, 0].set_xticklabels(bot_strategic_summary['player_name'], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bot slam rates
    bars = axes[1, 1].bar(range(len(bot_strategic_summary)), bot_strategic_summary['slam_mean'], 
                         color=colors, alpha=0.8, 
                         yerr=bot_strategic_summary['slam_std'], capsize=5)
    axes[1, 1].set_title("Slam Contract Rate - Bot Comparison")
    axes[1, 1].set_xlabel("Bot")
    axes[1, 1].set_ylabel("Average Slam Rate")
    axes[1, 1].set_xticks(range(len(bot_strategic_summary)))
    axes[1, 1].set_xticklabels(bot_strategic_summary['player_name'], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    print(f"\nSTRATEGIC RATES BY BOT:")
    print("-" * 25)
    for _, bot_row in bot_strategic_summary.iterrows():
        bot_name = bot_row['player_name']
        game_rate = bot_row['game_mean']
        game_std = bot_row['game_std']
        slam_rate = bot_row['slam_mean']
        slam_std = bot_row['slam_std']
        count = int(bot_row['count'])
        print(f"  {bot_name:>15}: Game {game_rate:.3f} ± {game_std:.3f}, "
              f"Slam {slam_rate:.3f} ± {slam_std:.3f} ({count} games)")

# Human statistics
human_game_stats = {
    'mean': human_strategic["game_rate"].mean(),
    'std': human_strategic["game_rate"].std(),
    'median': human_strategic["game_rate"].median()
}

human_slam_stats = {
    'mean': human_strategic["slam_rate"].mean(),
    'std': human_strategic["slam_rate"].std(),
    'median': human_strategic["slam_rate"].median()
}

print(f"\nHUMAN STRATEGIC STATISTICS:")
print("-" * 29)
print(f"  Game contracts:")
print(f"    Mean: {human_game_stats['mean']:.3f} ± {human_game_stats['std']:.3f}")
print(f"    Median: {human_game_stats['median']:.3f}")
print(f"  Slam contracts:")
print(f"    Mean: {human_slam_stats['mean']:.3f} ± {human_slam_stats['std']:.3f}")
print(f"    Median: {human_slam_stats['median']:.3f}")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "strategic_rates_analysis.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Strategic rates analysis saved to {FIG_DIR}/strategic_rates_analysis.png")
print("   - Top: Human distributions for game and slam contract rates")
print("   - Bottom: Bot comparisons for game and slam bidding frequencies")
print("   - Shows bidding ambition and risk-taking in high-level contracts")

# ============================================================================
# 10) PERFORMANCE BY CONTRACT AND STRAIN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("10. PERFORMANCE BY CONTRACT AND STRAIN ANALYSIS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Separate humans and bots
human_perf = performance_contract_strain_data[performance_contract_strain_data["is_bot"] == False]
bot_perf = performance_contract_strain_data[performance_contract_strain_data["is_bot"] == True]

print(f"\nCONTRACT & STRAIN PERFORMANCE ANALYSIS:")
print(f"- Human performance records: {len(human_perf)}")
print(f"- Bot performance records: {len(bot_perf)}")

# Human performance by contract category
human_contract_perf = human_perf.groupby("contract_category")["mean_rawscore"].agg(['mean', 'std', 'count']).reset_index()
human_contract_perf.columns = ['contract_category', 'avg_score', 'std_score', 'count']

contract_order = ['Part-score', 'Game', 'Small-slam', 'Grand-slam'] 
available_contracts = [c for c in contract_order if c in human_contract_perf['contract_category'].values]
human_contract_perf = human_contract_perf[human_contract_perf['contract_category'].isin(available_contracts)]
human_contract_perf['contract_category'] = pd.Categorical(human_contract_perf['contract_category'], 
                                                         categories=available_contracts, ordered=True)
human_contract_perf = human_contract_perf.sort_values('contract_category')

contract_colors = ['steelblue', 'green', 'orange', 'red']
colors = [contract_colors[i] for i in range(len(human_contract_perf))]

bars = axes[0, 0].bar(range(len(human_contract_perf)), human_contract_perf['avg_score'], 
                     color=colors, alpha=0.8, 
                     yerr=human_contract_perf['std_score'], capsize=5)
axes[0, 0].set_title("Human Performance by Contract Category")
axes[0, 0].set_xlabel("Contract Category")
axes[0, 0].set_ylabel("Average Raw Score")
axes[0, 0].set_xticks(range(len(human_contract_perf)))
axes[0, 0].set_xticklabels(human_contract_perf['contract_category'], rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Human performance by strain
human_strain_perf = human_perf.groupby("final_strain")["mean_rawscore"].agg(['mean', 'std', 'count']).reset_index()
human_strain_perf.columns = ['strain', 'avg_score', 'std_score', 'count']

strain_order = ['C', 'D', 'H', 'S', 'N']  # Clubs, Diamonds, Hearts, Spades, NoTrump
available_strains = [s for s in strain_order if s in human_strain_perf['strain'].values]
human_strain_perf = human_strain_perf[human_strain_perf['strain'].isin(available_strains)]
human_strain_perf['strain'] = pd.Categorical(human_strain_perf['strain'], 
                                            categories=available_strains, ordered=True)
human_strain_perf = human_strain_perf.sort_values('strain')

strain_colors = ['green', 'red', 'red', 'black', 'blue']
strain_colors_filtered = [strain_colors[strain_order.index(s)] for s in human_strain_perf['strain']]

bars = axes[0, 1].bar(range(len(human_strain_perf)), human_strain_perf['avg_score'], 
                     color=strain_colors_filtered, alpha=0.8, 
                     yerr=human_strain_perf['std_score'], capsize=5)
axes[0, 1].set_title("Human Performance by Final Strain")
axes[0, 1].set_xlabel("Final Strain")
axes[0, 1].set_ylabel("Average Raw Score")
axes[0, 1].set_xticks(range(len(human_strain_perf)))
axes[0, 1].set_xticklabels(human_strain_perf['strain'])
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Bot performance analysis
if len(bot_perf) > 0:
    # Bot performance by contract category
    bot_contract_perf = bot_perf.groupby(["player_name", "contract_category"])["mean_rawscore"].mean().reset_index()
    bot_contract_pivot = bot_contract_perf.pivot(index="player_name", columns="contract_category", values="mean_rawscore")
    bot_contract_pivot = bot_contract_pivot.reindex(BOT_ORDER)
    
    # Create heatmap for bot contract performance
    if not bot_contract_pivot.empty:
        im1 = axes[1, 0].imshow(bot_contract_pivot.values, cmap='RdYlGn', aspect='auto', 
                               vmin=-200, vmax=200)
        axes[1, 0].set_title("Bot Performance Heatmap by Contract Category")
        axes[1, 0].set_xlabel("Contract Category")
        axes[1, 0].set_ylabel("Bot")
        axes[1, 0].set_xticks(range(len(bot_contract_pivot.columns)))
        axes[1, 0].set_xticklabels(bot_contract_pivot.columns, rotation=45)
        axes[1, 0].set_yticks(range(len(bot_contract_pivot.index)))
        axes[1, 0].set_yticklabels(bot_contract_pivot.index)
        
        # Add text annotations
        for i in range(len(bot_contract_pivot.index)):
            for j in range(len(bot_contract_pivot.columns)):
                value = bot_contract_pivot.iloc[i, j]
                if not pd.isna(value):
                    text = axes[1, 0].text(j, i, f'{value:.0f}',
                                          ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
        cbar1.set_label('Average Raw Score', rotation=270, labelpad=15)
    
    # Bot performance by strain
    bot_strain_perf = bot_perf.groupby(["player_name", "final_strain"])["mean_rawscore"].mean().reset_index()
    bot_strain_pivot = bot_strain_perf.pivot(index="player_name", columns="final_strain", values="mean_rawscore")
    bot_strain_pivot = bot_strain_pivot.reindex(BOT_ORDER)
    
    if not bot_strain_pivot.empty:
        im2 = axes[1, 1].imshow(bot_strain_pivot.values, cmap='RdYlGn', aspect='auto',
                               vmin=-200, vmax=200)
        axes[1, 1].set_title("Bot Performance Heatmap by Final Strain")
        axes[1, 1].set_xlabel("Final Strain")
        axes[1, 1].set_ylabel("Bot")
        axes[1, 1].set_xticks(range(len(bot_strain_pivot.columns)))
        axes[1, 1].set_xticklabels(bot_strain_pivot.columns)
        axes[1, 1].set_yticks(range(len(bot_strain_pivot.index)))
        axes[1, 1].set_yticklabels(bot_strain_pivot.index)
        
        # Add text annotations
        for i in range(len(bot_strain_pivot.index)):
            for j in range(len(bot_strain_pivot.columns)):
                value = bot_strain_pivot.iloc[i, j]
                if not pd.isna(value):
                    text = axes[1, 1].text(j, i, f'{value:.0f}',
                                          ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
        cbar2.set_label('Average Raw Score', rotation=270, labelpad=15)

# Print detailed statistics
print(f"\nHUMAN PERFORMANCE BY CONTRACT CATEGORY:")
print("-" * 41)
for _, row in human_contract_perf.iterrows():
    print(f"  {row['contract_category']:>11}: {row['avg_score']:>6.1f} ± {row['std_score']:>5.1f} ({row['count']} contracts)")

print(f"\nHUMAN PERFORMANCE BY STRAIN:")
print("-" * 30)
strain_names = {'C': 'Clubs', 'D': 'Diamonds', 'H': 'Hearts', 'S': 'Spades', 'N': 'NoTrump'}
for _, row in human_strain_perf.iterrows():
    strain_name = strain_names.get(row['strain'], row['strain'])
    print(f"  {strain_name:>8}: {row['avg_score']:>6.1f} ± {row['std_score']:>5.1f} ({row['count']} contracts)")

if len(bot_perf) > 0:
    print(f"\nBOT PERFORMANCE SUMMARY:")
    print("-" * 25)
    bot_overall_perf = bot_perf.groupby("player_name")["mean_rawscore"].agg(['mean', 'std', 'count']).reset_index()
    bot_overall_perf = sort_by_order(bot_overall_perf, BOT_ORDER, "player_name")
    
    for _, bot_row in bot_overall_perf.iterrows():
        print(f"  {bot_row['player_name']:>15}: {bot_row['mean']:>6.1f} ± {bot_row['std']:>5.1f} ({bot_row['count']} contracts)")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "performance_contract_strain.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Performance by contract & strain saved to {FIG_DIR}/performance_contract_strain.png")
print("   - Top: Human performance by contract category and strain")
print("   - Bottom: Bot performance heatmaps showing strengths/weaknesses")
print("   - Raw scores: positive = good, negative = poor performance")

# ============================================================================
# 11) PLAY PHASE SHARPNESS ANALYSIS  
# ============================================================================
print("\n" + "="*80)
print("11. PLAY PHASE SHARPNESS ANALYSIS - OVERTRICKS & UNDERTRICKS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Separate humans and bots
human_sharpness = play_sharpness_data[play_sharpness_data["is_bot"] == False]
bot_sharpness = play_sharpness_data[play_sharpness_data["is_bot"] == True]

print(f"\nPLAY SHARPNESS ANALYSIS:")
print(f"- Human players analyzed: {len(human_sharpness)}")
print(f"- Bot players analyzed: {len(bot_sharpness)}")

# Human overtricks distribution
axes[0, 0].hist(human_sharpness["avg_overtricks"], bins=30, alpha=0.7, color='green', edgecolor='black')
axes[0, 0].set_title("Distribution of Average Overtricks - Humans")
axes[0, 0].set_xlabel("Average Overtricks per Game")
axes[0, 0].set_ylabel("Number of Players")
axes[0, 0].grid(True, alpha=0.3)

# Human undertricks distribution
axes[0, 1].hist(human_sharpness["avg_undertricks"], bins=30, alpha=0.7, color='red', edgecolor='black')
axes[0, 1].set_title("Distribution of Average Undertricks - Humans")
axes[0, 1].set_xlabel("Average Undertricks per Game")
axes[0, 1].set_ylabel("Number of Players")
axes[0, 1].grid(True, alpha=0.3)

# Bot sharpness comparison
if len(bot_sharpness) > 0:
    bot_sharpness_summary = bot_sharpness.groupby("player_name").agg({
        'avg_overtricks': ['mean', 'std', 'count'],
        'avg_undertricks': ['mean', 'std']
    }).round(4)
    
    bot_sharpness_summary.columns = ['overtricks_mean', 'overtricks_std', 'count', 'undertricks_mean', 'undertricks_std']
    bot_sharpness_summary = bot_sharpness_summary.reset_index()
    bot_sharpness_summary = sort_by_order(bot_sharpness_summary, BOT_ORDER, "player_name")
    
    # Bot overtricks
    bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
    colors = [bot_colors[i % len(bot_colors)] for i in range(len(bot_sharpness_summary))]
    
    bars = axes[1, 0].bar(range(len(bot_sharpness_summary)), bot_sharpness_summary['overtricks_mean'], 
                         color=colors, alpha=0.8, 
                         yerr=bot_sharpness_summary['overtricks_std'], capsize=5)
    axes[1, 0].set_title("Average Overtricks - Bot Comparison")
    axes[1, 0].set_xlabel("Bot")
    axes[1, 0].set_ylabel("Average Overtricks per Game")
    axes[1, 0].set_xticks(range(len(bot_sharpness_summary)))
    axes[1, 0].set_xticklabels(bot_sharpness_summary['player_name'], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bot undertricks
    bars = axes[1, 1].bar(range(len(bot_sharpness_summary)), bot_sharpness_summary['undertricks_mean'], 
                         color=colors, alpha=0.8, 
                         yerr=bot_sharpness_summary['undertricks_std'], capsize=5)
    axes[1, 1].set_title("Average Undertricks - Bot Comparison")
    axes[1, 1].set_xlabel("Bot")
    axes[1, 1].set_ylabel("Average Undertricks per Game")
    axes[1, 1].set_xticks(range(len(bot_sharpness_summary)))
    axes[1, 1].set_xticklabels(bot_sharpness_summary['player_name'], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    print(f"\nPLAY SHARPNESS BY BOT:")
    print("-" * 24)
    for _, bot_row in bot_sharpness_summary.iterrows():
        bot_name = bot_row['player_name']
        over_mean = bot_row['overtricks_mean']
        over_std = bot_row['overtricks_std']
        under_mean = bot_row['undertricks_mean']
        under_std = bot_row['undertricks_std']
        count = int(bot_row['count'])
        print(f"  {bot_name:>15}: Over {over_mean:.3f} ± {over_std:.3f}, "
              f"Under {under_mean:.3f} ± {under_std:.3f} ({count} games)")

# Human statistics
human_over_stats = {
    'mean': human_sharpness["avg_overtricks"].mean(),
    'std': human_sharpness["avg_overtricks"].std(),
    'median': human_sharpness["avg_overtricks"].median()
}

human_under_stats = {
    'mean': human_sharpness["avg_undertricks"].mean(),
    'std': human_sharpness["avg_undertricks"].std(),
    'median': human_sharpness["avg_undertricks"].median()
}

print(f"\nHUMAN PLAY SHARPNESS STATISTICS:")
print("-" * 34)
print(f"  Overtricks:")
print(f"    Mean: {human_over_stats['mean']:.3f} ± {human_over_stats['std']:.3f}")
print(f"    Median: {human_over_stats['median']:.3f}")
print(f"  Undertricks:")
print(f"    Mean: {human_under_stats['mean']:.3f} ± {human_under_stats['std']:.3f}")
print(f"    Median: {human_under_stats['median']:.3f}")

# Calculate sharpness ratio (overtricks / undertricks) for insights
if len(bot_sharpness) > 0:
    print(f"\nSHARPNESS RATIOS (Overtricks/Undertricks):")
    print("-" * 43)
    for _, bot_row in bot_sharpness_summary.iterrows():
        if bot_row['undertricks_mean'] > 0:
            ratio = bot_row['overtricks_mean'] / bot_row['undertricks_mean']
            print(f"  {bot_row['player_name']:>15}: {ratio:.3f}")

human_sharpness_ratio = human_over_stats['mean'] / human_under_stats['mean'] if human_under_stats['mean'] > 0 else 0
print(f"  {'Human Average':>15}: {human_sharpness_ratio:.3f}")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "play_sharpness_analysis.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Play sharpness analysis saved to {FIG_DIR}/play_sharpness_analysis.png")
print("   - Top: Human distributions for overtricks and undertricks")
print("   - Bottom: Bot comparisons showing play precision")
print("   - Higher overtricks = better execution, Lower undertricks = better accuracy")

# ============================================================================
# 12) BOT OPENING LEAD MATCH ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("12. BOT OPENING LEAD MATCH ANALYSIS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

if len(bot_opening_lead_data) > 0:
    # Sort bot opening lead data by our standard order
    bot_lead_sorted = sort_by_order(bot_opening_lead_data.copy(), BOT_ORDER, "player_name")
    
    # Define colors for bots
    bot_colors = ['darkred', 'gold', 'teal', 'magenta', 'olive', 'cyan', 'lime']
    colors = [bot_colors[i % len(bot_colors)] for i in range(len(bot_lead_sorted))]
    
    # Opening lead match rate comparison
    bars = ax1.bar(range(len(bot_lead_sorted)), bot_lead_sorted['lead_match_rate'], 
                   color=colors, alpha=0.8)
    ax1.set_title("Bot Opening Lead Match Rate")
    ax1.set_xlabel("Bot")
    ax1.set_ylabel("Opening Lead Match Rate")
    ax1.set_xticks(range(len(bot_lead_sorted)))
    ax1.set_xticklabels(bot_lead_sorted['player_name'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(bot_lead_sorted.iterrows()):
        ax1.text(i, row['lead_match_rate'] + 0.005, f'{row["lead_match_rate"]:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Number of leads analyzed
    bars = ax2.bar(range(len(bot_lead_sorted)), bot_lead_sorted['n_leads'], 
                   color=colors, alpha=0.8)
    ax2.set_title("Number of Opening Leads Analyzed")
    ax2.set_xlabel("Bot")
    ax2.set_ylabel("Number of Opening Leads")
    ax2.set_xticks(range(len(bot_lead_sorted)))
    ax2.set_xticklabels(bot_lead_sorted['player_name'], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(bot_lead_sorted.iterrows()):
        ax2.text(i, row['n_leads'] + 100, f'{row["n_leads"]:,}', 
                ha='center', va='bottom', fontweight='bold')
    
    print(f"\nOPENING LEAD MATCH ANALYSIS:")
    print("-" * 31)
    print(f"Total bots analyzed: {len(bot_lead_sorted)}")
    
    print(f"\nOPENING LEAD MATCH RATES:")
    print("-" * 27)
    for _, bot_row in bot_lead_sorted.iterrows():
        bot_name = bot_row['player_name']
        match_rate = bot_row['lead_match_rate']
        n_leads = bot_row['n_leads']
        print(f"  {bot_name:>15}: {match_rate:.3f} ({n_leads:,} leads analyzed)")
    
    # Find best and worst performers
    best_bot_idx = bot_lead_sorted['lead_match_rate'].idxmax()
    worst_bot_idx = bot_lead_sorted['lead_match_rate'].idxmin()
    
    best_bot = bot_lead_sorted.loc[best_bot_idx, 'player_name']
    best_rate = bot_lead_sorted.loc[best_bot_idx, 'lead_match_rate']
    worst_bot = bot_lead_sorted.loc[worst_bot_idx, 'player_name']
    worst_rate = bot_lead_sorted.loc[worst_bot_idx, 'lead_match_rate']
    
    print(f"\nOPENING LEAD PERFORMANCE RANKING:")
    print("-" * 35)
    print(f"Best matching bot:  {best_bot} ({best_rate:.3f})")
    print(f"Worst matching bot: {worst_bot} ({worst_rate:.3f})")
    print(f"Performance gap: {best_rate - worst_rate:.3f}")
    
    # Overall statistics
    avg_match_rate = bot_lead_sorted['lead_match_rate'].mean()
    std_match_rate = bot_lead_sorted['lead_match_rate'].std()
    total_leads = bot_lead_sorted['n_leads'].sum()
    
    print(f"\nOVERALL STATISTICS:")
    print("-" * 20)
    print(f"Average match rate: {avg_match_rate:.3f} ± {std_match_rate:.3f}")
    print(f"Total leads analyzed: {total_leads:,}")
    print(f"Average leads per bot: {total_leads / len(bot_lead_sorted):,.0f}")

else:
    ax1.text(0.5, 0.5, "No opening lead data available", 
            ha='center', va='center', transform=ax1.transAxes, fontsize=16)
    ax2.text(0.5, 0.5, "No opening lead data available", 
            ha='center', va='center', transform=ax2.transAxes, fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "bot_opening_lead_analysis.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"\n✅ Bot opening lead analysis saved to {FIG_DIR}/bot_opening_lead_analysis.png")
print("   - Left: Opening lead match rates comparing bot choices to expert choices")
print("   - Right: Number of opening lead decisions analyzed per bot")
print("   - Higher match rate indicates better opening lead selection")

print("✅ Advanced visualizations created successfully!")
print("\n" + "="*80)
print("VISUALIZATION SUMMARY - ALL PLOTS CREATED")
print("="*80)

print("\nFILES CREATED:")
print("-" * 15)
print("1. skill_bucket_comparison.png       - 6-metric comparison of humans vs bots")
print("2. bot_skill_matching.png           - Bot assignment to skill levels + performance")
print("3. risk_analysis.png                - Risk aversion and doubling behavior analysis") 
print("4. contract_success_analysis.png    - Success rates for Part Score/Game/Slam contracts")
print("5. contract_pick_rate.png           - Bidding preferences by contract level")
print("6. contract_heatmaps.png            - Pick rate and success rate heatmaps")

print("\nADDITIONAL ANALYSIS PLOTS:")
print("-" * 26)
print("8. thin_contract_analysis.png       - Thin contract rate distributions and comparisons")
print("9. strategic_rates_analysis.png     - Game and slam bidding rate analysis")
print("10. performance_contract_strain.png  - Performance by contract category and strain")
print("11. play_sharpness_analysis.png      - Overtricks and undertricks analysis")
print("12. bot_opening_lead_analysis.png    - Bot opening lead matching performance")

shadow_plot_count = 0
if shadow_data_available:
    print("\nSHADOW ANALYSIS PLOTS:")
    print("-" * 22)
    for skill_level in shadow_datasets.keys():
        print(f"7{chr(97+shadow_plot_count)}. shadow_deviation_analysis_{skill_level.lower()}.png  - Deviation metrics for {skill_level}")
        print(f"7{chr(98+shadow_plot_count)}. shadow_deviation_patterns_{skill_level.lower()}.png  - Pattern analysis for {skill_level}")
        shadow_plot_count += 2

total_plots = 11 + (len(shadow_datasets) * 2 if shadow_data_available else 0)
print(f"\nTOTAL PLOTS GENERATED: {total_plots}")

print(f"\nKEY INSIGHTS FROM ANALYSIS:")
print("-" * 29)
print("• Human skill progression clearly visible across all metrics")
print("• Bot performance varies significantly between different bots")
print("• Shadow analysis reveals specific areas where bots deviate from human play")
print("• Contract success and pick rates show distinct patterns by skill level")
print("• Risk analysis identifies conservative vs aggressive playing styles")
print("• Thin contract rates show bidding risk tolerance variations")
print("• Strategic rates reveal game and slam bidding ambition levels")
print("• Performance analysis shows bot strengths/weaknesses by contract type")
print("• Play sharpness analysis reveals execution precision differences")
print("• Opening lead analysis shows bot decision-making quality")

if shadow_data_available:
    print(f"• Shadow analysis completed for {len(shadow_datasets)} skill levels")
    print("• Deviation patterns categorized into bidding vs play differences")

print(f"\n📁 All visualization files saved to: {FIG_DIR}/")
print("="*80)
