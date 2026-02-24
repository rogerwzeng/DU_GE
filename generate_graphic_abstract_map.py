#!/usr/bin/env python3
"""
Generate Graphic Abstract Map - Clean, publication-ready US map with MSA quadrants
Optimized for visual impact and clarity at small sizes
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# Set professional style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# Get home directory for generic paths
HOME = Path.home()
BASE_DIR = HOME / 'DissipativeUrbanism'

# Load MSA data with coordinates
df = pd.read_csv(BASE_DIR / 'geodesic_efficiency/results/msa_data_with_coords.csv')
print(f"Loaded {len(df)} MSAs with coordinates")

# Define thresholds
sigma_threshold = 30
eta_threshold = 0.7

# Classify MSAs into quadrants
def classify_quadrant(row):
    if row['mean_entropy_production'] >= sigma_threshold and row['geodesic_efficiency'] >= eta_threshold:
        return 'I'  # Dissipative
    elif row['mean_entropy_production'] < sigma_threshold and row['geodesic_efficiency'] >= eta_threshold:
        return 'II'  # Stable
    elif row['mean_entropy_production'] >= sigma_threshold and row['geodesic_efficiency'] < eta_threshold:
        return 'III'  # Forced
    else:
        return 'IV'  # Stagnant

df['quadrant'] = df.apply(classify_quadrant, axis=1)

# Count by quadrant
for q in ['I', 'II', 'III', 'IV']:
    count = (df['quadrant'] == q).sum()
    pct = count / len(df) * 100
    print(f"  Quadrant {q}: {count} MSAs ({pct:.1f}%)")

# Vibrant color scheme for graphic abstract
colors = {
    'I': '#228B22',   # Forest Green - Dissipative
    'II': '#1E90FF',  # Dodger Blue - Stable
    'III': '#DC143C', # Crimson - Forced
    'IV': '#808080'   # Gray - Stagnant
}

# Filter to continental US
continental = df[(df['lat'] >= 24) & (df['lat'] <= 50) & 
                 (df['lon'] >= -125) & (df['lon'] <= -66)].copy()

print(f"\nContinental US: {len(continental)} MSAs")

# Create figure - optimized for graphic abstract (wider aspect ratio)
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
    central_latitude=37.5, 
    central_longitude=-96,
    standard_parallels=(29.5, 45.5)
))

# Set extent
ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

# Add land (clean white background)
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='50m',
    facecolor='#fafafa',
    edgecolor='none'
)
ax.add_feature(land, zorder=0)

# Add ocean (subtle blue)
ocean = cfeature.NaturalEarthFeature(
    category='physical',
    name='ocean',
    scale='50m',
    facecolor='#e8f4f8',
    edgecolor='none'
)
ax.add_feature(ocean, zorder=0)

# Add Great Lakes
lakes = cfeature.NaturalEarthFeature(
    category='physical',
    name='lakes',
    scale='50m',
    facecolor='#e8f4f8',
    edgecolor='#87ceeb',
    linewidth=0.5
)
ax.add_feature(lakes, zorder=1)

# Add state boundaries (cleaner, lighter)
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none',
    edgecolor='#aaaaaa',
    linewidth=0.6,
    linestyle='-'
)
ax.add_feature(states_provinces, zorder=2)

# Add US country boundary (bold)
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none',
    edgecolor='#333333',
    linewidth=2.0
)
ax.add_feature(countries, zorder=3)

# Plot MSAs with larger markers for visibility
# Order: IV (gray) first, then II (blue), I (green) last for visual hierarchy
for quadrant in ['IV', 'III', 'II', 'I']:
    subset = continental[continental['quadrant'] == quadrant]
    if len(subset) > 0:
        ax.scatter(subset['lon'], subset['lat'], 
                  c=colors[quadrant], 
                  s=80,  # Larger markers
                  alpha=0.9,
                  edgecolors='white',
                  linewidth=1.2,
                  zorder=5,
                  transform=ccrs.PlateCarree())

# Add title
ax.set_title('Predicting Resilience of U.S. Metropolitan Areas', 
             fontsize=24, fontweight='bold', pad=15)

# Create legend with larger fonts
legend_elements = [
    mpatches.Patch(facecolor=colors['I'], edgecolor='white', 
                   label=f"I: Dissipative (n={(continental.quadrant=='I').sum()})"),
    mpatches.Patch(facecolor=colors['II'], edgecolor='white', 
                   label=f"II: Stable (n={(continental.quadrant=='II').sum()})"),
    mpatches.Patch(facecolor=colors['III'], edgecolor='white', 
                   label=f"III: Forced (n={(continental.quadrant=='III').sum()})"),
    mpatches.Patch(facecolor=colors['IV'], edgecolor='white', 
                   label=f"IV: Stagnant (n={(continental.quadrant=='IV').sum()})")
]
legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=12, 
                   framealpha=0.95, title='Quadrant Classification', title_fontsize=13,
                   borderpad=1.0)
legend.get_frame().set_linewidth(1.5)

# Add note
excluded = len(df) - len(continental)
ax.text(0.98, 0.02, f'N = {len(continental)} MSAs\n({excluded} in AK, HI, PR excluded)', 
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        style='italic', alpha=0.8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Minimal gridlines
ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, 
             linewidth=0.5, color='gray', alpha=0.2, linestyle='--')

plt.tight_layout()

# Save high-resolution outputs
plt.savefig(BASE_DIR / 'geodesic_efficiency/figures/graphic_abstract_clean.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(BASE_DIR / 'geodesic_efficiency/figures/graphic_abstract_clean.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

# Also save the main graphic_abstract file
plt.savefig(BASE_DIR / 'geodesic_efficiency/figures/graphic_abstract.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(BASE_DIR / 'geodesic_efficiency/figures/graphic_abstract.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

plt.close()
print("\nGraphic abstract map saved!")
print("  - figures/graphic_abstract_clean.png (optimized version)")
print("  - figures/graphic_abstract.png (updated)")
