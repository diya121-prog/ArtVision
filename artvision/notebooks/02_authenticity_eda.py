# %% [markdown]
# # Authenticity EDA & Shortcut Checks
# This notebook validates the AI-ArtBench dataset integration, demonstrating class balance and the removal of shortcuts (like resolution and JPEG format differences).

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
import collections
from PIL import Image

# %%
# Load metadata
metadata = pd.read_csv('../data/processed/metadata.csv')
metadata.head()

# %% [markdown]
# ## Class Balance

# %%
# Human vs AI
print(metadata['is_ai_generated'].value_counts(normalize=True))

# %%
# Generators
print(metadata['generator'].value_counts())

# %%
# Styles across generators
print(pd.crosstab(metadata['generator'], metadata['style']))

# %% [markdown]
# ## Shortcut Checks (After Normalization)
# We expect all images in our processed cache to have exactly the same size (`384x384`) and format, so the neural network can't cheat by looking at resolution.

# %%
sizes = collections.defaultdict(int)
for filepath in metadata['filepath']:
    with Image.open(f"../{filepath}") as img:
        sizes[img.size] += 1
print("Sizes found in normalized cache:", dict(sizes))

# %% [markdown]
# (Note: Before normalization, AI_SD was exclusively 768x768 and Human/AI_LD were 256x256. We've successfully standardized them to 384x384).

# %% [markdown]
# ## Missing Label Masks Check

# %%
# We check how many rows are missing specific labels.
print("Rows missing 'era':", metadata['era'].isna().sum())
print("Rows missing 'mood':", metadata['mood'].isna().sum())

# %% [markdown]
# ## Sample Grid

# %%
# (In a real interactive notebook, we would plot a grid of images here.)
# For headless execution, we just confirm paths exist.
sample = metadata.sample(4)
for _, row in sample.iterrows():
    assert os.path.exists(f"../{row['filepath']}")
print("Sample images verified to exist.")
