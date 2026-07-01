---
name: cameras-color-day-ir-night
description: "stupid-cat cameras are colour by day, infrared (grayscale) by night — drives identity design"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f22c11-62f1-444b-9bb5-b950376a79ba
---

The user's two litter-box cameras output **colour video in daytime but switch to infrared (grayscale) at night**. Confirmed by the user ("用的红外") and by the exported daytime snapshot frames being colour.

**Why it matters:** any cat-identification (Re-ID) scheme that relies on colour (coat colour/histograms) works only in daylight and fails at night. Identity must be day/night-robust.

**How it's handled** (merged in [reid-day-night-identity-design](reid-day-night-identity-design.md)): embed on grayscale/luminance so day and night share one feature space; use colour only as a daytime-only bonus that auto-disables when a crop is ~grayscale.
