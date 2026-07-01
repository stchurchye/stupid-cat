---
name: reid-day-night-identity-design
description: stupid-cat cat-identity stack — grayscale gallery + daytime colour bonus + optional DINOv2
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f22c11-62f1-444b-9bb5-b950376a79ba
---

The cat-identification (Re-ID) approach, chosen with the user and shipped in PR #7 (merged 2026-06-15):

- **Grayscale-domain embedding** (`inference.reid_grayscale`, default on): crops embed on luminance so daytime-colour and night-IR map to one feature space → one gallery matches day or night. See [[cameras-color-day-ir-night]].
- **Multi-vector gallery + top-k** (`inference.reid_topk`): each cat keeps every usable ref embedding (`gallery.npy`), scored by mean of top-k cosine — pose-robust vs a single mean centroid.
- **Daytime colour bonus** (`inference.color_*`): hue/sat histogram (`color_gallery.npy`), only for colourful refs, only computed when a visit crop is colourful. The **embedding score is the threshold gate** (day & night identical); colour only re-ranks cats that already qualify.
- **Backbone**: `efficientnet_b0` default (zero download); `dinov2_vits14`/`vitb14` optional (stronger individual ID, first-run downloads ~85MB). **Switching backbone requires rebuilding every cat's refs** — stored embeddings of a different dim are skipped at match time.

**The 5 cats** (real ids/names): guagua 瓜瓜 (orange&white, distinctive), shoulang 寿郎 (orange long-hair), juanjuan 卷卷 / xuemei 雪梅 / leilei 雷雷 (all white/pale).

**Validated on real data (2026-06-17 export: 44 visits, user-labelled ~31 single-cat).** Honest **leave-one-VISIT-out** accuracy (the production metric; the earlier ~0.9 was leave-one-CROP-out, optimistic because same-visit crops are near-duplicates):
- EfficientNet gray+colour ≈ 0.54; **DINOv2 gray+colour ≈ 0.58** overall.
- Per-cat: guagua 90% (great), juanjuan ~71%, xuemei ~50%, shoulang ~20% (confused with guagua — both orange), **leilei ~0% (white, unidentifiable)**.
- Root cause is physical, not code: top-down + night-IR + look-alike cats; 3 white cats are one blob from above. DINOv2/colour/more-data only help marginally.

**Decision (2026-06-17): vision-only is not reliable → use distinct visual markers** (different-coloured collars/tags, NOT RFID) on at least the white trio + the two orange cats. Then re-collect a little data and rebuild galleries → expected 90%+. The existing daytime colour bonus already exploits collar colour; may need to raise `color_weight`/lower the gate once marker data exists so colour can decide between same-shaped cats. Next step waits on the user markering the cats + a fresh export.
