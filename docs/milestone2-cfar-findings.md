# Milestone 2: CFAR detection, findings

Scene: `S1A_IW_20250508T094925` (Sentinel-1A IW, HH/HV, HyP3 RTC gamma0, 20 m, UTM 21N), NE Newfoundland, 8 May 2025.

**Result:** CA-CFAR on HV finds **126 candidate targets** at open sea. Visual review shows mostly compact point targets (icebergs or ships). The remaining false alarms are mainly loose sea-ice fragments near the pack ice, at the low-contrast end. Open-sea clutter false alarms are rare.

> **Since this write-up:** the pipeline now runs on **10 m** RTC and adds an **HH co-pol check** (step 4 of the README method). At 10 m the oversampled HV speckle passes the 2 px minimum, and the HH check removes it. The numbers below are from the original 20 m run. For multi-date results and per-berg recall against IIP labels, see [validation-findings.md](validation-findings.md).

## Method

| Step | What | Key parameters (`config.yaml`) |
|---|---|---|
| 1. CA-CFAR | Pixel vs. mean of a background ring, on linear intensity; masked pixels excluded | `guard_m: 100`, `background_m: 400` (41×41 minus 11×11 px at 20 m), `pfa: 1e-6` |
| 2. Threshold | Gamma speckle model: pixel / ring-mean ratio ~ F(2L, 2NL); ENL `L` estimated from the scene | `enl: auto`, `enl_cell_m: 500` |
| 3. Components | 8-connected, 2 px to 0.8 km²; drop if within 200 m of land, pack ice or nodata | `min_area_px: 2`, `max_area_m2: 800000`, `edge_buffer_m: 200` |
| 4. Strip rejection | Grow each target to the 3×3-smoothed region > 4 dB above background; reject if longer than 300 m | `grow_db: 4`, `max_structure_m: 300` |

Size parameters were originally in pixels. They are now in metres and converted per scene, so these 20 m results are unchanged.

Implementation: box filters (`scipy.ndimage.uniform_filter`), row blocks with a halo so the 10k × 14k scene fits in memory. About 2 minutes per scene per band.

## Findings

### 1. HV water sits at the noise floor
Water statistics measured on 500 m cells:

| Band | Water level | CV | ENL | CFAR threshold (pfa 1e-6) |
|---|---|---|---|---|
| HH | −20.3 dB | 0.45 | 5.0 | 4.7× background (6.7 dB) |
| HV | −31.7 dB | 1.14 | 0.79 | 16.7× background (12.2 dB) |

HV water is below the Sentinel-1 IW noise floor (NESZ, about −25 dB), so what remains after thermal-noise removal is spikier than single-look speckle (ENL < 1). Estimating ENL from the scene sets a correspondingly stricter threshold instead of assuming a textbook value.

### 2. HV is the right CFAR band; HH produces sea-clutter false alarms
| Band | Detections | Matched in the other band (≤ 3 px) |
|---|---|---|
| HV | 126 | 109 of 126 also detected in HH |
| HH | 1009 | only 106 match an HV detection |

The ~900 HH-only detections are mostly at far range (42–46° incidence), have ~9 dB contrast, and are dim in HV (median −24 dB). That is HH sea clutter, whose tail is heavier than the gamma model assumes. Targets detected in HV are bright in both bands: median HH peak −8 dB, about 12 dB above HH water.

### 3. Thin sea-ice strips were the main false-alarm source
Without step 4, CFAR gave 165 HV detections. Chip review showed that most of the weakest ones were the brightest 2–4 pixels on thin, curved bright filaments 1–2 km long: strips of loose ice too narrow for the Milestone 1 pack-ice mask.

Structure length (after growing) separates them:

| Group (hand-labelled from chips) | Structure length |
|---|---|
| Ice strips | 16–45 px (320–900 m), except one short knot at 9 px |
| Compact targets | 5–15 px (100–300 m) |

The grow level (2–4 dB) and smoothing (3×3 vs. 5×5) barely changed this. The 300 m cutoff removed 39 of 165 detections.

### 4. Final HV detection set (126 targets)

| Attribute | Min | Median | 90th pct | Max |
|---|---|---|---|---|
| Area (CFAR pixels) | 800 m² | 1,200 m² | 4,000 m² | 18,000 m² |
| Extent | 40 m | 40 m | 120 m | 200 m |
| Grown structure | 80 m | 160 m | 280 m | 300 m |
| Contrast (HV peak − background) | 12.5 dB | 15.0 dB | 19.0 dB | 24.9 dB |
| HH peak | −19.0 dB | −8.0 dB | −1.6 dB | +4.7 dB |
| Incidence angle | 33.3° | 42.3° | 44.5° | 45.5° |

Spatially, detections cluster in the western half of the scene, near the coast and around the pack-ice patches. That is consistent with icebergs drifting inshore in May.

Chip review of the final set (weakest 8, random 8, largest 4):
- Random and strongest detections: all compact point targets. The largest are 150–300 m blobs, plausibly icebergs.
- Weakest 8: about half are still ice-related (knots on strips, small floe clusters).

**Rough estimate:** ice-related false alarms are about 10–15% of the set, concentrated at contrast < 13 dB and near masked pack ice.

## Ground-truth comparison (NAIS iceberg chart)

Source: North American Ice Service (International Ice Patrol + Canadian Ice Service) Iceberg Analysis, valid 2025-05-08 00:00 UTC. That is 10 h before the scene. The chart note warns of "reduced iceberg reconnaissance", and the last reconnaissance was 6 May. It gives estimated icebergs per 1° square, and those counts include bergy bits and growlers too small for 20 m SAR. Counts were hand-transcribed from the GIF into [`validation/nais_20250508_counts.csv`](../validation/nais_20250508_counts.csv).

The chart count is scaled by the fraction of each square the masked scene actually sees (`expected`):

| Square (SW corner) | Chart count | Scene coverage | Expected | SAR detections |
|---|---|---|---|---|
| 50N 56W | 10 | 76% | 7.6 | 28 |
| 50N 55W | 9 | 95% | 8.6 | 27 |
| 50N 54W | 4 | 100% | 4.0 | 11 |
| 50N 53W | 5 | 63% | 3.2 | 3 |
| 51N 56W | 11 | 31% | 3.4 | 31 |
| 51N 55W | 2 | 36% | 0.7 | 17 |
| 51N 54W | 5 | 41% | 2.1 | 4 |
| 51N 53W | 7 | 25% | 1.8 | 2 |
| **Total** | | | **31.4** | **123** |

- **East of about 54.5°W, where the chart shows open water, counts agree.** SAR finds 20 detections against 11 expected, and the 53W squares match closely (3 vs 3.2, 2 vs 1.8).
- **West of about 54.5°W, SAR finds about 5× more than the chart** (103 vs 20). The chart's sea-ice limit runs through the scene at about 54.5°W, so the chart counts everything west of it as inside the sea ice. Only the dense core was removed by our pack-ice mask. The 88 detections west of 54.5°W are weaker on average: 31% have contrast < 14 dB, against 16% east of it. This fits loose ice floes being counted as targets.
- Other likely contributors to the western excess:
  - Nearshore fishing vessels (the snow-crab season runs in May).
  - The chart undercounting inside sea ice, where reconnaissance is sparse.

**Conclusion:** in open water the detector's counts are consistent with the chart. Inside the chart's sea-ice limit, a large share of detections are probably not icebergs. Masking to the chart's sea-ice limit, or adding `distance_to_ice_km` as a feature, should be the first fix.

Reproduce:
```powershell
python -m iceberg_sar.cli ground-truth 2025-05-08          # charts for 7-9 May -> data/ground_truth/
python -m iceberg_sar.cli validate data\raw\S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_FDE3 --counts validation\nais_20250508_counts.csv
```

## Limitations
- **CFAR area underestimates target size.** Only the brightest pixels pass a 12 dB threshold. `structure_m` is the better size proxy.
- **Large targets are rejected.** Bergs or ships longer than about 300 m are removed by the strip filter. They are rare off NL.
- **The CFAR model is a simplification.** Gamma speckle understates real sea-clutter tails. The realised false-alarm rate is higher than `pfa`, which is noticeable in HH.
- **Tuning is based on one scene.** Parameters were tuned on a single scene with moderate sea state and hand labels from about 40 chips. Other dates, especially rough seas, need checking.
- **Coarse validation only.** The ground truth is per 1° square (about 7,800 km²) and 10 h old, with reconnaissance from 2 days earlier. It can check counts, not individual detections.

## Next steps
1. Mask or flag detections west of the NAIS sea-ice limit. *Done:* each detection now has `distance_to_ice_km`, measured to the edge of the Milestone 1 pack-ice mask (which is already buffered by 1 km). On 8 May, 24 of 126 detections lie within 2 km of it, and chip review shows ice fragments are concentrated there. The viewer greys out and can hide candidates within `viewer.near_ice_km` (5 km).
2. Stretch: OS-CFAR or K-distribution CFAR for heavier-tailed clutter (`cfar.variant` is ready for it). *Not done.*
3. Run on 2–3 more spring scenes to check parameter stability. *Done:* 2025-05-02 and 05-14 at 10 m with NAIS counts, plus per-berg validation on 2019 scenes. See [validation-findings.md](validation-findings.md).

## Reproduce
```powershell
python -m iceberg_sar.cli detect data\raw\S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_FDE3
python -m iceberg_sar.cli detect data\raw\S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_FDE3 --band HH
```
Outputs in `data/outputs/detections/` (gitignored):
- `…_HV_detections.geojson`: the detection points.
- `…_HV_detections.png`: overlay quicklook.
- `…_HV_detections.json`: run summary.
- `…_HV_chips.png`: the chip review figure.
