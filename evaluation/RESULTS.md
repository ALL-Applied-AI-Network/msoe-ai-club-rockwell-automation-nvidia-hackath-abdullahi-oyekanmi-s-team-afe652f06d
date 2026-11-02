# Measured results - Dataset 1 final vial inspection

Model: `meta/llama-3.2-90b-vision-instruct` on build.nvidia.com (zero-shot, no training).
Images: real overhead camera frames from the CSI line, 2026-09-23 (Dataset 1 download).
Labels: by eye from the 640x480 frames, one labeller, made before seeing any system output on those frames.
Heavily shifted caps that could not be judged by eye (G13, G14, G18 in later frames) were left out of scoring.
S01-S05 are half hidden at the plate edge and reported as NO VIEW, so they are excluded.

## Held-out TEST set (6 frames, never used for any tuning)
`cap_seal_20260923_195648, 200050, 200315, 200716, 201029, 201206`
321 scored positions: 69 good vials, 247 empty holes, 6 defects (3 missing caps, 3 crooked caps).

| | Defects caught | Good vials wrongly flagged | Empty holes wrongly flagged |
|---|---|---|---|
| Baseline: NVIDIA model alone (raw answer per position) | 1 / 6 | 9 / 69 | 13 / 247 |
| **This system** (model + physical rules + second look) | **6 / 6** | **0 / 69** | **13 / 247** (REVIEW, none rejected) |

- Missed defects: **0**. False alarm rate at the setting that catches every known defect:
  13 / 316 non-defect positions (4.1 %), all of them empty holes sent to human REVIEW, none auto-rejected.
- Per-position output: `test_results.csv`. Labels: `test_labels.csv`.

### Second reading of two positions (team review, 01:58 Nov 1)
The official label definition says "crooked_cap: the cap is visibly tilted or not seated. A leaning vial by
itself is not enough." On review, G16 and G19 look like **leaning vials with seated caps**, not crooked caps.
Scored that way (3 defects, all missing caps; 72 good vials):

| | Defects caught | Good vials wrongly flagged | Empty holes wrongly flagged |
|---|---|---|---|
| Baseline: NVIDIA model alone | 0 / 3 | 10 / 72 | 13 / 247 |
| **This system** | **3 / 3** | **3 / 72** (G16 x2 REVIEW, G19 x1 REJECT as "missing cap") | 13 / 247 |

Either way: no missed defects. Leaning vials are flagged for a human, and one (G19 at 20:12:06) was wrongly
auto-rejected - the model called it a missing cap and the cap covered 83 % of the opening, just under the 90 %
second-look threshold. Thresholds were NOT changed after seeing this, to keep the test honest.

## With the robot's placement count (recommended on the dataset page)
The model cannot tell a bright empty hole from an uncapped glass vial. The robot/PLC knows how many vials it
has placed, which settles it: a no-cap hole beyond the count is waiting (EMPTY); a no-cap hole within the
count is a defect (missing cap or missed placement). Counts were read off each frame (what the PLC would
report) and are listed in `test_images_with_counts.txt`. Scored with the same saved model answers:

| Set | Defects caught | Good vials flagged | Empty holes flagged |
|---|---|---|---|
| Held-out test, no count | 6 / 6 | 0 / 69 | 13 / 247 |
| **Held-out test, with count** | **6 / 6** | **0 / 69** | **0 / 247** |
| Held-out test, with count, G16/G19 as leaning vials | 3 / 3 | 3 / 72 | 0 / 247 |
| Extra (another day + 2 empty fixtures), no count | - | 1 / 6 | 35 / 157 |
| **Extra, with count** | - | **1 / 6** | **0 / 157** |

The extra set also contains one missed placement (empty G01 while G02-G08 are filled, 2026-09-22 19:27:45):
flagged in both modes. A vial with a dark grey cap (G03, not one of the 5 labels) goes to REVIEW.

## Fill estimated from the image (no count typed) - default in v15
The last position (in fill order) with a seated lemon-yellow cap marks how far the robot has got. No-cap holes more
than 2 positions past it -> EMPTY; no-cap holes before it -> REJECT (missing cap or missed placement); the 2 positions
right after it -> REVIEW. The margin (2) was chosen on the DEV frames (margin 1 missed G17 at 20:08:53, where the
two newest vials were both defective).

| Set | Defects caught | Good vials flagged | Empty holes flagged |
|---|---|---|---|
| Dev | 8 / 9 (same as before) | 0 / 72 | 4 / 240 (was 9) |
| **Held-out test** | **6 / 6** | **0 / 69** | **4 / 247** (was 13) |
| Extra (other day + empty plates) | missed placement flagged | 1 / 6 | 6 / 157 (was 35) |

Limit: if 3 or more vials in a row at the very end of the fill are uncapped, the 3rd+ would be called EMPTY.
The robot's real placement count removes that risk.

## Change-aware inspection (1,097 real frames, 2026-09-23)
Each position's picture is compared (lighting-normalised) with the last time it was inspected; only changed
positions go to the model. On labelled consecutive frames every real change scored >= 0.35 and every unchanged
position <= 0.23; the threshold is 0.15. Over the whole day's sequence: 16,446 model calls if every uncertain
position is asked on every frame vs 248 change-aware (98.5 % fewer). Reuse is only allowed between frames of the
same run (<= 10 minutes apart).

## Development set (6 other frames, used to find and fix error patterns)
`cap_seal_20260923_195825, 200138, 200403, 200853, 201117, 201254`
- First run (before fixes): 8/9 defects caught, 22 false alarms incl. 8 good vials REJECTED.
- After fixes: 8/9 defects caught, 9 false alarms, 0 good vials flagged. One crooked cap (201117 G19) was passed.

## What changed between the runs (all rule-based or prompt-based, no training)
1. "Missing cap" while a yellow cap covers the vial opening is physically impossible -> ask the model a
   second, pointed question (seated or crooked?). This corrected 16/16 such answers to pass.
2. A model "pass" on a hole with no cap is not counted as a filled position (removed false GAP alarms).
3. Tried and rejected: a "glass vial or empty hole?" second question. On the dev set it said "vial" for
   3/3 open vials but also for 7/8 empty holes, so it is switched off (USE_VIAL_CHECK = False).

## Per class, held-out frames only (v32)
Held-out real frames only (never tuned on), labels as written; "caught" = REJECT, REVIEW or GAP.
Reproduce: `python evaluation/perclass.py <labels+images folder> vlm_cache.json`.

| Class | NVIDIA model + rules (6 frames) | Camera checks only (22 frames) |
|---|---|---|
| missing_cap caught | 3/3 | 7/7 |
| crooked_cap caught | 3/3 | 4/7 |
| missed placement caught (GAP) | - | 1/1 |
| damaged_vial | no real example | no real example |
| good vials flagged (false alarms) | 0/69 | 1/134 |
| empty holes flagged, robot count | 0/247 | 0/1046 |
| empty holes flagged, fill from image | 4/247 | 32/1046 |

Missed defects (false negatives): local: G19 (one leaning vial, labelled crooked_cap by eye) in 3 frames; model mode flags it.

Locked round 2 (13 frames, camera checks only): good flagged 0/59, defects flagged 6/8 (the 2 unflagged are the leaning G19).

## Limitations
- Test and dev frames come from the same fill sequence on one day; a different day / lighting may differ.
- The model cannot reliably tell a bright empty hole from an uncapped glass vial. Without a placement count
  those positions go to REVIEW (about 2 per image while filling, about 13 on an empty plate).
- Subtly crooked caps depend on the model (one was missed on the dev set).
- Leaning vials (seated cap, tilted body) are often flagged; the model sometimes calls them missing caps.
- About 20-35 s of model time per frame with 3 parallel requests on the free API; a local NIM would be faster.
- No damaged_vial example in the real data; 14 held-out real defects in total (3 are one leaning vial, G19).
- Not yet run live on the CSI line.

## Digital twin stress test (NVIDIA Isaac Sim, 60 synthetic frames, 1,800 G positions, exact labels) - v16
Local checks + fill order, no vision model. "Slipped through" = a defect ended as EMPTY.

| | before (v15) | after (v16) |
|---|---|---|
| Good caps recognised | 907/950 | 950/950 |
| ...under dim light (light < 0.8x) | 338/378 | 378/378 |
| Fill level read from the image exact | 42/60 frames | 55/60 frames |
| Defects slipped through, fill estimated from image | 9/286 | 1/286 |
| Defects slipped through, robot count given | 0/286 | 0/286 |
| Missed placements flagged, robot count given | 59/59 | 59/59 |

The v16 change (brightness floor follows the light) changed 0 of 17,820 real positions (297 real frames, including
every labelled DEV/TEST/EXTRA frame), so the real-frame numbers above are unchanged.

Sim-to-real: a classifier trained only on synthetic crops caught 15/15 real defects (DEV+TEST) but flagged 141/141
real good vials, so it is not used for decisions.

Leaning vs tilted (digital twin): of 128 synthetic crooked caps (tilted 12-40 degrees on an upright vial), 55 look
like a whole, round, normal-size cap from above (24/29 at 12-20 degrees, 25/45 at 20-30, 6/54 at 30-40). So a round
off-centre cap cannot be auto-passed as a leaning vial; it stays REVIEW. Real leaning vials G16/G19 stay REVIEW (never
PASS, never missed).

## Leaning vials (digital twin, 150 frames, 506 good leaning vials, 355 crooked caps) - v18
Local checks + robot count. Before -> after the v18 "own whole cap" rule:

| | before | after |
|---|---|---|
| Leaning (good) vials rejected as missing cap | 66 | 20 |
| Leaning vials as one-click operator checks | - | 162 |
| Crooked caps passed | 0 | 0 |
| Missing caps / missed placements passed | 0 | 0 |

Real labelled frames (DEV/TEST/EXTRA, local + count and local + estimate): 0 status changes. Quick checks offered on
real frames: G16 (leaning vial, 5 frames) and G07 in 192745 (a good cap seen off-centre) - nothing defective.
Shape-only auto-pass (3-fold by frame on the twin): 110/221 leaning vials cleared, 3/152 crooked caps wrongly
cleared -> not used.

## Learning loop (learn_loop.py) - v19
Twin-only model, fixed held-out twin test set (47 frames of the camera-shift batch). Real frames scored, never trained on.
See digital_twin/learning_curve.json. Twin false alarms on good vials 150/753 -> 13/753 over 4 rounds; leaning vials
misread 147/188 -> 12/188. Real: good vials flagged 123/141 -> 106/141, real defects flagged 15/15 -> 11/15. Not used
for decisions (sim-to-real gap).

## Round 2 real frames (v20) - 15 practice + 13 locked, labelled before any system output
Local checks only, fill estimated from the image (with robot count in brackets).
- Locked: good vials flagged 0/59; defects flagged 6/8 (unflagged: G19 leaning vial x2); empty holes to a person 22/642 (0/642).
- Practice: good vials flagged 0/25; missed placement G01 flagged; empty holes to a person 30/795 (0/795).
Change made after looking at the practice set: none to the decisions. The "next after last cap" reviews became
one-click operator checks (65 such empty holes across all labelled real frames; the 2 real uncapped vials in that
spot - G20 - also land there, so the margin stays).

## Reasoning audit (v23)
Saved NVIDIA answers re-scored with the corrected rules (no new API calls):
- DEV, TEST (with/without robot count): good vials rejected 0 (was 0); with G16/G19 counted as leaning (official
  rule) good vials REJECTED 1 -> 0 on each set (G19 now REVIEW); real defects flagged unchanged.
- Model answers on hard positions: G19 leaning -> "missing_cap" (1.0); G17/G20 open vials -> "pass". The decision
  rules, not the model's words, are what keep these safe.

