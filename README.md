# Dataset 1 - Final Vial Inspection (CSI testbed) - v33

Locates all 60 fixture positions (G01-G30 good fixture, S01-S30 scrap fixture), checks each one,
and shows every problem as a clear crop with the reason it was flagged.

Files: `app.py` (dashboard), `vial_core.py` (inspection logic), `ui.py` (styling). Keep all three in one folder.
`isaac/` holds the NVIDIA Isaac Sim digital-twin generator, `digital_twin/` sample synthetic frames with exact labels.

## Quick start (any computer)
Team: Malik Bennani, Abdullahi Oyekanmi, Markos Lezama, Oluwatobi Odekunle - Rockwell Automation x NVIDIA hackathon.

1. Install **Python 3.10 or newer** from https://www.python.org/downloads/ (Windows: tick "Add python.exe to PATH").
2. Download this folder and unzip it.
3. Start it:
   - **Windows:** double-click `run_app.bat`
   - **Mac / Linux:** open a terminal in the folder and run `bash run_app.sh`
4. The first start installs what the app needs (1-3 minutes, needs internet). Then the website opens at
   http://localhost:8501 (open that address yourself if the browser does not open).

**No API key needed to try it:** pick "Local CV only" in the sidebar. With no camera image, open the 🧪 Digital
twin panel on the start page, or drop one of the synthetic Isaac Sim frames from `digital_twin/samples/` into the
uploader. For the NVIDIA model, get a free key at
https://build.nvidia.com and paste it in the sidebar (it is never saved into the project).

Camera images: the real CSI frames (Dataset 1) are lab data shared through the hackathon Slack, so they are not in
this download. Drop any overhead image of the two fixtures into the website; auto-fit handles other cameras.

Manual setup instead: `python -m pip install -r requirements.txt` then `python -m streamlit run app.py`.

## Highlights
- **Every position, not the whole picture:** 60 numbered positions per frame, each judged on its own crop.
- **Physical rules check the AI:** a PASS needs a yellow cap over the vial opening; impossible answers get a
  pointed second question; errors, refusals and disagreements go to a human - never to PASS.
- **Closes the loop with the cell:** each result becomes a line message (HOLD / CHECK / CONTINUE, per-position
  actions such as DIVERT_TO_SCRAP or REPLACE_MISSED_VIAL, and 30-bit PLC tag masks) - downloadable as JSON.
- **Uses the fill order:** the robot fills in a fixed order, so the last seated cap shows how far it has got. Holes well
  past it are waiting (EMPTY), holes before it must hold a capped vial, and the 2 right after it go to a human. This
  is estimated from the image automatically for the good plate; the robot's real placement count (if typed / sent)
  overrides it. The scrap plate is not estimated from caps, because scrap vials are often uncapped (a Cap_seal2 photo
  showed clear glass scrap vials being called EMPTY before this rule; labelled real frames are unchanged by it).
- **Change-aware:** across frames of the same run only positions that changed are re-inspected - measured
  98.5 % fewer model calls over 1,097 real frames (16,446 -> 248), with every labelled change re-inspected.
- **Self-correcting alignment:** one click re-fits the grid if the camera moves (<= 7.4 px error in 54 tests).
- **Traceability:** one-click HTML inspection record per image or per session (images, decisions, reasons,
  line actions). Download one from the website after an inspection.
- **Tested in the digital twin (NVIDIA Isaac Sim):** 1,800 automatically labelled positions rendered from the CSI
  twin's own camera. It exposed a dim-light weakness that the real frames never showed; fixed, with 0 decisions
  changed on 17,820 real positions. See "Digital twin" below and the 🧪 Digital twin tab.
- **Explainable:** every flagged vial shows its evidence - cap coverage, model answer, second look, deciding rule.

## What changed in v32 (after a judge-criteria review)
- **📈 Evidence and 🧪 Digital twin tabs restored.** Their contents had been lost since v25 when the review tab was
  added (the tabs were there but empty). Now with a per-class results table (held-out only), a card on whether a
  twin-only classifier transfers (it does not: 141/141 good vials flagged at the threshold that catches every real
  defect, so it is not in the decision), honest limits, and a Markdown results download for the report.
- **Failed model calls retried once automatically** when only a few failed (a single timeout used to leave a REVIEW
  with no answer). If many fail (API down) it does not retry, so an outage does not double the wait.
- **Scrap plate fix (v31):** the image fill estimate applies to the good plate only; scrap vials are often uncapped.

## Auto-fit: photos of any size, shape or framing (v26)
**Why it was needed.** The grid was stored for the 640x480 CSI camera and simply stretched to whatever photo was
given, the hole finder assumed a fixed hole size relative to the photo width, and Auto-align only searched +-45 px. A
photo from another camera (higher resolution, 16:9, wider or tighter view) put every circle in the wrong place.

**How it works** (`vial_core.auto_fit`):
1. Find round things at 14 sizes and keep the size where they form a fixture-like lattice: neighbours are the
   **same size**, about **3 radii apart along a row or column** (the plate's small holes between the big ones and
   diagonal pairs are rejected), and each one is **hole-like** - darker than the plate around it, or a yellow cap
   (round shapes in machinery or texture are not).
2. Group the lattice into plates and fit each plate's row/column steps.
3. Pick the 6x5 block that is the fixture: detected holes inside, hole-like spots inside, no hole-like spots just
   outside (the plate ends there - stops the grid sliding by a column), and the two plates' fixed layout (the scrap
   plate's position must agree with the good plate's, and either plate can place the other if one is mostly capped).
4. Straighten the photo into the standard 640x480 view (homography) and fine-tune with the existing Auto-align.
5. **Self-check:** several candidate fits (both plates, good plate only, scrap plate only, top block choices, and a
   plain resize) are tried; the one that puts the most grid circles on real holes and the fewest just outside the
   plates wins - a wrong fit cannot beat a plain resize.
Rectifying (instead of re-scaling every threshold) keeps every rule, crop size and model crop exactly as validated.

**Tested** (`evaluation/autofit_test.py`): 16 real labelled frames x 6 camera changes. Decisions identical to the
original frame on **96.4-98.0 %** of positions in every variant; the only defect label that changed was the leaning
vial G16 (now seen as capped - correct by the official rule). Standard 640x480 frames are untouched (0 of 394 checked
trigger auto-fit), so all earlier results stand. Your 707x403 screenshot from a different camera: both plates found,
grids on the holes. While testing, auto-fit also exposed a fill-estimate weakness (a leaning vial's cap did not count as
"filled", so an uncapped vial right after it could be called empty) - fixed: a vial's own whole cap beside the hole
now counts as filled.

### v27: second-look consistency fix
On a photo from a new camera, 5 seated caps (G08, G09, G10, G19, G20) went to review: the model's first answer was
"missing cap", the camera saw a cap over 48-86 % of the opening, and the model's **second look said "pass"** - but a
second-look pass was only accepted from 90 % coverage, while a first-look pass needs 45 %. Now both use the same 45 %
rule. Re-scored on the labelled real frames: no change (all defects still flagged, 0 good vials flagged).

## 💬 Vial Helper - ask the website (v28)
Top of the sidebar, on every page. Ask how to do something ("how do I align the circles?"), what something means
("what does GAP mean?", "what is a crooked cap?"), or about a position in the **current inspection** ("why is G15
flagged?", "what was flagged in this image?").

**Knowledge:** 130 checked topics in `helpbot_kb.py` - getting started, photos, grid & auto-fit, inspecting, every
status and official label, how decisions are made (pass rule, cap coverage, second look, fill estimate, strict mode,
change-aware), one-click checks and the review workstation, robot count, line message / PLC masks, reports, accuracy
and test protocol, the NVIDIA model and API key, the Isaac Sim digital twin and learning loop, lab rules, the project's
value, troubleshooting and practical questions (GPU, Mac, phone, privacy, submission).

**Built to be trustworthy:**
- Every answer was written against the real button / tab / setting names; numbers are read live from
  `vial_core.MEASURED` and the rule constants, so they cannot go out of date.
- Position questions ("why is G15 flagged?", "is G16 ok?") are answered from the actual inspection; "which one is
  G01?" gets the numbering topic instead.
- A question about the system always gets the best answer plus "Not what you meant?" buttons; only off-topic
  questions are declined (a single everyday word like "color" or "time" is not enough to answer).
- Chat feel: bubbles, a typing indicator, word-by-word replies. With an API key the NVIDIA text model
  (`meta/llama-3.1-8b-instruct`) writes the reply from the checked answers only; invented numbers, refusals or API
  errors fall back to the checked answer.
- Pasted API keys are spotted and the user is warned; invalid positions (G45) are explained.

**Tested** (`python evaluation/helper_tests.py`): 127 tuning questions - 113 direct, 14 offered as the first button,
0 wrong. **50 locked questions, first score: 33 direct, 44 answered or offered, 6 wrong, 4/6 off-topic declined**;
the 6 misses were real gaps (e.g. "difference between G and S" went to the modes topic) and were fixed afterwards
(now 40 direct, 50 reachable, 13/13 off-topic declined - optimistic, since the set was then seen). 15 edge cases
(position questions with/without an inspection, numbering vs position, invalid positions, pasted keys, empty and
very long input, off-topic) all pass, and the model-wording guard falls back on invented numbers, refusals, HTTP
errors and network failures.

## Lab rule compliance
The lab rules say the real line data is "there to look at, not to train on". This system trains
nothing on real data: the vision model is used zero-shot (prompt only), and the local checks are fixed,
physically motivated rules (caps are lemon-yellow, an empty hole is dark and uniform, the robot arm is a
yellow blob far bigger than a cap). Real frames were used only to look at and to test.

## How it works
1. **Geometry** - 4 calibrated corner holes per fixture -> 30 centers each (6 columns x 5 rows).
   Numbering follows the robot fill order: rightmost column top-to-bottom is 01-05, then one column left.
   Checked by eye on 16 camera frames from two days.
   **Auto-align** (button in step 1): finds the round holes in the image (Hough circles) and re-fits both grids.
   Holes vote for the best shift at small rotations/scales; holes found one pitch OUTSIDE the grid (the plate
   edge) are penalised so the grid cannot slide by a whole hole spacing. Tested on 6 real frames moved by known
   amounts (up to 40 px and 3 degrees, 54 cases): every grid recovered to within 7.4 px (a hole radius is ~10.5 px),
   median 4.4 px; a correct grid is left in place. ~0.15 s per image.
2. **Robot-arm check** - positions under the arm are reported **NO VIEW**, not guessed.
3. **Local rules** - cap colour must be bright lemon-yellow (glass seen down an open vial is orange-brown and
   must not count as a cap); only clearly dark, uniform holes are treated as empty and skip the model.
4. **Vision model (zero-shot)** - NVIDIA NIM (default `meta/llama-3.2-90b-vision-instruct`) labels each other
   position: pass, missing_cap, crooked_cap, damaged_vial, no_vial. The prompt explains camera perspective
   and asks two self-check questions (opening_visible, cap_tilted).
5. **Decision rules (a missed defect is worse than a false alarm)**
   - Hard safety rule: a PASS needs a yellow cap covering the vial opening. Otherwise -> REVIEW.
   - API error, timeout, refusal, invalid JSON, or local/model disagreement -> **REVIEW**, never PASS.
   - "crooked" on a cap that fully covers the opening -> REVIEW (often just perspective).
   - S01-S05 are half hidden at the plate edge -> **NO VIEW**.
   - Placement count ("Good placed / Scrap placed" per image): a hole whose CENTRE has no cap is EMPTY (waiting)
     beyond the count, and a REJECT (missing cap or missed placement) within it. Judged on the hole centre only,
     because a neighbour's cap can leak yellow into the edge of the crop (dev frames: good vials >= 62 % centre
     coverage, open vials 0 %). Leave blank if unknown - then those positions stay REVIEW.
   - Fill pattern: if you enter the placement count (sidebar), any empty position within that count is a
     **GAP** (missed placement) and later empties are waiting. Without a count, an empty before the last
     filled position is a GAP.
6. **Check accuracy tab** - pick the true label for each position; the app reports defects caught vs missed,
   false alarms, and a confusion matrix.

## Digital twin (NVIDIA Isaac Sim)
The lab rules say teams generate training data with the digital twins. We used NVIDIA Isaac Sim 6.0 with the CSI
digital twin (`tray_inspection.usdc`) to make synthetic CapSealCam frames whose labels are exact by construction.

**How the frames are made (`isaac/gen_synth.py`, run with `isaac/run_synth.bat`)**
- Renders the twin's own `CapSealCam` (RTX ray tracing, 768x576 with overscan), then applies the real camera's lens
  model (k1 = -0.1527, same maths as the twin's `tools/apply_lens.py`) -> 640x480, like the real camera.
- For every frame: a random robot fill level (vials 01..n in fill order), and among placed vials random missing caps
  (cap removed), crooked caps (cap tipped 12-40 degrees about its rim, slid off-centre) and missed placements. Both
  station lights are scaled 0.55-1.45x and warm/cool tinted.
- All edits go to a USD session layer, so the twin's files are never changed. Each frame has its own random seed, and
  the script restarts itself after a driver crash, so a run is exactly reproducible.
- Ground truth goes to `labels.json` (per position: pass / missing_cap / crooked_cap / no_vial, fill level, light).

**What we used it for**
1. *Geometry check* - projecting the twin's vial positions through its CapSealCam calibration lands every hole within
   2.7 px (median) of our hand-calibrated grid (`isaac/twin_inventory.py`).
2. *Stress test of the rules* - 60 frames, 1,800 G positions (950 good, 141 missing caps, 145 crooked caps,
   59 missed placements): `python evaluation/twin_safety.py synth.zip`.
3. *It found a weakness.* With the light ~40 % lower, good caps fell below our fixed "bright yellow" rule (V >= 170):
   only 338/378 good caps were recognised under dim light, the fill level read from the image was exact on only 42/60
   frames, and 9 defects could slip through as "empty" when the fill level was estimated from the image.
   **Fix:** the brightness floor now follows the light (measured from the metal plate around each hole: normally 170,
   down to 120 in a dim scene). After the fix: 378/378 dim good caps, 950/950 overall, fill level exact on 55/60
   frames, 1 defect could slip through without a count, and 0 with the robot's count. On the real camera the plate is
   bright, so the rule is unchanged: **0 of 17,820 real positions (297 frames) changed decision**, so all real-frame
   results below still hold.
4. *Sim-to-real, honestly.* A classifier trained only on synthetic crops (`sim2real.py`, with brightness, noise, blur
   and JPEG augmentation) caught 15/15 real defects but also flagged 141/141 real good vials: the twin's plate, glass
   and lighting look different from the real camera. So the twin is used to test and tune rules, not to make the
   final call. The twin also showed that an uncapped vial can look exactly like an empty hole from above; the only
   check that catches every one is the robot's placement count, which is why the app takes it.

5. *Leaning vials (common on the line, OK by the official rule).* A second twin run (150 frames, `--n 150 --seed 11`)
   tips 1 in 4 good vials 3-12 degrees in their hole with the cap seated. Of 506 leaning vials, 221 went to review and
   66 were even **rejected as "missing cap"** (cap fully off the hole, so the placement-count rule fired).
   **Fix:** each position now records its own cap blob (size in hole areas, distance from the hole). If one whole cap
   (0.8-1.2 hole areas) sits within 1.8 hole radii, it is "leaning vial or tipped cap?", never a blind REJECT, and
   the dashboard offers a one-click **✔ Just leaning / ✖ Cap tipped** for the operator. Result on the twin: rejected
   leaning vials 66 -> 20, 162 become one-click checks, and no crooked or missing cap passes. Two merged caps (an open
   vial beside a neighbour's cap, like real G17, area ~1.4) are never offered as a quick check. 0 decisions changed on
   the real labelled frames; the real leaning vial G16 becomes a one-click check.
   *Why not pass them automatically?* From straight above, 55 of 128 tipped caps look like neat round caps. A shape rule
   learned on the twin cleared 110 of 221 leaning vials on held-out frames but let 3 of 152 crooked caps through, so a
   person confirms. A side camera would settle it.

6. *Learning from its mistakes (`learn_loop.py`).* A twin-only model (gradient-boosted trees on crop features,
   trained with camera-shift / lighting / noise / JPEG augmentation) goes through rounds: train on all synthetic
   frames so far -> test on a fixed held-out twin set (47 frames, never trained on) -> list the mistakes by scenario ->
   write the next Isaac Sim settings that render more of them (`focus_next.txt`, e.g. `--tilt-min 10 --tilt-max 18
   --p-lean 0.45`). Mistakes from the last round count 3x in the next.

   | round | twin: defects caught | twin: good vials flagged | twin: leaning vials misread | real: good vials flagged (looked at only) |
   |---|---|---|---|---|
   | 1 basic (60 frames) | 208/215 | 150/753 | 147/188 | 123/141 |
   | 2 + leaning vials (150 more) | 200/215 | 28/753 | 24/188 | 101/141 |
   | 3 + camera shift (93 more) | 202/215 | 9/753 | 8/188 | 101/141 |
   | 4 + learn from mistakes | 201/215 | 13/753 | 12/188 | 106/141 |

   On the twin it gets much smarter each round (false alarms 150 -> 9-13, leaning vials misread 147 -> 8-12). On the
   real camera it still flags most good vials (and calls many empty holes "missing cap"): the twin's empty holes are
   dark, the real ones are bright and brown. So this model is **not** used for decisions; what the twin taught us is
   used as tested rules (dim-light fix, leaning-vial quick check). Closing that gap needs twin materials that match the
   real plate and holes - the next step, and exactly what the loop is built to measure.

The dashboard's **🧪 Digital twin** tab (also on the start page) shows sample synthetic frames: ground truth next
to our checks, run live on the frame, with or without the robot's count.

## Reasoning audit (v23): where the model's thinking went wrong, checked against the official labels
We read the NVIDIA model's saved answers for the hard positions on the labelled real frames:

| position | truth | model said |
|---|---|---|
| G19 (20:12:06, 20:12:54) | leaning vial, cap seated | **missing_cap, confidence 1.0, "mouth is open"** |
| G17 (several frames) | open vial | **pass, "the cap is seated flat and covers the opening"** |
| G20 (20:12:06) | open vial | **pass, "the cap is seated flat"** |
| G16 | leaning vial | pass |

Almost every answer repeats a sentence from our own prompt and the confidence is nearly always 0.8, so the model's
words are not evidence on their own. Our physical rules caught G17/G20, but two rules let the model's wrong answers
through, against the official definitions:

1. **missing_cap = "a vial is present, but its mouth is open."** If the model said missing cap while a yellow cap
   covered 45-90 % of the opening, we REJECTED it (the "ask again" check only fired at >= 90 %). A cap over the mouth
   means the mouth is not open. **Fixed:** the second look now fires from 45 %; without a confident second look and
   a fully seated cap, it is REVIEW "leaning vial or tilted cap?", never a blind REJECT.
2. **crooked_cap = "visibly tilted or not seated. A leaning vial by itself is not enough."** If the model said
   crooked and the cap sat off the hole centre, we REJECTED it. Off-centre is exactly what a leaning vial looks like.
   **Fixed:** an off-centre cap that covers most of the opening, or this vial's own whole cap next to the hole, is a
   REVIEW (one-click "Just leaning / Cap tipped"), not a REJECT.
3. **pass = "visible cap and vial body appear acceptable."** Kept: a PASS still needs a yellow cap over the opening,
   which is what catches the model's "pass" on open vials (G17, G20).
4. **no_vial = "whether that is an error depends on fill progress."** Kept: fill order / robot count decides EMPTY vs
   GAP vs missing cap.

Re-scored from the saved answers (DEV + TEST, with and without the robot count): good or leaning vials REJECTED
2 -> **0**; real defects flagged unchanged (all of them; G19 counted either way). Every flagged card now shows the
official definition of its label.

## Measured results
See evaluation/RESULTS.md. On 6 held-out real frames: **6/6 defects caught, 0/69 good vials flagged**
(the NVIDIA model alone caught 1/6 and flagged 9 good vials). 13 of 247 empty holes go to human REVIEW.
If two leaning vials are counted as good (official rule), it is 3/3 defects caught and 3/72 good vials flagged,
versus 0/3 and 10/72 for the model alone. **With the robot's placement count** (sidebar "Vials placed so far"),
false alarms on empty holes drop from 13/247 to 0/247 on the test set, and from 35/157 to 0/157 on empty plates.

### Per class, held-out frames only (v32)
The lab asks for results per class, false negatives per class, and the false-alarm rate at the operating point that
catches every defect. Scored on held-out real frames only (never tuned on), labels as written; "caught" = REJECT,
REVIEW or GAP. Also in the website's 📈 Evidence tab, with a "⬇ Results summary (Markdown)" download.

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

### Round 2: 28 more real photos (v20)
Labelled by eye from position-numbered views **before** the system ran on them
(`evaluation/round2_practice_labels.csv`, `evaluation/round2_locked_test_labels.csv`). Split by time: 15 practice
frames (looked at for mistakes) and 13 **locked** frames (scored only, never used to change anything).
Local checks only (no API), fill estimated from the image:

| | practice (15) | locked test (13) |
|---|---|---|
| Good vials wrongly flagged | 0/25 | **0/59** |
| Defects / missed placements flagged | 1/1 (G01 missed placement) | 6/8 - the 2 unflagged are G19 (one vial, two frames), labelled crooked_cap by eye; it leans and its cap reads as seated. Counted as misses. |
| Empty holes left to a person | 30/795 | 22/642 |
| ...with the robot's placement count | 0/795 | **0/642** |

The only pattern in the practice mistakes: the 2 holes right after the last capped vial are left to a person when no
count is given. We kept that on purpose - on real frame 20:12:06, G20 is an uncapped vial exactly there and this
rule is what catches it - but made it a one-click **✔ Empty hole / ✖ Uncapped vial** check.

## Reproduce
    run_test.bat        (asks for an NVIDIA key once, saves it to nvidia_key.txt - never commit that file)
    python batch_eval.py --list test_images.txt
    python batch_eval.py --list test_images_with_counts.txt   (same frames, with the robot's placement counts)
Runs the full pipeline on the listed frames from the Dataset 1 zip and writes runs/<time>/report.json.
Model answers are cached in vlm_cache.json, so re-scoring after a logic change is free.

## Demo
    run_app.bat         (or: python -m streamlit run app.py) -> opens http://localhost:8501
- "Local CV only" mode (no API) now has the same Step 1 as the model mode: check the grid, 🎯 Auto-align if the
  circles are off, tick "Alignment looks correct", then ▶ Inspect (instant). Re-aligning asks for a new inspection.
- Each inspected image keeps its own grid and results: flip to another image, re-align there, flip back - nothing
  is lost. Re-aligning an inspected image asks for a fresh inspection of that image only.
- **🔎 Review every vial** tab (human review workstation): the scene with every position coloured by class (pass,
  missing cap, crooked cap, damaged vial, no vial, needs review); dashed ring = awaiting review, solid ring + ✔ =
  reviewed; ← Previous / Next → / ⏭ Next unconfirmed (flagged first, then the next image); for the selected position:
  close-up, row/column, **Visible vial class** drop-down (change it and the line message / reports use your class),
  "I reviewed this position", an observation note, and *why the system decided this* (plain language + the model's
  answer + the official definition). Progress counter across the whole inspection, "Confirm every PASS / EMPTY
  verdict", and **Export / Load review JSON**. The 🚨 Needs attention tab is unchanged.
- Drop one or many camera images. Flip between them with the ◀ ▶ buttons or the picker (✅ inspected, ⏳ waiting).
- "▶ Inspect this image" runs one; "⏩ Inspect all waiting images" runs the whole batch; "📚 All images in this
  inspection" shows every inspected image side by side plus a summary table (downloadable CSV). "➕ New inspection" clears the images and starts over.
- Each image keeps its own "Good placed / Scrap placed" count.
The dashboard reuses answers saved by batch_eval.py for the same image, so demos cost no API calls.

## Limitations (say these in the video)
- The digital twin's caps, glass and plate look cleaner than the real camera, so models trained only on it do not
  transfer yet (measured above); it is used to test rules, not to decide.
- Labels are by eye from 640x480 images, by one person.
- The model cannot reliably tell a bright empty hole from an uncapped glass vial; those go to REVIEW.
- The black robot cable is not detected as an obstruction (only the yellow arm is).
- Small real test set: 14 held-out defects, and 3 of the crooked caps are the same leaning vial (G19) in 3 frames.
- No damaged_vial example in the real data, so that class is untested on real frames.
- The cap check looks for the lemon-yellow cap; another cap colour needs new settings.
- On a new camera (Cap_seal2, 960x540) caps can sit off the hole centre after auto-fit and get flagged (never passed).
- Model mode needs internet; failed calls are retried once automatically (if only a few failed), then become REVIEW.
- Not yet tested on the physical CSI line (the lab asks for a supervised test - see #data-and-testbed-requests).
