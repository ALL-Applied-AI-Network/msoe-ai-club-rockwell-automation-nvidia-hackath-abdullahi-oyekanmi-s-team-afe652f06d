"""
helpbot_kb.py - the checked knowledge base behind Vial Helper.

Every answer was written against the real button / tab / setting names in app.py and the rules in vial_core.py.
Numbers are {placeholders} filled from vial_core.MEASURED and the rule constants at answer time (see helpbot._facts),
so they can never go out of date. Literal braces are not used in answers.
Each entry: id, title (the question it answers), keys (words / phrases people use), a (answer), rel (related ids).
"""

KB = []


def E(id, title, keys, a, rel=()):
    KB.append(dict(id=id, title=title, keys=list(keys), a=a, rel=list(rel)))


# ===================================================================== A. GETTING STARTED
E("howto", "How do I use the model / website, step by step?",
  ["how does this work", "how this works", "im new", "i'm new", "do first", "what do i do", "new user", "new here", "how do i use", "how to use", "use the model", "use the website", "use the app", "use this", "use it",
   "step by step", "walkthrough", "walk me through", "getting started", "get started", "beginner", "tutorial",
   "guide", "instructions", "how does it work", "first time", "how do i inspect", "where do i start"],
  "Five steps:\n1. **Start the website** (Windows key + R → the `cmd /k ... streamlit run app.py` command) - it opens at "
  "http://localhost:8501.\n2. **Pick a mode** in the sidebar: *Local CV only (instant, no API)* or *Local CV + NVIDIA "
  "vision model* (paste your API key).\n3. **Drop one or more camera photos** onto the upload box.\n4. **Step 1 · Align "
  "the grid**: check that every circle sits on a hole (🎯 Auto-align if not), tick **Alignment looks correct**, click "
  "**▶ Inspect this image**.\n5. **Read the results**: the verdict banner, **🚨 Needs attention** (each flagged vial "
  "with its close-up and reason, plus one-click checks) and **🔎 Review every vial**. Download the line message or a "
  "report if needed.\nOptional: type the robot's count in **Good placed** for even fewer checks.",
  ["start", "modes", "align", "statuses"])
E("start", "How do I start the website?",
  ["start", "launch", "open the website", "open the app", "run the app", "run the website", "get it running",
   "get this running", "running", "boot", "command", "windows key", "windows key r", "what command", "cmd",
   "streamlit", "localhost", "url", "8501", "turn on"],
  "Press **Windows key + R**, paste this and press Enter:\n\n`cmd /k \"cd /d C:\\Users\\malik\\Downloads\\"
  "vial_inspector_v7\\vial_inspector_v7 && python -m streamlit run app.py\"`\n\nThe website opens at "
  "**http://localhost:8501** (copy that into a browser if it doesn't open by itself). Or double-click **run_app.bat** in "
  "the folder. Keep the black window open while you use it.",
  ["stop", "install", "howto"])
E("stop", "How do I stop or restart the website?",
  ["stop", "restart", "close the app", "shut down", "quit", "ctrl c", "ctrl+c", "reload", "refresh"],
  "Click the black command window and press **Ctrl+C** (or close it). To restart, run the start command again. After "
  "the code files are updated you must restart, otherwise the old version keeps running. Refreshing the browser page "
  "only restarts the page, not the code.", ["start", "update"])
E("install", "What do I need to install?",
  ["install", "requirements", "pip", "python version", "setup", "dependencies", "libraries", "packages",
   "module not found", "modulenotfounderror", "no module named"],
  "Python 3 plus the packages in **requirements.txt** (streamlit, pillow, numpy, pandas, requests, "
  "opencv-python-headless). In the app folder run:\n\n`python -m pip install -r requirements.txt`\n\nA "
  "\"No module named …\" error means this step is missing. The Isaac Sim parts are optional and have their own "
  "**isaac\\setup_isaac.bat**.", ["start"])
E("update", "How do I update to a new version?",
  ["nothing changed", "updated the files", "still old", "no change after update", "update", "new version", "upgrade", "latest version", "zip", "extract", "replace files", "copy files"],
  "Extract the new zip and either run it from its own folder, or copy its files over your working folder (keep your "
  "own **nvidia_key.txt** and **vlm_cache.json**). Then stop the website (Ctrl+C) and start it again - a refresh "
  "alone keeps the old code.", ["stop", "start"])
E("demo", "Which photos should I use for a demo?",
  ["demo", "demo images", "show the judges", "presentation", "example images", "sample images", "test photos",
   "which photos", "present"],
  "Use the photos in **demo_images**: their NVIDIA model answers are saved, so they inspect instantly and cost no API "
  "calls. Good ones to show: the 20:12:06 frame (an open vial G17, an uncapped G20, a leaning vial G16), an "
  "early frame with few vials, and an empty-plate frame. Also show **🔎 Review every vial**, **📈 Evidence** and "
  "**🧪 Digital twin**.", ["savedanswers", "evidence"])
E("runbat", "What are the .bat files?",
  ["bat", "run_app.bat", "run_test.bat", "change_key_and_run", "batch file", "double click", "run_extra_test"],
  "- **run_app.bat** - starts the website (and reads nvidia_key.txt if it exists, so the key box is pre-filled).\n"
  "- **run_test.bat** - runs the batch test on the frames in test_images.txt and asks for a key the first time.\n"
  "- **run_extra_test.bat** - the extra test set.\n- **isaac\\run_synth.bat / resume_synth.bat** - render synthetic "
  "images in NVIDIA Isaac Sim.", ["start", "batchtest", "twinrun"])

# ===================================================================== B. PHOTOS
E("upload", "How do I add photos?",
  ["upload", "add photo", "add image", "add a picture", "drop", "drag", "photos", "pictures", "more images",
   "add more", "another image", "load image", "folder of", "several images", "multiple images", "multiple pictures"],
  "Drag photos onto **\"Drop one or more overhead camera images\"** (or click it to browse). You can add more at any "
  "time; each one appears in the image picker.", ["flip", "newinsp", "formats"])
E("formats", "Which photo formats and sizes work?",
  ["format", "formats", "jpg", "jpeg", "png", "file type", "resolution", "photo size", "image size", "screenshot"],
  "PNG, JPG and JPEG. Any size: standard 640x480 CSI frames are used as they are, other sizes / shapes are "
  "auto-fitted into the standard view. Use the original camera file, not a screenshot - drawings on a screenshot "
  "confuse the hole finder.", ["autofit"])
E("flip", "How do I switch between photos?",
  ["switch", "flip", "next image", "previous image", "other image", "go back", "picker", "◀", "▶", "which image"],
  "Use **◀ / ▶** next to the image picker, or pick from the list (✅ inspected, ⏳ waiting). Each image keeps its own "
  "grid, count and results - flipping away and back never loses them.", ["multi"])
E("multi", "Can I inspect many photos at once?",
  ["many", "batch", "all at once", "several", "in one go", "10 images", "inspect all", "multiple", "bulk",
   "whole folder", "all images"],
  "Yes. Upload them all, check the grid on one, then click **⏩ Inspect all … waiting images** - it runs them one "
  "after another with the grid you checked. **📚 All images in this inspection** shows them side by side with a "
  "summary table, and **📄 Build session report** makes one report for all.", ["inspect", "sessionreport"])
E("newinsp", "How do I start over / remove photos?",
  ["new batch", "new set of photos", "fresh batch", "start over", "clear", "remove", "delete image", "new inspection", "reset everything", "fresh start", "➕"],
  "**➕ New inspection** clears all photos and starts over (saved model answers are kept, so re-inspecting the same "
  "photos is free). Single photos can't be removed one by one - start a new inspection instead.", ["upload"])
E("order", "Does the order of photos matter?",
  ["order", "sequence", "time order", "same run", "frames from one shift"],
  "Only for **change-aware** inspection: with photos from the same run, a position that looks the same as in the "
  "photo before keeps its earlier answer. Photos are best uploaded in time order (the CSI file names sort that way).",
  ["changeaware"])

# ===================================================================== C. GRID & ALIGNMENT
E("align", "The circles are not on the holes - how do I align?",
  ["align", "alignment", "circles", "grid", "off", "not on the holes", "misaligned", "shifted", "line up",
   "circles dont match", "wrong place", "move the grid", "snap"],
  "Open **Step 1 · Align the grid…**. If the circles are off, click **🎯 Auto-align** - it finds the round holes and "
  "snaps both 6x5 grids (tested on {al_cases} shifted/rotated frames: worst error {al_worst} px, median {al_med} px; a "
  "hole's radius is ~10.5 px). **↺ Reset** goes back to the default grid. When every circle sits on a hole, tick "
  "**Alignment looks correct** and inspect.", ["autofit", "gauges", "calib"])
E("autoalign", "What does Auto-align do exactly?",
  ["auto-align", "auto align", "autoalign", "how does auto align", "🎯"],
  "It detects the round holes in the photo and tries small shifts, rotations and scales of each 6x5 grid; each "
  "candidate scores +1 for every grid circle on a hole and is penalised for holes found just outside the grid (where "
  "the plate ends), so it can't slide a whole column. It never makes a good grid worse. If the plates moved a lot, "
  "it falls back to the full auto-fit.", ["align", "autofit"])
E("reset", "What does Reset do?",
  ["reset", "↺", "default grid", "undo align", "go back to default"],
  "**↺ Reset** puts the grid back to the default CSI camera calibration for this image (and forgets its auto-fit "
  "override). Use it if Auto-align made things worse.", ["align"])
E("autofit", "Photos from another camera / different size (auto-fit)",
  ["auto-fit", "autofit", "auto fit", "different camera", "other camera", "new camera", "bigger image",
   "larger image", "16:9", "wide", "zoom", "zoomed", "framing", "aspect", "straighten", "1920", "1280", "960",
   "different shape", "handle"],
  "Sidebar → **🧭 Auto-fit photos of any size / framing** (on). For a photo that isn't a standard 640x480 CSI frame, "
  "the system finds both fixtures by their hole pattern (same-size holes ~3 radii apart in rows and columns, darker "
  "than the plate or yellow caps), straightens the photo into the standard view and fine-tunes. A blue note says how "
  "many holes it found. Tested on {af_frames} real frames x {af_var} camera changes: {af_min}-{af_max}% of decisions "
  "identical to the original.", ["align", "formats"])
E("gauges", "What do the hole-match numbers next to the grid mean?",
  ["gauge", "gauges", "holes matched", "match", "25/30", "30/30", "ring", "quality", "how many holes"],
  "They count how many of each plate's grid circles sit on a detected round hole or cap (good plate out of 30, scrap "
  "out of 25 - its edge column is half hidden). Low numbers mean the grid is probably off - try 🎯 Auto-align. Low "
  "numbers are expected when the robot arm covers part of a plate.", ["align"])
E("calib", "Can I set the grid by hand?",
  ["manual", "by hand", "calibration", "corner", "corners", "coordinates", "advanced", "fine tune", "fine-tune"],
  "Sidebar → **Calibration (advanced)**: the centres of the 4 corner holes of each 6x5 grid in 640x480 pixels "
  "(TL, TR, BR, BL for the good and scrap plates). Use Auto-align first; edit only for small corrections.",
  ["align"])
E("numbering", "How are the positions numbered (G01, S30...)?",
  ["numbering", "numbered", "position number", "g01", "which is g01", "where is", "fill order", "column", "row",
   "position id", "naming"],
  "**G** = good fixture (left), **S** = scrap fixture (right), 30 each. Numbering follows the robot's fill order: "
  "01-05 is the **rightmost column, top to bottom**, then 06-10 one column to the left, and so on to 26-30 on the left.",
  ["count", "fixtures"])
E("fixtures", "What are the good and scrap fixtures?",
  ["difference between g and s", "g and s", "g vs s", "g or s", "fixture", "fixtures", "good fixture", "scrap fixture", "scrap tray", "good tray", "plate", "plates", "tray"],
  "Two 6x5 plates: the **good fixture** (left, positions G01-G30) holds vials that passed; the **scrap fixture** "
  "(right, S01-S30) holds rejected vials. A defect found in the good fixture must leave it (DIVERT_TO_SCRAP); a defect "
  "in the scrap fixture is only logged (LOG_SCRAP_DEFECT).", ["numbering", "line"])
E("realign", "Why did I have to inspect again after aligning?",
  ["inspect again", "re-inspect", "reinspect", "after aligning", "results reset", "lost after align"],
  "Results belong to the grid they were made with. Re-aligning an image that was already inspected moves its circles, "
  "so that image (only that one) asks for a fresh inspection. Other images keep their results.", ["flip"])

# ===================================================================== D. INSPECTING
E("inspect", "How do I inspect an image?",
  ["inspect", "inspect this image", "run the model", "start inspection", "check the image", "analyze", "analyse",
   "scan", "process", "evaluate the image", "▶"],
  "In **Step 1**, check the circles, tick **Alignment looks correct**, then click **▶ Inspect this image** (local "
  "mode: instant; model mode: the button shows how many model calls it needs). Results appear below.\n\n**Button "
  "greyed out?** Tick *Alignment looks correct* first; in model mode also paste an API key. If the image is already "
  "inspected, there is nothing left to do.", ["multi", "statuses"])
E("calls", "How many model calls does one image use?",
  ["how many calls", "model calls", "api calls", "requests", "how much api", "usage", "per image"],
  "Only positions that are not clearly empty go to the model (sidebar: **Skip the API for clearly empty holes**), "
  "plus a second look where the model contradicts the camera. A full plate is ~30-60 calls, a nearly empty one much "
  "fewer. Change-aware inspection reuses answers between photos ({ch_saved} fewer calls on a real shift).",
  ["changeaware", "savedanswers"])
E("speed", "Why is it slow / how long does it take?",
  ["slow", "how long", "time", "takes long", "speed", "faster", "waiting", "seconds", "loading forever"],
  "Local CV only is instant. With the NVIDIA model each position takes a few seconds (the model runs on NVIDIA's "
  "servers), several in parallel (sidebar **Parallel requests**). Saved demo photos are instant. A first auto-fit of a "
  "photo takes ~3 s.", ["modes", "settings"])
E("savedanswers", "What are saved model answers?",
  ["saved answers", "cached", "cache", "vlm_cache", "free", "no calls", "reuse answers", "reused"],
  "Model answers from the batch test are stored in **vlm_cache.json**. When you inspect the same photo with the "
  "default grid, those answers are reused - no API calls, no cost. That's why the demo_images photos inspect "
  "instantly.", ["demo", "batchtest"])
E("batchtest", "What is the batch test (batch_eval.py)?",
  ["batch_eval", "batch test", "run_test", "test_images.txt", "report.json", "runs folder", "command line test"],
  "`python batch_eval.py --list test_images.txt` runs the full pipeline (rules + model) on the listed frames without "
  "the browser and writes runs/<time>/report.json; model answers are cached, so re-scoring after a rule change is "
  "free. `test_images_with_counts.txt` adds the robot's counts.", ["savedanswers", "accuracy"])

# ===================================================================== E. RESULTS & STATUSES
E("statuses", "What do PASS / REJECT / REVIEW / GAP / EMPTY / CAP / NO VIEW mean?",
  ["status", "statuses", "colors", "colours", "color", "colour", "icons", "legend", "what do the colors",
   "what does each status"],
  "- **PASS** (green ✓) - a yellow cap covers the vial opening and the model agrees.\n- **CAP** (blue, local mode) - a "
  "cap is present; seating not judged without the model.\n- **REJECT** (red ✗) - evidence of a defect.\n- **REVIEW** "
  "(orange ?) - uncertain or contradictory; a person checks.\n- **GAP** (purple !) - empty although later positions "
  "are filled: a missed placement.\n- **EMPTY** (grey) - empty hole, still waiting.\n- **NO VIEW** (dark) - the camera "
  "can't see it; no guess.", ["s_pass", "s_review", "s_gap"])
E("s_pass", "What does PASS mean?",
  ["pass", "passed", "green", "green check", "what is pass", "what does pass mean"],
  "PASS = a vial whose yellow cap covers at least {cov_min} of the vial opening **and** the NVIDIA model agrees it is "
  "fine (or its second look does), with nothing contradicting it. No person needs to look. Official: \"the visible "
  "cap and vial body appear acceptable; it does not confirm properties hidden from the camera.\"", ["statuses"])
E("s_cap", "What does CAP mean?",
  ["cap status", "blue", "what does cap mean", "cap present", "local check only"],
  "CAP appears in **Local CV only** mode: a yellow cap covers the opening, but without the model the system does not "
  "judge whether it is seated or tilted (or the vial damaged). In model mode these become PASS or something else.",
  ["modes"])
E("s_reject", "What does REJECT mean?",
  ["reject", "rejected", "red", "red x", "what does reject mean", "defect found", "defective"],
  "REJECT = the evidence points to a defect: an open vial mouth (missing_cap), a tilted / unseated cap (crooked_cap) "
  "or a damaged vial, or a missing cap where the robot count says a vial must be. In the good fixture the line "
  "message says **DIVERT_TO_SCRAP**.", ["labels", "line"])
E("s_review", "What does REVIEW mean?",
  ["review", "orange", "question mark", "what does review mean", "check by eye", "uncertain", "needs review"],
  "REVIEW = not passed automatically because something is uncertain or contradictory - e.g. the model and the camera "
  "disagree, the cap is off the opening (leaning vial or tipped cap?), a model error, or the hole right after the last "
  "cap. A person decides; many have one-click buttons. A missed defect is treated as worse than a false alarm.",
  ["quick", "whyflag"])
E("s_gap", "What does GAP mean?",
  ["gap", "purple", "exclamation", "missed placement", "missing vial", "skipped", "what does gap mean"],
  "GAP = the hole is empty although positions later in the robot's fill order already hold vials, so the robot "
  "probably skipped it - a **missed placement**. The line message says **REPLACE_MISSED_VIAL**. Official: an empty "
  "hole is no_vial; whether that is an error depends on fill progress.", ["count", "line"])
E("s_empty", "What does EMPTY mean?",
  ["empty", "grey", "gray", "waiting", "what does empty mean", "empty hole"],
  "EMPTY = no vial there yet and that is fine: the hole is beyond how far the robot has filled (from the robot count "
  "or estimated from the image), or a clearly dark, even hole at the end of the fill.", ["count", "estimate"])
E("s_noview", "Why are some positions NO VIEW?",
  ["no view", "hidden", "s01", "s02", "s03", "s04", "s05", "plate edge", "robot arm", "blocked", "arm", "covered",
   "can't see", "cannot see", "why is s0"],
  "S01-S05 are half hidden at the scrap plate's edge in this camera view, and positions under the yellow robot arm "
  "can't be seen. They are reported as **NO VIEW** (and RECAPTURE_FRAME when the arm is in the way) instead of "
  "guessing. The edge rule can be turned off: sidebar **S01–S05 hidden at the plate edge → NO VIEW**.",
  ["statuses", "line"])
E("verdict", "What does the big verdict banner mean?",
  ["banner", "verdict", "defective vials found", "positions to check", "all clear", "headline", "top message"],
  "It summarises the image: **\"N defective vials found\"** (red) if there are rejects, **\"N positions to check by "
  "eye\"** (orange) if only reviews/gaps, or **\"All clear\"** (green). The cards below count each status.",
  ["statuses"])
E("fillstrip", "What is the FILL PROGRESS bar?",
  ["fill progress", "progress bar", "strip", "occupied", "placed bar", "robot order 01"],
  "One small bar per position in the robot's order 01→30 for each plate, coloured by status, with a marker at the "
  "count (typed or estimated). It shows at a glance where the fill stopped and where gaps or defects are.",
  ["count", "numbering"])
E("map", "What is the Fixture map tab?",
  ["fixture map", "map", "laid out", "plate layout", "🧭 fixture"],
  "**🧭 Fixture map** draws both plates like the real ones with every position coloured by status; hover for "
  "details.", ["tabs"])
E("needs", "What is the Needs attention tab?",
  ["pictures of flagged", "see pictures", "picture of", "images of flagged", "see the flagged", "needs attention", "attention", "flagged list", "🚨", "cards", "close-up", "close up", "picture of the vial",
   "image of the flagged"],
  "**🚨 Needs attention** lists every REJECT, GAP and REVIEW as a card: the close-up picture of that vial with a "
  "coloured ring, its status and label, the reason, the official definition, what the model said and what decided "
  "it. One-click checks are listed last with their buttons.", ["quick", "tabs"])
E("tabs", "What are all the tabs?",
  ["tab", "tabs", "where do i find", "menu", "sections", "pages"],
  "- **🚨 Needs attention** - flagged vials with close-ups and reasons.\n- **🔎 Review every vial** - review "
  "workstation for all 60 positions.\n- **🧭 Fixture map** - plates laid out like the real ones.\n- **📋 All "
  "positions** - table, CSV and all crops.\n- **📈 Evidence** - what we measured.\n- **🧪 Digital twin** - NVIDIA Isaac "
  "Sim test frames.\n- **🎯 Check accuracy** - label the truth, get your own numbers.", ["needs", "review"])
E("allpos", "Where is the table of all positions?",
  ["table", "all positions", "📋", "every position", "spreadsheet", "all crops", "data table"],
  "**📋 All positions**: one row per position (status, label, reason, cap coverage, yellow, brightness, model "
  "latency), **⬇ Download results (CSV)**, and an **All crops** section with every position's picture.",
  ["reports"])

# ===================================================================== F. LABELS
E("labels", "Official label definitions",
  ["label", "labels", "definition", "definitions", "official", "classes", "defect types", "types of defects",
   "categories", "definition of"],
  "OFFICIAL_LIST\n\nEvery flagged card shows the definition of its label.", ["l_missing", "l_crooked", "lean"])
E("l_missing", "What is a missing cap?",
  ["missing cap", "missing_cap", "open vial", "no cap", "uncapped", "mouth open", "what is a missing cap",
   "cap missing", "open mouth"],
  "Official: **missing_cap** - \"A vial is present, but its mouth is open. Do not confuse the transparent vial body "
  "with an empty fixture hole.\" From above an open vial shows a glass rim and no yellow over the opening; an empty "
  "hole is dark. The camera rule: a pass needs a yellow cap over at least {cov_min} of the opening.",
  ["l_novial", "emptyvsopen"])
E("l_crooked", "What is a crooked cap?",
  ["crooked cap", "crooked_cap", "what is a crooked cap", "tilted cap", "tipped cap", "not seated", "cap not seated",
   "is a tilted cap a defect"],
  "Official: **crooked_cap** - \"The cap is visibly tilted or not seated. A leaning vial by itself is not enough.\" "
  "From straight above, a tipped cap and a leaning vial can look alike, so off-centre caps become a one-click check "
  "(✔ Just leaning / ✖ Cap tipped).", ["lean", "quick"])
E("l_damaged", "What is a damaged vial / can it find cracks?",
  ["cracks", "breaks", "chip", "damaged", "damaged_vial", "broken", "crack", "cracked", "chipped", "damage", "broken glass", "what counts as damaged"],
  "Official: **damaged_vial** - \"The target vial body is visibly damaged. Overlapping transparent bodies may require "
  "the full scene for context.\" Damage is judged by the NVIDIA model (the camera rules can't see cracks). Honest "
  "limit: the labelled real frames contain no damaged vials, so this is untested on real data.", ["labels", "limits"])
E("l_novial", "What is no_vial?",
  ["no_vial", "no vial", "empty aperture", "what is no vial", "what is no_vial"],
  "Official: **no_vial** - \"The target aperture is empty. Whether that is an error depends on fill progress.\" So "
  "an empty hole is EMPTY (waiting) or GAP (missed placement) depending on the robot's fill order.",
  ["s_gap", "s_empty"])
E("l_pass", "What is the official meaning of pass?",
  ["official pass", "pass definition", "definition of pass"],
  "Official: **pass** - \"The visible cap and vial body appear acceptable. It does not confirm properties hidden from "
  "the camera.\"", ["s_pass"])
E("whatdefect", "What counts as a defect?",
  ["what counts as a defect", "is it a defect", "defect", "defects", "what is wrong", "faulty"],
  "Defects are **missing_cap**, **crooked_cap** and **damaged_vial**; a **missed placement** (GAP) is a fill error. A "
  "leaning vial by itself is **not** a defect, and an empty hole is only an error if the robot should already have "
  "filled it.", ["labels", "lean"])

# ===================================================================== G. HOW DECISIONS ARE MADE
E("pipeline", "How does the system decide? (the pipeline)",
  ["how does it decide", "pipeline", "how does the model work", "how it works", "algorithm", "process",
   "decision process", "logic", "behind the scenes", "under the hood", "architecture"],
  "1. **Grid**: 60 positions from 4 corner holes per plate (auto-align / auto-fit).\n2. **Camera checks** per position: "
  "is there lemon-yellow over the opening (cap coverage), is the hole dark and even (empty)?\n3. **NVIDIA vision model** "
  "(model mode): labels each non-empty position from its close-up.\n4. **Physical rules check the model**: a PASS needs "
  "a cap over the opening; contradictions get a second look or REVIEW; errors become REVIEW.\n5. **Fill order** (robot "
  "count or estimate): EMPTY vs GAP vs missing cap.\n6. **Line message** for the robot/PLC.", ["passrule", "secondlook",
                                                                                             "count"])
E("passrule", "What is needed for a PASS?",
  ["needed for a pass", "pass rule", "when does it pass", "why did it pass", "45%", "coverage rule",
   "a 'pass' needs", "a pass needs"],
  "A yellow cap covering at least **{cov_min}** of the vial opening (measured in the centre of the hole) **and** no "
  "contradiction: in model mode the model (or its second look) must say pass. Sidebar **A 'pass' needs a yellow cap** "
  "keeps this safety rule on - leave it on.", ["coverage", "secondlook"])
E("coverage", "What is cap coverage / \"cap over hole\"?",
  ["coverage", "cap over hole", "center coverage", "centre coverage", "cap covers", "percent covered", "yellow nearby"],
  "**Cap over hole** = how much of the centre of the hole is covered by lemon-yellow (a seated cap covers it; an open "
  "vial or a displaced cap doesn't). **Yellow nearby** = yellow anywhere around the hole (a neighbour's or a fallen cap "
  "can count here, so it isn't enough for a pass). A pass needs at least {cov_min} cap over hole.", ["passrule"])
E("yellow", "How does it recognise a cap?",
  ["recognise a cap", "recognize a cap", "detect cap", "yellow detection", "cap color", "cap colour", "lemon",
   "hue", "how does it see the cap"],
  "Caps are lemon-yellow: hue in a narrow yellow band, strongly coloured and bright. The brightness floor follows the "
  "light (measured from the metal plate around each hole) so dim lighting doesn't hide caps - a weakness found in the "
  "digital twin and fixed without changing any real-frame result. Glass seen down an open vial is orange-brown and "
  "doesn't count.", ["dimlight"])
E("dimlight", "Does it work in dim or bright light?",
  ["dim", "dark", "lighting", "light", "brightness", "bright", "shadow", "night", "lamp"],
  "Yes, within limits. The digital twin showed that with ~40% less light, good caps fell below the old fixed brightness "
  "rule; now the rule follows the light measured from the plate, and all dim-light good caps in the twin are "
  "recognised. On the real camera this changed none of 17,820 positions.", ["twin", "yellow"])
E("secondlook", "What is the second look?",
  ["second look", "ask again", "asked again", "verify", "second opinion", "contradiction", "contradicts"],
  "When the model says \"missing cap\" but the camera sees a yellow cap over at least {cov_min} of the opening (so the "
  "mouth is not open), the model is asked a focused second question: is this cap seated or crooked? If it says pass, "
  "the vial passes; otherwise it goes to REVIEW (\"leaning vial or tilted cap?\").", ["confidence", "passrule"])
E("emptyvsopen", "How does it tell an empty hole from an open vial?",
  ["empty or open", "empty hole vs", "open vial vs empty", "transparent", "glass", "clear vial", "tell apart"],
  "Hard from above - a clear glass vial without a cap can look like a bright empty hole, and the model confuses them "
  "too. The system uses the **fill order**: within the robot's count (or before the last capped vial) a hole with no "
  "cap must hold a vial → REJECT/GAP; beyond it → EMPTY; right after the last cap → a one-click \"Empty hole / "
  "Uncapped vial\" check.", ["count", "quick"])
E("estimate", "What is the fill estimate (when no count is typed)?",
  ["estimate", "estimated", "fill from image", "estimate the fill", "no count", "without count", "last seated cap",
   "last capped"],
  "Without a robot count, the system estimates how far the robot has filled the **good** plate from the last position that has "
  "its own cap (seated, or a leaning vial's cap beside the hole). Holes before it with no cap → defect / GAP; the "
  "{margin} holes right after it → one-click check; holes further on → EMPTY. Sidebar: **Estimate the fill from the "
  "image when no count is typed**. The scrap plate is not estimated: scrap vials are often uncapped, so a missing cap "
  "there does not mean an empty hole - each scrap position keeps its own camera/model verdict (a typed count still "
  "applies).", ["count", "quick"])
E("modelerror", "What happens if the model errors or times out?",
  ["model error", "timeout", "timed out", "error", "refused", "invalid", "no answer", "fails", "api error"],
  "That position becomes **REVIEW** (\"No model answer…\") - never PASS. If only a few calls fail, the system retries "
  "them once automatically; pressing Inspect again retries any that are still missing. Fix the key / credits or raise the sidebar "
  "**Timeout per request**, then inspect again (only missing answers are re-asked).", ["apikey", "settings"])
E("strict", "What does Strict mode do?",
  ["strict", "strict mode", "flag more", "two crops", "wide and tight"],
  "Sidebar **Strict mode (flag more)**: every position is asked twice (a wide and a tight crop); a PASS must survive "
  "both answers, any disagreement becomes REVIEW, and if the model reports a visible opening or a tilted cap the vial "
  "is not passed. More reviews, fewer misses - and twice the model calls.", ["settings"])
E("skipempty", "What does Skip the API for clearly empty holes do?",
  ["skip the api", "skip empty", "clearly empty", "save api calls on empty"],
  "Holes that are clearly dark and even are marked EMPTY by the camera rules without asking the model - fewer calls "
  "and no chance for the model to call them \"pass\". Brighter empty holes still go to the model and are then settled "
  "by the fill order.", ["calls"])
E("changeaware", "What is change-aware inspection?",
  ["change-aware", "change aware", "only changed", "reuse between frames", "previous frame", "fewer calls",
   "save calls", "carried", "reuse answers between"],
  "Sidebar → **Change-aware (only re-inspect what changed)**. A position that looks the same as in the previous photo "
  "of the run keeps its answer; only new or changed positions go to the model. On {ch_frames} real frames: "
  "{ch_all:,} calls became {ch_aware:,} ({ch_saved} fewer), with every labelled change re-inspected. Only photos "
  "taken within 10 minutes of each other are linked.", ["calls"])
E("settings", "What do the sidebar settings do?",
  ["setting", "settings", "sidebar", "options", "parallel", "parallel requests", "timeout", "timeout setting",
   "crop size", "configuration"],
  "- **Mode** - local only or + NVIDIA model.\n- **NVIDIA API key / Model** - model mode only.\n- **Parallel requests / "
  "Timeout per request** - speed vs rate limits.\n- **Skip the API for clearly empty holes**, **Strict mode (flag "
  "more)**, **A 'pass' needs a yellow cap** (keep on).\n- **Change-aware**, **Estimate the fill from the image…**, "
  "**🧭 Auto-fit**, **S01–S05 hidden…**, **Crop size**, **Calibration (advanced)**. Ask me about any of them.",
  ["strict", "skipempty", "changeaware"])
E("hiddenrule", "Can I turn off the S01-S05 hidden rule?",
  ["turn off hidden", "s01-s05 rule", "edge rule", "show s01"],
  "Yes: untick sidebar **S01–S05 hidden at the plate edge → NO VIEW**. Their verdicts are then guesses from half a "
  "hole, so keep it on unless your camera sees that column fully.", ["s_noview"])
E("robotarm", "What if the robot arm is in the picture?",
  ["robot arm", "arm in the way", "yellow arm", "blocked by arm", "recapture"],
  "Positions under the big yellow arm are detected (it's a yellow blob far bigger than a cap) and marked **NO VIEW**; "
  "the line message asks for **RECAPTURE_FRAME**. Use a photo where the arm is out of the way for a full inspection.",
  ["s_noview", "line"])

# ===================================================================== H. QUICK CHECKS & REVIEW
E("lean", "Leaning vial vs crooked cap - why is it flagged?",
  ["lean", "leaning", "tilted vial", "slightly tilted", "just tilted", "off center", "off-centre", "off centre",
   "perspective", "why flag leaning", "flagged for no reason", "is a leaning vial a defect", "leaning vial"],
  "Official rule: *a leaning vial by itself is not a defect*. But from straight above a leaning vial and a tipped cap "
  "look alike - in the digital twin, 55 of 128 tipped caps still looked like neat round caps. Passing them "
  "automatically would miss real defects, so they become a **one-click check**: **✔ Just leaning** (OK) or **✖ Cap "
  "tipped** (defect). A side camera would settle it.", ["quick", "l_crooked"])
E("quick", "One-click checks (Just leaning / Empty hole / Uncapped vial)",
  ["one click", "one-click", "quick check", "buttons", "just leaning button", "cap tipped button", "empty hole button",
   "uncapped vial button", "button do", "⚡"],
  "Two kinds of uncertain positions get buttons on their card in **🚨 Needs attention**:\n- A whole cap off the "
  "opening → **✔ Just leaning** (PASS) / **✖ Cap tipped** (REJECT crooked_cap).\n- The {margin} holes right after the "
  "last capped vial with no cap → **✔ Empty hole** (EMPTY) / **✖ Uncapped vial** (REJECT missing_cap).\nA click "
  "updates the line message and counts as reviewed.", ["margin", "undo"])
E("margin", "Why are the holes right after the last cap asked about?",
  ["after the last cap", "last cap", "asking me", "next in fill order", "holes after", "empty hole or uncapped",
   "why ask empty"],
  "Right after the last capped vial, an empty hole and the robot's newest vial without its cap look the same. On a "
  "real frame (20:12:06) G20 is an uncapped vial exactly there, so these {margin} holes are a one-click check instead "
  "of a guess. Type the robot's count and they disappear.", ["count", "quick"])
E("undo", "How do I undo a one-click decision?",
  ["undo", "mistake", "clicked wrong", "wrong button", "revert", "↩"],
  "**↩ Undo operator decisions on this image** (below the cards in 🚨 Needs attention) removes all one-click decisions "
  "for that image. In 🔎 Review every vial you can also set the class back with **Visible vial class**.",
  ["quick", "review"])
E("review", "How do I use the review workstation?",
  ["review workstation", "review every vial", "🔎", "review all", "reviewed", "go through every vial",
   "check every position", "mark as checked", "mark as reviewed", "checked"],
  "**🔎 Review every vial**: the photo shows every position coloured by class - dashed ring = waiting, solid ring + ✔ "
  "= reviewed. Pick a position (or use the buttons below), look at its close-up and **Why the system decided this**, "
  "correct **Visible vial class** if needed, tick **I reviewed this position**, add a note.", ["nextun", "classdrop",
                                                                                             "exportreview"])
E("nextun", "How do I go to the next position I haven't checked?",
  ["next unconfirmed", "next unreviewed", "unreviewed", "unconfirmed", "⏭", "skip to next", "next one"],
  "**⏭ Next unconfirmed** in 🔎 Review every vial jumps to the next position nobody has reviewed - flagged ones first, "
  "then the rest, then the next image. **← Previous / Next →** step one position at a time.", ["review"])
E("classdrop", "How do I correct the system's verdict?",
  ["correct", "correction", "change the class", "override", "wrong verdict", "system is wrong", "fix a mistake",
   "visible vial class", "change label"],
  "In 🔎 Review every vial pick the position and set **Visible vial class** (pass, missing_cap, crooked_cap, "
  "damaged_vial, no_vial, needs_review). Your class is used in the line message and reports and is logged as "
  "decided by the operator; choosing the system's own class again removes the override.", ["review", "undo"])
E("dashed", "What do the dashed and solid rings mean?",
  ["dashed", "dashed ring", "solid ring", "white halo", "ring colors", "ring colours"],
  "In 🔎 Review every vial: **dashed ring** = not reviewed yet, **solid ring with a green ✔** = reviewed, **white "
  "halo** = the position you are looking at. The ring colour is the class (green pass, orange missing cap, purple "
  "crooked cap, red damaged, blue no vial, grey needs review).", ["review"])
E("confirmall", "Can I confirm all the passed ones at once?",
  ["confirm all", "confirm every", "approve all", "bulk confirm", "mark all"],
  "Yes: **✔ Confirm every PASS / EMPTY verdict in this image** marks all positions the system passed or found empty as "
  "reviewed, so you only go through the flagged ones.", ["review"])
E("exportreview", "How do I save or load a review?",
  ["export review", "save review", "load review", "review json", "vial_review.json", "continue later",
   "save my work", "export my review"],
  "**⬇ Export review JSON** saves every position's system status, final class, who decided, reviewed flag, note and "
  "reason for all inspected images. The upload box next to it loads such a file back (same image names) to continue "
  "a review.", ["review"])
E("notes", "Can I add notes?",
  ["note", "notes", "comment", "observation", "write a note"],
  "Yes: **Observation or correction** under each position in 🔎 Review every vial. Notes are saved in the review JSON.",
  ["exportreview"])
E("progress", "What is the reviewed counter?",
  ["reviewed counter", "positions reviewed", "progress", "819/840", "how many reviewed"],
  "At the top of 🔎 Review every vial: how many positions have been reviewed in the whole inspection and in this image "
  "(NO VIEW positions are not counted), with a progress bar.", ["review"])

# ===================================================================== I. ROBOT COUNT, LINE MESSAGE, REPORTS
E("count", "What is the robot's placement count (Good placed / Scrap placed)?",
  ["count", "placement count", "good placed", "scrap placed", "placed", "how many vials", "robot count",
   "number of vials", "type the number", "vials placed so far"],
  "The robot fills each plate in a fixed order. Type how many vials it has placed in **Good placed / Scrap placed** "
  "for that image: positions beyond the count are waiting (EMPTY); a position within it with no cap is a defect or a "
  "missed placement. On the held-out frames, empty holes sent to a person: {e_nc}/{e_n} with no help, {e_est}/{e_n} "
  "with the image estimate, {e_cnt}/{e_n} with the count.", ["estimate", "numbering"])
E("line", "What is the line message / PLC JSON?",
  ["line message", "plc", "json", "line_state", "line state", "tell the robot", "send to plc", "machine", "cell",
   "automation", "integration", "mes"],
  "**⬇ Line message (JSON for robot / PLC)**: what the cell should do. **line_state**: HOLD (a defect must leave the "
  "good plate or a missed vial must be re-placed), CHECK (an operator must look) or CONTINUE. Per-position **actions** "
  "and 30-bit **plc_tags** masks (bit n-1 = position n) for reject / review / gap per plate.", ["actions", "masks"])
E("actions", "What do DIVERT_TO_SCRAP, REPLACE_MISSED_VIAL, OPERATOR_CHECK... mean?",
  ["divert", "divert_to_scrap", "replace_missed_vial", "operator_check", "recapture_frame", "log_scrap_defect",
   "action", "actions", "hold", "continue"],
  "- **DIVERT_TO_SCRAP** - a defective vial in the good plate must be moved out.\n- **REPLACE_MISSED_VIAL** - a GAP: "
  "the robot should place a vial there.\n- **OPERATOR_CHECK** - a REVIEW: a person looks.\n- **RECAPTURE_FRAME** - the "
  "robot arm blocked the view.\n- **LOG_SCRAP_DEFECT** - a defect in the scrap plate (already where it belongs).\n"
  "HOLD = any divert/replace; CHECK = only operator checks; CONTINUE = nothing to do.", ["line"])
E("masks", "How do the PLC bit masks work?",
  ["mask", "masks", "bit", "bits", "plc tags", "integer", "16384", "binary"],
  "Each plate has three 30-bit integers - reject, review, gap. Bit n-1 is position n: e.g. G15 rejected → bit 14 → "
  "16384 in G_reject_mask. A PLC can test one bit per position without parsing JSON.", ["line"])
E("reports", "How do I download a report?",
  ["report", "download", "html", "pdf", "print", "record", "manager", "share results", "export results",
   "inspection report", "save results"],
  "Click **📄 Build inspection report (this image)**, then **⬇ Download inspection report (HTML)** - a self-contained "
  "page with the images, every decision and reason, and the line actions. Open it in a browser and print to PDF if "
  "needed.", ["sessionreport", "csv"])
E("sessionreport", "Can I get one report for all images?",
  ["session report", "all images report", "one report", "whole inspection report", "summary of all"],
  "Yes: open **📚 All images in this inspection**, click **📄 Build session report (all inspected images)** and "
  "**⬇ Download session report (HTML)**. **⬇ Download session summary (CSV)** gives one row per image.",
  ["reports", "multi"])
E("csv", "Can I get the results as a CSV / Excel?",
  ["csv", "excel", "spreadsheet", "results csv", "get a csv", "download table"],
  "**📋 All positions → ⬇ Download results (CSV)** (one row per position; opens in Excel). For many images: "
  "**⬇ Download session summary (CSV)**.", ["allpos"])

# ===================================================================== J. ACCURACY & EVIDENCE
E("accuracy", "How accurate is it?",
  ["accurate", "accuracy", "how good", "score", "results", "how well", "test results", "performance", "numbers",
   "catch", "caught", "6/6", "how many defects"],
  "On {t_n} held-out real positions (labelled before seeing any output): **{t_sys}/{t_def} defects caught, "
  "{t_sysfa}/{t_good} good vials wrongly flagged** (the NVIDIA model alone: {t_base}/{t_def} caught, {t_basefa} false "
  "alarms). On {r2_frames} more real frames ({r2_locked} locked): {r2_gf}/{r2_good} good vials flagged, "
  "{r2_df}/{r2_def} defects flagged (the other 2 are one leaning vial, G19, labelled crooked by eye - we count them "
  "as misses). See 📈 Evidence (results per class) and evaluation/RESULTS.md.", ["perclass", "protocol", "limits"])
E("perclass", "What are the results per class (false negatives)?",
  ["per class", "each class", "by class", "false negative", "false negatives", "missed defects", "escapes",
   "escaped", "false alarm rate", "crooked cap results", "missing cap results", "which defects did it miss"],
  "Held-out real frames only, labels as written. **NVIDIA model + rules (6 frames):** missing_cap 3/3, crooked_cap "
  "3/3 caught, 0/69 good vials flagged. **Camera checks only (22 frames):** missing_cap 7/7, crooked_cap 4/7, missed "
  "placement 1/1, 1/134 good flagged, 0/1046 empty holes flagged with the robot count. The 3 misses are one leaning "
  "vial (G19) in 3 frames. No damaged_vial example in the real data. Table: 📈 Evidence.", ["accuracy", "limits"])
E("protocol", "How was it tested? Is the test fair?",
  ["fair", "how was it tested", "held out", "held-out", "test set", "labelled", "labeled", "cheat", "overfit",
   "is the 6/6 real", "validation"],
  "Real frames were labelled by eye **before** seeing the system's output and split: a development set (used to find "
  "and fix error patterns) and **held-out / locked** sets never used to change anything - only scored. Leaning vials "
  "are reported both ways. Nothing was trained on real images.", ["accuracy", "rules"])
E("modelalone", "How does it compare to the NVIDIA model alone?",
  ["model alone", "compared to", "baseline", "without rules", "just the model", "is the nvidia one more accurate",
   "more accurate than local"],
  "On the held-out frames the model alone caught {t_base}/{t_def} defects and flagged {t_basefa} good vials; with our "
  "camera rules and fill order: {t_sys}/{t_def} caught and {t_sysfa} flagged. Local CV only can't judge cap seating "
  "but is instant and still flags missing caps and missed placements.", ["accuracy", "modes"])
E("confidence", "Why not trust the model's confidence (e.g. 90%)?",
  ["confidence", "confident", "90%", "80%", "95%", "percent", "percentages", "probability", "trust the model",
   "just pass it", "high confidence", "sure"],
  "Measured on the labelled real frames: when the model said 80%+ it was wrong 85 of 220 times, at 90%+ 31 of 69, and "
  "**all 15 of its 95-100% answers were wrong** - including open vials called \"pass\" at 80-90%. So no confidence "
  "cut-off is safe. A vial passes when the model **and** the camera agree; it is flagged only when they disagree.",
  ["passrule", "modelalone"])
E("limits", "What are the honest limits?",
  ["can't the system do", "cant the system do", "can't do", "cant do", "what can it not do", "limit", "limits", "limitation", "limitations", "weakness", "weaknesses", "what can't it do", "fail", "failures",
   "not perfect", "sometimes wrong"],
  "- Small real test: 14 held-out defects; labels by eye from 640x480 photos; no damaged vials in the real data.\n"
  "- The cap check looks for the lemon-yellow cap; another colour needs new settings.\n- From one overhead "
  "camera a leaning vial and a tipped cap can look alike → one-click checks.\n- An uncapped clear vial can look like an "
  "empty hole → the robot count settles it.\n- The model's words and confidence are unreliable → physical rules check "
  "it.\n- Not yet run live on the CSI line.", ["accuracy", "lean"])
E("evidence", "What is in the Evidence tab?",
  ["evidence", "📈", "measured", "proof", "cards", "what did you measure"],
  "**📈 Evidence**: defects caught vs the model alone, false alarms on good vials and empty holes (with/without "
  "the count), change-aware savings, auto-align robustness, the 28 extra real frames, auto-fit across camera changes, "
  "results per class (held-out only), whether a twin-only model transfers, and honest limits, plus a Markdown "
  "download for the report. Real-frame numbers come from labelled frames.", ["accuracy", "perclass"])
E("checkacc", "How do I measure accuracy myself?",
  ["measure", "check accuracy", "🎯", "true label", "ground truth", "confusion", "confusion matrix", "label the truth",
   "labels csv", "evaluate", "my own accuracy", "myself"],
  "Open **🎯 Check accuracy**, pick the TRUE label for each position (or load a CSV with position_id,true_label) and "
  "it reports exact matches, defects caught vs missed, false alarms on good vials and a confusion matrix. Label every "
  "position, not just flagged ones, and don't tune settings on the images you report.", ["protocol"])
E("autofitresult", "How well does auto-fit work?",
  ["auto-fit test", "autofit accuracy", "how well does auto fit", "camera changes test"],
  "{af_frames} real labelled frames were each changed {af_var} ways a different camera would (2x resolution, 16:9, "
  "rotated 7°, zoomed out, zoomed in, moved): {af_min}-{af_max}% of decisions identical to the original. Standard "
  "frames are never auto-fitted.", ["autofit"])

# ===================================================================== K. NVIDIA MODEL & API
E("modes", "What is the difference between the two modes?",
  ["mode", "modes", "local cv", "local only", "no api", "without api", "nvidia mode", "vision model mode",
   "difference between the modes", "local vs", "which mode", "internet", "offline", "without internet", "no internet", "without a key",
   "inspect without a key"],
  "Sidebar → **Mode**:\n- **Local CV only (instant, no API)** - camera rules only: caps, empty holes, missed "
  "placements, fill order. Can't judge whether a cap is seated or tilted (shown as CAP). Works offline.\n- **Local CV "
  "+ NVIDIA vision model** - non-empty positions also go to the NVIDIA model (`{model}`), checked against the camera "
  "rules. Needs internet and an API key.", ["apikey", "s_cap"])
E("whichmodel", "Which AI model does it use?",
  ["which llm", "what llm", "llm reads", "reads the images", "which ai reads", "which model", "what model", "llama", "vision model", "vlm", "nim", "ai model", "what ai", "90b"],
  "The NVIDIA NIM vision-language model **`{model}`** (Llama 3.2 90B Vision), used zero-shot through NVIDIA's API - "
  "no training. You can type another NVIDIA model name in the sidebar **Model** box. Vial Helper's optional wording "
  "uses the small text model `meta/llama-3.1-8b-instruct`.", ["zeroshot", "apikey"])
E("zeroshot", "Was the model trained / fine-tuned?",
  ["trained", "fine-tuned", "fine tuned", "finetune", "zero-shot", "zero shot", "prompt", "did you train the model"],
  "No. The NVIDIA model is used **zero-shot**: it gets a cropped close-up with a cyan ring around the target and a "
  "prompt explaining the camera perspective and the labels, and answers in JSON. The lab rules forbid training on real "
  "line images, and the rules + fill order make up for its mistakes.", ["rules", "prompt"])
E("prompt", "What does the model see / what is it asked?",
  ["what does the model see", "crop", "close-up", "cyan ring", "what is it asked", "prompt text", "question to the model"],
  "A square close-up around each position (sidebar **Crop size**, 56 px at 640x480) with a thin cyan ring on the "
  "target hole. It is told that neighbouring caps don't count, that perspective shifts a seated cap, asked two checks "
  "(opening visible? cap tilted?) and one of the five labels with a confidence and a short reason, as JSON.",
  ["whichmodel", "secondlook"])
E("apikey", "Where does the NVIDIA API key go?",
  ["api key", "key", "nvapi", "where do i put", "paste key", "key box", "set key", "nvidia_key.txt", "my key"],
  "Paste it into the sidebar box **NVIDIA API key** (shown in model mode). It isn't saved in the project files. "
  "**run_app.bat** can pre-fill it from nvidia_key.txt in the app folder - never share that file (it's in .gitignore). "
  "Get a key at build.nvidia.com.", ["credits", "keysafe"])
E("credits", "I ran out of credits / is it free?",
  ["credits", "ran out", "out of credits", "quota", "free", "cost", "money", "charge", "pay", "402", "429",
   "rate limit", "another model", "different model", "free model", "stopped working", "key stopped"],
  "NVIDIA gives free trial credits per account; they are shared by all models, so switching models doesn't help when "
  "they run out. Options: a teammate's own key, the saved answers for demo_images (no calls), or **Local CV only** "
  "mode. A 429 error means too many requests at once - lower **Parallel requests**.", ["modes", "savedanswers"])
E("keysafe", "Is my key safe? I pasted it somewhere.",
  ["safe", "security", "leak", "leaked", "share key", "revoke", "github", "commit key", "exposed"],
  "Keep the key only in the sidebar box (or nvidia_key.txt, which is in .gitignore). Never put it in files you share "
  "or in GitHub. If it was exposed, revoke it at build.nvidia.com and create a new one. The project zips contain no "
  "key.", ["apikey"])
E("othermodel", "Can I use a different model?",
  ["different model", "change model", "another vision model", "gpt", "other llm", "swap model", "model box"],
  "Type another NVIDIA NIM vision model name in the sidebar **Model** box (it must accept images). Saved answers only "
  "apply to the default model. All accuracy numbers were measured with `{model}`.", ["whichmodel"])

# ===================================================================== L. DIGITAL TWIN & ISAAC SIM
E("twin", "What is the digital twin / NVIDIA Isaac Sim part?",
  ["digital twin", "twin", "isaac", "isaac sim", "synthetic", "simulation", "sim", "render", "fake images",
   "generated", "synthetic data", "🧪"],
  "We rendered the CSI digital twin in **NVIDIA Isaac Sim** through its own CapSealCam with random fill, missing "
  "caps, tipped caps, leaning vials, missed placements and lighting - {tw_pos} automatically labelled positions in "
  "the first set. It found weaknesses the real photos never showed (dim light, leaning vials rejected as missing caps), "
  "fixed without changing any real-frame result. See **🧪 Digital twin**.", ["twinrun", "learnloop", "twintab"])
E("twintab", "How do I use the Digital twin tab?",
  ["digital twin tab", "synthetic frame", "ground truth", "twin tab", "lean_synth", "synth_0054"],
  "**🧪 Digital twin** (also on the start page): pick a **Synthetic frame**; left = the twin's true labels, right = "
  "our checks run live on it (no API). Switch **Give the checks the robot's placement count** on/off to see the "
  "difference. lean_synth_… frames contain leaning vials.", ["twin"])
E("twinrun", "How do I generate my own synthetic images?",
  ["generate", "run_synth", "resume_synth", "make synthetic", "render images", "isaac run", "--n", "seed"],
  "With Isaac Sim installed (**isaac\\setup_isaac.bat**), run `isaac\\run_synth.bat --n 150 --seed 11` (new set) "
  "or `isaac\\resume_synth.bat --n 150 --seed 11` (continue). Leave the window open: it restarts Isaac Sim after a "
  "driver crash. Output: isaac\\synth.zip with images + labels.json.", ["twin", "twincrash"])
E("twincrash", "Isaac Sim keeps crashing",
  ["crash", "crashes", "access violation", "driver", "assertion failed", "being used by another process",
   "isaac crashed", "stopped early"],
  "The NVIDIA driver occasionally crashes while rendering (RTX 50 series). The generator saves after every image and "
  "the window restarts it; an image that crashes twice is skipped. If you see \"being used by another process\", close "
  "old windows (or restart the PC) and run **resume_synth.bat**. Click Ignore on an \"Assertion failed\" box.",
  ["twinrun"])
E("learnloop", "Does it learn from its mistakes?",
  ["learn", "learning", "learns", "smarter", "mistakes", "learning loop", "machine learning", "retrain", "improve",
   "get better", "more photos", "self-improving"],
  "On the digital twin, yes: **learn_loop.py** trains a twin-only model, tests it on held-out twin frames, finds its "
  "mistakes and renders more of them (false alarms fell from 150 to about 10 over 4 rounds). On the real camera that "
  "model still flags most good vials (the twin looks too clean), so it is **not** used for decisions. Real photos are "
  "never trained on (lab rule).", ["rules", "sim2real"])
E("sim2real", "Why doesn't the twin-trained model work on real photos?",
  ["sim2real", "sim to real", "domain gap", "transfer", "doesn't transfer", "too clean"],
  "The twin's empty holes are dark, the real ones bright and brown; its plate, glass and caps look cleaner than the "
  "real camera. A model trained only on twin images therefore flags most real good vials. The twin is used to "
  "**test and improve rules**, which do carry over.", ["learnloop"])

# ===================================================================== M. PROJECT & RULES
E("rules", "Does it follow the lab rules (no training on real data)?",
  ["rule", "rules", "lab rule", "allowed", "compliance", "real data", "train on real", "trained on our",
   "our pictures", "our photos", "train it on", "is it allowed"],
  "Yes. The lab says real line data is *there to look at, not to train on*. Nothing is trained on real images: the "
  "NVIDIA model is zero-shot, the camera checks are fixed physical rules, and the only trained model was trained on "
  "synthetic Isaac Sim frames. Real frames were labelled by eye and used only to test.", ["protocol", "zeroshot"])
E("about", "What is this project?",
  ["what is this", "about", "project", "what does it do", "what does the website do", "overview", "purpose",
   "summary of the project", "vial inspector", "hackathon"],
  "Vial Inspector checks the CSI testbed's final vial inspection camera (Dataset 1): it finds all 60 fixture "
  "positions, judges each one (pass, missing cap, crooked cap, damaged vial, no vial) with camera rules + the NVIDIA "
  "vision model, explains every decision, tells the robot/PLC what to do, and was stress-tested in an NVIDIA Isaac Sim "
  "digital twin. A missed defect is treated as worse than a false alarm.", ["unique", "accuracy"])
E("unique", "What makes this project different?",
  ["unique", "different", "special", "stand out", "why is it good", "innovation", "novel", "judges"],
  "- Physical rules **check the AI** (the model alone caught {t_base}/{t_def} defects; with the rules {t_sys}/{t_def}).\n"
  "- Every decision is explained, with the official label definitions.\n- One-click checks for the cases one camera "
  "can't settle.\n- Closes the loop with the cell: line message + PLC masks.\n- Any camera via auto-fit; "
  "change-aware ({ch_saved} fewer model calls).\n- Stress-tested and improved in an NVIDIA Isaac Sim digital twin.",
  ["about", "accuracy"])
E("value", "Why does this matter for a factory?",
  ["factories", "factory", "matter", "value", "impact", "business", "why does it matter", "factory", "manufacturing", "benefit", "roi", "broader"],
  "A missed missing or crooked cap can reach a customer; a person checking every vial is slow and tiring. The system "
  "checks all 60 positions in seconds, only asks a person about the genuinely uncertain ones (one click each), tells "
  "the robot what to do, and leaves a traceable record (reports, review JSON).", ["unique"])
E("generalize", "Would it work for other products or lines?",
  ["different cap color", "different cap colour", "other cap color", "different cap", "other caps", "other products", "generalize", "generalise", "other lines", "other caps", "other colors", "reuse", "scale",
   "another factory"],
  "The pieces are general: grid fitting from a hole pattern, colour-based cap checks, an AI model checked by physical "
  "rules, fill-order logic and the line message. A new product needs its cap colour range and grid layout, and the "
  "same honest test (labelled frames, locked test set).", ["unique"])
E("team", "Who made it?",
  ["built this", "who built", "made this", "who made this", "who made", "team", "authors", "creators", "built by"],
  "Built for the Rockwell Automation + NVIDIA Innovation Lab hackathon (Dataset 1 - final vial inspection) by Malik "
  "Bennani, Abdullahi Oyekanmi, Markos Lezama and Oluwatobi Odekunle.", ["about"])

# ===================================================================== N. TROUBLESHOOTING
E("troubleshoot", "Something isn't working",
  ["not working", "broken", "doesn't work", "doesnt work", "problem", "issue", "bug", "help me", "stuck",
   "nothing happens", "wrong"],
  "Common fixes:\n- **Circles off** → 🎯 Auto-align; check 🧭 Auto-fit is on.\n- **Inspect button greyed out** → tick "
  "**Alignment looks correct** (and add an API key in model mode).\n- **Changes not showing** → stop (Ctrl+C) and start "
  "the website again.\n- **Model errors** → those positions are REVIEW; check key / credits.\n- **\"No module named …\"** "
  "→ `python -m pip install -r requirements.txt`.\nAsk me about the specific problem for more.", ["start", "greyed"])
E("greyed", "The Inspect button is greyed out",
  ["greyed out", "grayed out", "disabled", "can't click", "cant click", "button not working", "inspect disabled"],
  "Tick **Alignment looks correct** in Step 1 first. In model mode the sidebar also needs an **NVIDIA API key**. If the "
  "image is already fully inspected, the button is off because there is nothing left to do (the success note says "
  "so).", ["inspect", "apikey"])
E("disappeared", "My results disappeared",
  ["disappeared", "results gone", "lost results", "results went away", "have to inspect again", "vanished"],
  "Results stay with each image when you flip between images. They are cleared only for an image whose grid you moved "
  "after inspecting (re-align → inspect again), after **➕ New inspection**, or after restarting the website.",
  ["realign", "flip"])
E("wontstart", "The website won't start / the page doesn't load",
  ["won't start", "wont start", "doesn't load", "not loading", "site isnt loading", "blank page", "page not found",
   "connection refused", "can't reach", "localhost refused"],
  "Check the black window: if it shows an error, read the last lines. \"No module named …\" → install the "
  "requirements. \"Address already in use\" → another copy is running; close it or open the URL it shows. If the "
  "window closed straight away, start it with `cmd /k` (the command keeps the window open so you can read the error).",
  ["start", "install"])
E("portinuse", "It says the port is already in use",
  ["port", "already in use", "8502", "address already in use", "two copies"],
  "Another copy of the website is already running. Use the URL it prints (often http://localhost:8502), or close the "
  "other black window first.", ["wontstart"])
E("wrongresult", "The result is wrong for a vial - what should I do?",
  ["wrong result", "incorrect", "wrong answer", "false alarm", "false positive", "missed defect", "false negative",
   "it rejected a good", "flagged a good vial", "passed a bad", "not accurate", "accuracy is off"],
  "1. Open the position in **🔎 Review every vial** and read **Why the system decided this** - it shows which rule or "
  "model answer decided.\n2. Correct it with **Visible vial class** (it's logged as your decision).\n3. If one rule is "
  "consistently wrong, export the review JSON / line message and note the image - that is how we found and fixed the "
  "G19 and second-look issues.", ["whyflag", "classdrop"])
E("warnsize", "It warns about the image size",
  ["warning", "warns", "4:3", "screenshot warning", "image is", "aspect warning"],
  "With auto-fit on, other sizes are straightened automatically. With auto-fit off, the default grid assumes a 4:3 "
  "640x480 frame, so use the raw camera frame (not a screenshot) and check the circles.", ["autofit"])

# ===================================================================== O. WHY / EXPLAIN & HELPER
E("whyflag", "Why was a position flagged or passed?",
  ["reason", "explain", "flagged", "decided", "decision", "why flagged", "why was it flagged", "reasoning",
   "why did it flag", "why did it reject", "why did it pass", "why was it rejected"],
  "Ask me about the position directly, e.g. *\"why is G15 flagged?\"* - I answer from the current inspection. Or open "
  "**🔎 Review every vial**, pick it, and read **Why the system decided this**: cap coverage, what the model said, any "
  "second look, and the rule that decided.", ["passrule", "s_review"])
E("helper", "What can you (Vial Helper) do?",
  ["what can you do", "who are you", "vial helper", "helper", "chatbot", "bot", "assistant", "are you an ai",
   "is there an llm"],
  "I answer questions about using the website, what the results and labels mean, how decisions are made, accuracy, "
  "the NVIDIA model, the digital twin and troubleshooting - from checked notes. After an inspection I can explain any "
  "position (*\"why is G15 flagged?\"*) or summarise the image (*\"what was flagged?\"*). With an API key, an NVIDIA "
  "text model can write my replies, but only from the checked notes.", ["howto"])
E("helperacc", "How accurate is Vial Helper?",
  ["helper accuracy", "how accurate are you", "can i trust you", "do you make mistakes"],
  "I only answer from checked notes and the actual inspection, so I don't invent facts; if I'm unsure I say so and "
  "offer topics. I can pick the wrong topic for an unusual wording - the \"Not what you meant?\" buttons offer the "
  "next best ones. evaluation/helper_tests.py measures this.", ["helper"])

# ===================================================================== P. PRACTICAL EXTRAS
E("gpu", "Do I need a GPU or a special computer?",
  ["gpu", "graphics card", "hardware", "computer requirements", "laptop", "powerful", "rtx", "specs"],
  "No GPU is needed for the website: the camera checks run on the CPU and the NVIDIA model runs on NVIDIA's servers. "
  "Only the optional Isaac Sim digital-twin rendering needs an NVIDIA RTX GPU.", ["install", "twinrun"])
E("os", "Does it work on Mac or Linux?",
  ["macbook", "apple", "windows", "mac", "macos", "linux", "ubuntu", "operating system", "windows only", "other os"],
  "The website itself is plain Python and runs anywhere: in the app folder run `python -m streamlit run app.py` (after "
  "installing requirements.txt). The .bat files are Windows shortcuts, and the Isaac Sim scripts were set up for "
  "Windows.", ["install", "start"])
E("phone", "Can I open it on my phone or another computer?",
  ["phone", "tablet", "ipad", "another computer", "other computer", "network url", "share with team", "remote",
   "same wifi", "teammates"],
  "When the website starts, the black window also prints a **Network URL**; devices on the same network can open it "
  "(firewall permitting). To give the project to teammates, share the zip or GitHub repo - never the key file.",
  ["start", "keysafe"])
E("downloads", "Where do downloaded files go?",
  ["where do downloads go", "downloaded file", "where is the file", "saved where", "download folder", "where did it save"],
  "Downloads (reports, CSV, JSON) go to your browser's normal **Downloads** folder. The website does not write files "
  "into the app folder (only batch_eval.py writes runs/ and vlm_cache.json).", ["reports"])
E("privacy", "What data is sent to NVIDIA?",
  ["privacy", "data sent", "upload to nvidia", "cloud", "is my data", "what is sent", "confidential", "send images"],
  "Only in model mode, and only the small close-up crops of positions that need the model (not the whole photo) are "
  "sent to NVIDIA's API with the prompt. Local CV only mode sends nothing anywhere. Vial Helper's optional wording sends "
  "your question and the checked answer text.", ["modes", "prompt"])
E("streamlit", "What is Streamlit / what is the website built with?",
  ["language", "written in", "programming language", "coded in", "streamlit", "built with", "tech stack", "technology", "framework", "python", "opencv", "code"],
  "Python: **Streamlit** for the website, **OpenCV / NumPy / Pillow** for the camera checks, the **NVIDIA NIM API** for "
  "the vision model, and **NVIDIA Isaac Sim** for the digital twin. Main files: app.py (website), vial_core.py "
  "(inspection logic), ui.py (look), helpbot.py + helpbot_kb.py (Vial Helper).", ["about"])
E("allimages", "What is the 📚 All images section?",
  ["all images", "📚", "side by side", "summary table", "all images in this inspection"],
  "**📚 All images in this inspection** (below the results when you have several images) shows every inspected image "
  "side by side with its overlay and a summary table (counts per image), plus the session report and session CSV.",
  ["multi", "sessionreport"])
E("decidedby", "What does \"decided by\" mean on a card?",
  ["decided by", "source", "vlm source", "conflict", "fill pattern source", "operator source"],
  "Which part made the final call: **vlm** (the model, checked by the camera), **conflict** (model and camera disagreed "
  "→ REVIEW), **second look**, **local** (camera rules only), **placement count / fill pattern** (the robot's fill "
  "order), **operator** (a one-click or review decision), **geometry** (NO VIEW), **error** (model unavailable).",
  ["whyflag", "pipeline"])
E("modeltime", "What does \"model time per position\" mean?",
  ["model time", "per position", "seconds per position", "positions sent to the model", "latency", "⏱"],
  "The note under the overlay: the average time the NVIDIA model took to answer one position, how many positions were "
  "sent to it, and the mode (balanced / strict). Several positions are asked in parallel, so an image takes far less "
  "than time x positions.", ["speed", "calls"])
E("sharecode", "How do we share or submit the project?",
  ["submit", "submission", "github", "share the project", "send the code", "zip", "repository", "hand in"],
  "Share the latest **vial_inspector_vNN.zip** (it contains no API key) or push the folder to GitHub - but never "
  "nvidia_key.txt, vlm_cache.json or runs/ (they are in .gitignore). The README explains how to run and reproduce "
  "everything.", ["keysafe", "about"])
