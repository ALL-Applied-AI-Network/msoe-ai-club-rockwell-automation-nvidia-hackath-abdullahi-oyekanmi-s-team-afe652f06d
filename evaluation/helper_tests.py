"""helper_tests.py - tests for Vial Helper (helpbot.py + helpbot_kb.py, 130 checked topics).
TUNE   (127 questions): used while building the knowledge base.
LOCKED (50 questions):  written after tuning and first scored ONCE - first score: 33/50 direct, 44/50 answered or
                        offered as the first "Not what you meant?" button, 6 wrong, 4/6 off-topic declined.
                        The 6 misses were real knowledge gaps and were then fixed, so later scores on it are optimistic.
Each question lists the acceptable topic ids.     python evaluation/helper_tests.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import helpbot as hb
TUNE = [
 ("how do i use the model", "howto"), ("how does this work", "howto pipeline"), ("im new what do i do first", "howto"),
 ("give me a tutorial", "howto"), ("how do i start the website", "start"), ("whats the command to run it", "start"),
 ("where is the url", "start"), ("how do i close the app", "stop"), ("i updated the files but nothing changed", "update stop"),
 ("what do i pip install", "install"), ("it says no module named streamlit", "install wontstart"),
 ("which images should i show the judges", "demo"), ("what does run_app.bat do", "runbat"),
 ("how do i upload a photo", "upload"), ("can i use png files", "formats"), ("how do i go to the next image", "flip"),
 ("can i check 20 photos at once", "multi"), ("how do i clear all images", "newinsp"), ("does photo order matter", "order changeaware"),
 ("the circles are off the holes", "align"), ("what does the auto align button do", "autoalign align"),
 ("what does reset do", "reset"), ("my photo is from a different camera", "autofit"), ("can it handle 1920x1080", "autofit"),
 ("what do the hole match numbers mean", "gauges"), ("can i move the grid corners manually", "calib"),
 ("which one is G01", "numbering"), ("how are positions numbered", "numbering"), ("what is the scrap fixture", "fixtures"),
 ("why do i have to inspect again after aligning", "realign"), ("how do i inspect", "inspect howto"),
 ("how many api calls per image", "calls"), ("why is it so slow", "speed"), ("what are saved answers", "savedanswers"),
 ("what is batch_eval", "batchtest"), ("what do the colors mean", "statuses"), ("what does pass mean", "s_pass l_pass"),
 ("what is the blue CAP status", "s_cap"), ("what does reject mean", "s_reject"), ("what does the orange ? mean", "s_review statuses"),
 ("what is a gap", "s_gap"), ("what does empty mean", "s_empty"), ("why is S03 no view", "s_noview"),
 ("what does 2 defective vials found mean", "verdict"), ("what is the fill progress bar", "fillstrip"),
 ("what is the fixture map", "map"), ("where do i see pictures of flagged vials", "needs"), ("what are the tabs", "tabs"),
 ("where is the table of results", "allpos csv"), ("what are the official labels", "labels"), ("what is a missing cap", "l_missing"),
 ("what is a crooked cap", "l_crooked"), ("can it detect cracks", "l_damaged"), ("what does no_vial mean", "l_novial"),
 ("is an empty hole a defect", "whatdefect l_novial s_empty"), ("how does it decide", "pipeline whyflag"),
 ("what is needed to pass", "passrule s_pass"), ("what is cap over hole", "coverage"), ("how does it see the yellow cap", "yellow"),
 ("does it work in dark lighting", "dimlight"), ("what is the second look", "secondlook"),
 ("how does it know an empty hole from an open vial", "emptyvsopen"), ("what is the fill estimate", "estimate"),
 ("what if the model times out", "modelerror"), ("what does strict mode do", "strict"), ("should i skip empty holes", "skipempty"),
 ("what is change aware", "changeaware"), ("can i turn off the S01-S05 rule", "hiddenrule"), ("the robot arm is in my photo", "robotarm"),
 ("why does it flag leaning vials", "lean"), ("is a leaning vial a defect", "lean"), ("what does just leaning do", "quick lean"),
 ("why does it ask about the hole after the last cap", "margin"), ("i clicked the wrong button", "undo"),
 ("how do i review every vial", "review"), ("how do i get to the next unchecked vial", "nextun"),
 ("how do i change a vial's class", "classdrop"), ("what is the dashed ring", "dashed"), ("can i approve all passes", "confirmall"),
 ("how do i save my review", "exportreview"), ("can i write notes", "notes"), ("what is the reviewed counter", "progress"),
 ("what number do i put in good placed", "count"), ("what is the line message", "line"), ("what does DIVERT_TO_SCRAP mean", "actions"),
 ("how do the plc masks work", "masks"), ("how do i download a report", "reports"), ("one report for all images", "sessionreport"),
 ("can i export to excel", "csv"), ("how accurate is it", "accuracy"), ("was the testing fair", "protocol"),
 ("is it better than the nvidia model alone", "modelalone"), ("why not trust 90 percent confidence", "confidence"),
 ("what are the limitations", "limits"), ("what is in the evidence tab", "evidence"), ("how do i check accuracy myself", "checkacc"),
 ("how well does auto fit work", "autofitresult"), ("local cv vs nvidia", "modes"), ("which ai model is used", "whichmodel"),
 ("did you fine tune the model", "zeroshot"), ("what does the model see", "prompt"), ("where do i put my api key", "apikey"),
 ("i ran out of credits", "credits"), ("is my api key safe", "keysafe"), ("can i use gpt instead", "othermodel"),
 ("what is the digital twin", "twin"), ("how do i use the digital twin tab", "twintab"), ("how do i make synthetic images", "twinrun"),
 ("isaac sim crashed", "twincrash"), ("does it learn from mistakes", "learnloop"), ("why doesnt the twin model work on real", "sim2real"),
 ("are we breaking the lab rules", "rules"), ("what is this project", "about"), ("what makes it unique", "unique"),
 ("why does this matter for factories", "value"), ("can it work on other products", "generalize"), ("who built this", "team"),
 ("it is not working", "troubleshoot"), ("the inspect button is greyed out", "greyed inspect"), ("my results disappeared", "disappeared"),
 ("the site wont load", "wontstart"), ("port already in use", "portinuse"), ("it rejected a good vial", "wrongresult"),
 ("why was it flagged", "whyflag"), ("what can you do", "helper"), ("are you accurate", "helperacc"),
 ("what settings are there", "settings"),
]
LOCKED = [
 ("how do i get going with this tool", "howto start"), ("can you explain how to use the inspector", "howto"),
 ("the black window closed right away", "wontstart start"), ("what python packages are needed", "install"),
 ("can i inspect jpeg photos from my phone", "formats upload"), ("how do i look at the previous photo", "flip"),
 ("i want a new batch of photos, how do i reset", "newinsp"), ("the grid is shifted down", "align"),
 ("my image is 4k from another camera", "autofit formats"), ("what are the numbers 27/30 next to the grid", "gauges"),
 ("where is G26 on the plate", "numbering"), ("what is the difference between G and S", "fixtures"),
 ("i dont have credits can i still inspect", "modes credits"), ("why does inspecting cost api calls", "calls"),
 ("what does the red x on a vial mean", "s_reject statuses"), ("what does a purple exclamation mean", "s_gap statuses"),
 ("when is a vial passed", "passrule s_pass"), ("what percentage of the hole must the cap cover", "passrule coverage"),
 ("will it still find caps if the light is weak", "dimlight yellow"), ("what happens when the ai contradicts the camera", "secondlook"),
 ("how does it know the robot skipped a spot", "s_gap estimate count"), ("why is it asking me about an empty hole", "margin quick"),
 ("what does the uncapped vial button do", "quick"), ("how do i undo my click", "undo"),
 ("how can i override the verdict for a position", "classdrop"), ("what does the white circle around a position mean", "dashed"),
 ("can i continue my review tomorrow", "exportreview"), ("where do i type the robot count", "count"),
 ("what does line_state CHECK mean", "line actions"), ("what is bit 14 in the reject mask", "masks"),
 ("i need a pdf of the inspection", "reports"), ("how many defects did it miss in testing", "accuracy"),
 ("were the test labels made before seeing the results", "protocol"), ("why dont you use the confidence score", "confidence"),
 ("what can't the system do", "limits"), ("which llm reads the images", "whichmodel"), ("was the ai trained on our line", "zeroshot rules"),
 ("someone saw my api key what do i do", "keysafe"), ("what does isaac sim add to this project", "twin"),
 ("how do i render more twin images", "twinrun"), ("will it get better over time", "learnloop"),
 ("why should the judges care", "unique value"), ("could this work for a different cap color", "generalize"),
 ("nothing shows up after i click inspect", "troubleshoot greyed"), ("it flagged a perfectly good vial", "wrongresult lean"),
 ("does it run on a macbook", "os"), ("is the photo uploaded to the cloud", "privacy"), ("do i need a graphics card", "gpu"),
 ("what language is it written in", "streamlit"), ("how do we hand in the project", "sharecode"),
]
OFF_TOPIC = ["whats the weather", "tell me a joke", "who won the super bowl", "how do i bake bread", "what is the capital of japan",
       "recommend a song", "solve 12 times 7", "what's your favorite color" + "how tall is mount everest", "write a haiku about cats", "what time is it in tokyo", "who is the president",
       "give me a pancake recipe", "what's 15 percent of 80"]


def run(name, items):
    ok = alt = 0
    for q, want in items:
        wants = want.split()
        a = hb.answer(q)
        if a.get("kb_id") in wants:
            ok += 1
        elif any(hb.KB_BY_ID[w]["title"] in a.get("related", []) for w in wants):
            alt += 1
    print(f"{name}: {ok}/{len(items)} direct, {alt} more offered as a button, {len(items) - ok - alt} wrong")


if __name__ == "__main__":
    print(f"{len(hb.KB)} topics")
    run("TUNE", TUNE)
    run("LOCKED (first score was 33 direct / 44 reachable)", LOCKED)
    print(f"off-topic declined: {sum(not hb.answer(q)['confident'] for q in OFF_TOPIC)}/{len(OFF_TOPIC)}")
