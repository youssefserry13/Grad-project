# Level 2 dry run: real lab photos through the camera pipeline (2026-11-24)

18 photos of the lab's Shizuoka vertical milling machine (`C:\Grad\dataset\level2\processed_1024\REAL_MILL_*.jpg`,
not committed) run through the same camera pipeline `/live` uses, called directly (no audio):
single-model mode (qwen3-vl:8b-instruct, ctx 4096), `image_pipeline.preprocess` (1024 px) +
`run_mode_b2_look`, max 256 tokens, as in the bridge. Bridge stopped during the run.

- **Time** = camera pipeline only (model warm, run in sequence like live use). A spoken question adds Whisper (~1–2 s).
- **Route**: `manual p.N` = answered from the manual excerpts (B2), `general` = image-only answer. Every answer here went to the manual.
- **Check** is a loose manual comparison with `ground_truth_machine.csv` (it describes what each photo shows) and,
  for the specific questions, with the manual text: **OK** / **partly** / **wrong**.

| Photo | Question | Answer (Nova) | Route | Time | Check |
|---|---|---|---|---|---|
| 001 | What is this? | That's the quill vertical feed handwheel, used to move the main spindle up and down manually. Turn it right to move up and left to move down. | manual p.16 | 7.8 s | partly: a handwheel, yes; which one can't be confirmed |
| 002 | What is this? | This is a vertical milling machine, specifically the SV-CH or SV-F model, as shown in the manual's cover and main part diagrams. | manual p.1 | 9.7 s | partly: right machine, misses the handwheel / lever in focus |
| 003 | What is this? | This is a vertical milling machine, specifically the SV-CH or SV-F model, as shown in the manual's cover. | manual p.1 | 9.0 s | partly: misses spindle + end mill + vise |
| 004 | What is this? | This is the vertical rapid feed lever for the SV-CH or SV-F milling machine, used to move the knee up or down quickly. | manual p.60 | 10.3 s | partly: UP/DN fits knee up/down, but it is a handwheel, not a lever |
| 005 | What is this? | That's the table vertical feed handle, used to control the vertical movement of the table. | manual p.16 | 11.5 s | partly: a crank handle under the saddle, plausible |
| 006 | What is this? | This is the SV-CH or SV-F vertical milling machine, as shown in the manual's cover. | manual p.1 | 7.5 s | **wrong**: it is the machine vise |
| 007 | What is this? | That's the automatic feed change knob for the SV-F head, used to switch between automatic and manual feed modes. | manual p.20 | 8.2 s | partly: a dial/knob, identity unverified (dial is on the base, not the head) |
| 008 | What is this? | This is the automatic feed change knob for the SV-F head, used to set feed rates. | manual p.20 | 10.0 s | **wrong**: it is the spindle speed selector plate |
| 008 | What spindle speeds can I select? | You can select spindle speeds from 90 to 1,400 revolutions per minute in nine steps. | manual p.51 | 7.2 s | partly: manual fact (p.11 spec "9 steps: 90~1400"), page cited wrong; the photo's dial shows 180–990 |
| 009 | What is this? | That's the table vertical feed handle, used to control the vertical movement of the table. | manual p.16 | 9.9 s | **wrong**: long lever + H/L range drum |
| 010 | What is this? | That's the automatic feed change knob for the SV-F head, used to switch between automatic and manual feed modes. | manual p.20 | 11.2 s | partly: dial with feed markings, identity unverified |
| 011 | What is this? | This is the table feed speed change handwheel, used to set the feed rate for the table. | manual p.22 | 8.4 s | partly: handwheel + range drum, plausible |
| 012 | What is this? | That's the quill vertical manual feed lever, used to manually move the quill vertically. | manual p.20 | 9.9 s | partly: T-handle on the column, unverified |
| 013 | What is this? | This is a vertical milling machine, specifically the SV-CH or SV-F model, as shown in the manual's cover. | manual p.1 | 11.1 s | OK |
| 013 | What machine is this? | This is a vertical milling machine, specifically the SV-CH or SV-F model as shown in the manual. | manual p.1 | 4.0 s | OK |
| 014 | What is this? | This is the quill automatic/manual switch lever for the SV-F machine, used to select between manual and automatic feed modes. | manual p.53 | 9.2 s | **wrong**: it is the lubrication pump |
| 014 | How do I use the lubrication pump? | Pull the lever two or three times every hour to lubricate the slideways. Fill the tank through the oil supply port when the oil level is low. | manual p.34 | 6.7 s | OK |
| 015 | What is this? | This is a vertical milling machine, specifically the SV-CH or SV-F model, as shown in the manual's cover and main part diagrams. | manual p.1 | 13.2 s | partly: right machine, misses crank lever / handwheel |
| 016 | What is this? | This is part of the machine's control system, likely related to manual feed or movement adjustments as described in the manual. | manual p.36 | 9.8 s | partly: vague, hedged |
| 017 | What is this? | That's the coolant lubrication system, which uses a pump to circulate coolant oil for cutting. The pump is 100 watts and can deliver up to 10 litres per minute. | manual p.43 | 10.8 s | OK (coolant line / nozzle) |
| 017 | How do I turn on the coolant? | Turn the coolant ON/OFF switch on the operation panel to the ON position. The red pilot lamp will light up when the pump is running. | manual p.49 | 5.1 s | OK (the yellow ball valve on the line is not mentioned) |
| 018 | What is this? | This is the operation panel for the milling machine, showing buttons to control the main spindle, feed motor, and vertical rapid feed. | manual p.49 | 10.0 s | OK |
| 018 | What does the red button do? | The red mushroom-shaped button is the emergency stop button, which stops all machine movements instantly. | manual p.45 | 5.2 s | OK (panel also has red OFF buttons) |

## Summary

| | OK | partly | wrong | Time (median) |
|---|---|---|---|---|
| "What is this?" (18) | 3 | 11 | 4 | 9.9 s (7.5–13.2) |
| Specific questions (5) | 4 | 1 | 0 | 5.2 s (4.0–7.2) |

- **Specific questions work; "What is this?" on close-ups does not.** A vague question has no words to search with, so
  the description of the photo drives retrieval, and the model then names *a* part from the retrieved excerpts with a
  page, confidently, even when it is the wrong part (vise → "the machine", speed plate / lube pump → feed knob / quill lever).
  The manual has few photos of the real parts, so a close-up rarely matches the right excerpt.
- **No general answers**: every photo shows the machine, so the manual route always produced an answer (none fell back to "general").
- **Specific questions are about 2x faster** (one vision call, no describe step): 4–7 s vs 8–13 s.
- **The lab machine is not exactly the manual's machine**: the speed plate in 008 shows 180–990, the manual's
  spec (p.11) says 90–1400 min-1 in 9 steps. Avoid quoting numeric specs while pointing at the real machine.
- Page citations are sometimes off (008: numbers from p.11, cited p.51; "manual's cover" p.1 for the whole machine).

## Safe demo questions (correct and fastest)

| Photo / camera on | Question | Time |
|---|---|---|
| 013 whole machine | What machine is this? | 4.0 s |
| 017 coolant nozzle / panel | How do I turn on the coolant? | 5.1 s |
| 018 push-button panel (emergency stop in view) | What does the red button do? | 5.2 s |
| 014 lubrication pump | How do I use the lubrication pump? | 6.7 s |
| 018 push-button panel | What is this? | 10.0 s |
| 017 coolant line | What is this? | 10.8 s |

**Avoid**: "What is this?" on close-ups of the vise (006), speed selector (008), H/L range lever (009) or lubrication
pump (014); they are named as the wrong part with a page. Avoid numeric spec questions about the real machine
("What spindle speeds can I select?" quotes the manual's 90–1400 while the dial shows 180–990).
