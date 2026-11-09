# Dota 2 typing minigame bot (macOS)

A bot for the typing minigame in Dota 2's **Dark Carnival** event. It watches the Dota window, reads the hero and item
names as they fly in, and types them with real key presses. Best score so far: **17,230**.

Built end to end with Claude Code (an AI coding agent); I didn't write the code by hand.

## How it works
1. **Watch:** ScreenCaptureKit streams only the Dota window at 60 fps (scaled to 960 px wide).
2. **Read:** Apple Vision OCR finds the text and where each word is.
3. **Decide:** it checks the colour of every pixel inside each word: white = still to type, yellow = the cursor letter,
   grey = already typed. The word that is already started goes first, then the ones about to land, then the oldest.
   A list of known names (`words.txt`, grows every run) fixes misreads; GPT-4.1-mini helps spell odd phrases.
4. **Type:** System Events key presses, so Dota registers them. It only types while the minigame is on.

## Requirements
- A Mac (macOS 14+)
- Swift compiler: `xcode-select --install`
- Screen Recording and Accessibility permission for your terminal (System Settings > Privacy & Security)
- An OpenAI API key (optional: run with `NOGPT=1` to skip GPT)

## Run
```sh
cp .env.example .env        # put your key(s) in .env, it is git-ignored
swiftc -O -swift-version 5 dotatyper.swift -o dotatyper
./dotatyper                 # then open the minigame in Dota 2
```
Options: `NOGPT=1` (no GPT), `DRY=1` (log what it would type, type nothing), `DEBUG=1` (save frames to `debug/`),
`JEV=1` (use the TypeSafe AI Jev API to tell when the game is on; needs `JEV_API_KEY`). Ctrl+C to stop.
It prints the API cost per session when you leave Dota.

## Notes
- Calibrated on one screen layout; if words are missed, adjust the knobs at the top of `dotatyper.swift` (`ROI`, `DANGER`).
- Screen recording at the same time slows it down.
- Made for fun. Automating input may break game rules; use at your own risk.
