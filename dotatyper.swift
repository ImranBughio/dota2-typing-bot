// Dota 2 typing-minigame bot (macOS): 60fps window stream -> Apple Vision OCR -> type the highlighted word.
// Build:  swiftc -O -swift-version 5 dotatyper.swift -o dotatyper
// Run:          ./dotatyper                 Test on an image:  ./dotatyper shot.png
// Key method:   KEYMODE=se (default, System Events, ~230ms/letter, proven)  or  KEYMODE=cg:10 (raw key events, 10ms apart)
// Jev:          JEV=1 ./dotatyper           Jev (TypeSafe AI) decides "is the minigame on?" instead of the TIME label
// Debug:        DEBUG=1 ...                 saves every frame it typed from into debug/
// Dry run:      DRY=1 ./dotatyper           watches + logs what it would type, types nothing, Dota needn't be in front
// Every run logs to dotatyper.log and learns finished words into words.txt (used to fix misreads).
import AppKit
import CoreImage
import ScreenCaptureKit
import Vision
_ = NSApplication.shared  // a CLI must connect to the window server before screen capture, or it crashes
// Opt out of App Nap / timer coalescing: in the background (Dota in front) macOS otherwise stretches every short
// sleep, which made 8ms key gaps take ~100ms and slows the frame loop.
let NO_NAP = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical, .idleSystemSleepDisabled],
                                                   reason: "typing bot: real-time screen reading and key timing")

// Calibration knobs
let CAPTURE_WIDTH = 960.0          // px the window is scaled to. 960 + .accurate read every real test frame right (~38ms)
let ROI = CGRect(x: 0.19, y: 0.05, width: 0.61, height: 0.92)  // game panel only (Vision coords, origin bottom-left)
let SETTLE = 0.06                  // s after typing before trusting the screen again (game redraw lag)
let REPEAT_GAP = 0.6               // s before an unchanged leftover is retyped (game took nothing)
// Bottom centre of the game panel, where falling names are lost (frame coords 0-1, origin bottom-left).
// ponytail: fixed box measured from saved frames; adjust if the danger zone sits elsewhere on your screen.
let DANGER = CGRect(x: 0.36, y: 0.28, width: 0.26, height: 0.24)
let LOCK_WAIT = 0.5                // s to wait for a just-typed name that's on screen but unreadable, before typing another
let WHITE_MIN = 200                // channel value a pixel needs to count as "white"
let BIG_RATIO = 1.6                // big word width / its label width (measured 1.5-1.7 on real frames)
let GAME_STICKY = 1.5              // s the game counts as on after the TIME label was last read (OCR misses it at times)
let OVERTIME = 15.0                // s after the clock/Jev says over that leftover words on screen are still typed
let MIN_SCORE = 25                 // ignore words with fewer white+yellow pixels than this (faded / background text)
let JEV_EVERY = 1.0                // s between Jev checks (each ~2.7k tokens = ~$0.0001)
let JEV_PRICE = 0.042 / 1_000_000  // $ per input token (output is free)
let HUD: Set<String> = ["SCORE", "TIME", "HIGH SCORE", "HIGHSCORE"]
let ENV = ProcessInfo.processInfo.environment

let KEYS: [Character: CGKeyCode] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13,
    "e": 14, "r": 15, "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45,
    "m": 46, " ": 49, "-": 27, "'": 39,
]
// Relative letter widths in the game's caps font, to turn the yellow cursor's x position into a letter index.
let WIDTH: [Character: Double] = [  // N = 1; rough advance widths of the game's serif caps
    "A": 0.95, "B": 0.8, "C": 0.9, "D": 0.95, "E": 0.75, "F": 0.7, "G": 0.95, "H": 1.0, "I": 0.45, "J": 0.5, "K": 0.9,
    "L": 0.7, "M": 1.2, "N": 1.0, "O": 1.05, "P": 0.75, "Q": 1.05, "R": 0.85, "S": 0.7, "T": 0.8, "U": 0.95, "V": 0.9,
    "W": 1.3, "X": 0.9, "Y": 0.85, "Z": 0.8, " ": 0.45, "-": 0.5, "'": 0.3,
]
/// Index of the letter whose span contains `at` (0...1 across the word, by estimated widths).
func letterIndex(_ word: String, _ at: Double) -> Int {
    let ws = word.map { WIDTH[$0] ?? 1 }, total = ws.reduce(0, +)
    var acc = 0.0
    for (n, wd) in ws.enumerated() { acc += wd; if acc / total > at { return n } }
    return max(0, ws.count - 1)
}

/// Words the game has shown before (words.txt, grows every run). OCR tokens 1-2 letters off snap to these.
let WORDS_URL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("words.txt")
var knownPhrases = Set(((try? String(contentsOf: WORDS_URL, encoding: .utf8)) ?? "").split(separator: "\n").map { String($0) })
var savedPhrases = knownPhrases  // what's in words.txt (knownPhrases also holds this run's GPT-read names)
var gptNames = Set<String>()     // names GPT added this run; saved to words.txt once typed and completed
var knownTokens = Set(knownPhrases.flatMap { $0.split(separator: " ").map(String.init) })
func snap(_ line: String) -> String {
    line.split(separator: " ", omittingEmptySubsequences: false).map { t -> String in
        let tok = String(t)
        // Short words have real neighbours 1 letter away (WEAVER/REAVER, BAND/BANE): only fix long ones.
        if tok.count < 7 || knownTokens.contains(tok) { return tok }
        // Same length only (keeps letter positions valid), at most 1 wrong letter per 4.
        let best = knownTokens.filter { $0.count == tok.count }
            .map { k in (k, zip(k, tok).filter { $0 != $1 }.count) }.min { $0.1 < $1.1 }
        if let (k, d) = best, d <= tok.count / 4 { return k }
        return tok
    }.joined(separator: " ")
}
/// A label cut off by something drawn over it ("CLAYMO") -> the one known name it starts.
func expand(_ s: String) -> String {
    guard s.count >= 4, !knownPhrases.contains(s) else { return s }
    let m = knownPhrases.filter { $0.hasPrefix(s) }
    return m.count == 1 ? m.first! : s
}
/// A read the list knows once spaces are ignored ("STAFF OFWIZARDRY") -> the listed spelling.
var squashed = Dictionary(knownPhrases.map { ($0.replacingOccurrences(of: " ", with: ""), $0) }, uniquingKeysWith: { a, _ in a })
func known(_ s: String) -> String {
    if knownPhrases.contains(s) { return s }
    if let k = squashed[s.replacingOccurrences(of: " ", with: "")] { return k }
    let e = expand(s)
    if e != s { return e }
    // Off by a letter or two ("STORME SPIRIT", "NULUFIER"): the one known name that close.
    guard s.count >= 5 else { return s }
    let maxD = s.count >= 10 ? 2 : 1
    let near = knownPhrases.filter { abs($0.count - s.count) <= maxD && editDistance($0, s, max: maxD) <= maxD }
    return near.count == 1 ? near.first! : s
}
/// Levenshtein distance, giving up (returns limit+1) once it can't stay within `limit`.
func editDistance(_ x: String, _ y: String, max limit: Int) -> Int {
    let a = Array(x), b = Array(y)
    if a.isEmpty || b.isEmpty { return Swift.max(a.count, b.count) }
    var prev = Array(0...b.count)
    for i in 1...a.count {
        var cur = [Int](repeating: 0, count: b.count + 1)
        cur[0] = i
        for j in 1...b.count {
            let sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)
            cur[j] = Swift.min(Swift.min(prev[j] + 1, cur[j - 1] + 1), sub)
        }
        if cur.min()! > limit { return limit + 1 }
        prev = cur
    }
    return prev[b.count]
}
/// GPT sometimes joins neighbouring names into one ("ORACLE REAVER") or misspells a known one ("OBLIVION STAG").
/// Adding those would teach the bot fake names, so: reject a result made only of known names, or one within two
/// letters of a known name. Returns why it was rejected, nil to accept.
func gptReject(_ w: String) -> String? {
    // Judged against the trusted list (words.txt) only, never against other GPT names from this run: one bad GPT
    // answer ("STRENGTH OF THE TRADE") must not block a correct one ("TRICKS OF THE TRADE").
    let trusted = savedPhrases
    let toks = w.split(separator: " ").map(String.init)
    // Only known names side by side? (dp over word positions)
    var ok = [Bool](repeating: false, count: toks.count + 1); ok[0] = true
    for i in 0..<toks.count where ok[i] {
        for j in (i + 1)...toks.count where trusted.contains(toks[i..<j].joined(separator: " ")) { ok[j] = true }
    }
    if ok[toks.count] { return "just known names side by side" }
    // A piece of a longer known name or phrase ("HOLDING UP", "SNOW ANGE", "TRADE").
    if let k = knownPhrases.first(where: { $0.count > w.count && $0.contains(w) }) { return "part of \(k)" }
    for k in trusted {
        // A known name plus a one-word scrap ("BUTTERFLY SILIUS"): two names overlapping.
        if w.count > k.count, (w.hasPrefix(k + " ") || w.hasSuffix(" " + k)), toks.count - k.split(separator: " ").count == 1 {
            return "known name \(k) plus a scrap"
        }
        // Contains a whole known name plus other words: two overlapping names merged ("STORMCRAFTER DUELLING BLADE",
        // "ARCANE SPLINT MAIL"). Short names (AXE in "AXE LIKES THIS VERY MUCH") don't count.
        let ks = k.replacingOccurrences(of: " ", with: ""), wsq = w.replacingOccurrences(of: " ", with: "")
        if w != k, (ks.count >= 8 && wsq.contains(ks)) || (k.count >= 6 && (" " + w + " ").contains(" " + k + " ")) {
            return "contains known name \(k)"
        }
        // A known phrase with one word swapped ("BOOTS OF VIPER", "YOU CAN RUN MNKNIGHT SHOULD RUN").
        let kt = k.split(separator: " ").map(String.init)
        // Sentences only: short names share words for real (RING OF HEALTH / RING OF REGEN, VOODOO MASK / VOODOO RESTORATION).
        if kt.count == toks.count, kt.count >= 4, zip(kt, toks).filter({ $0 != $1 }).count == 1 { return "one word off \(k)" }
        // A distorted known name ("PRYLACTERYSTER", "SHADOW KNIGHT"), unless one simply extends the other.
        if w.contains(k) || k.contains(w) { continue }
        let limit = Swift.max(2, w.count / 4)  // 40% rejected real new names; this is the setting from the 14.9k round
        if abs(k.count - w.count) <= limit && editDistance(k, w, max: limit) <= limit { return "looks like a misspelling of \(k)" }
    }
    return nil
}

/// Adds a name to the in-memory list (this run only). learn() also saves it to words.txt.
func addVocab(_ phrase: String) {
    knownPhrases.insert(phrase)
    phrase.split(separator: " ").forEach { knownTokens.insert(String($0)) }
    squashed[phrase.replacingOccurrences(of: " ", with: "")] = phrase
}
func learn(_ phrase: String) {
    guard !knownPhrases.contains(phrase) || !savedPhrases.contains(phrase) else { return }
    addVocab(phrase); savedPhrases.insert(phrase)
    try? (savedPhrases.sorted().joined(separator: "\n") + "\n").write(to: WORDS_URL, atomically: true, encoding: .utf8)
}

/// One OCR'd line. rect = normalized frame coords (origin bottom-left). start = index of the first lit letter
/// (typed letters turn grey; the cursor letter is yellow; untyped are white). score = lit pixels.
/// cursor = a yellow cursor letter was found inside this line (start then points at it).
/// typedAbove = grey "typed" ink in the line-height strip right above the line (the earlier part of a phrase).
struct Cand { let text: String; let start: Int; let score: Int; let yellow: Int; let rect: CGRect; var cursor = false; var typedAbove = 0, whiteAbove = 0 }
/// What to type: the full name and the part still left. x = screen position, for typing order.
/// verified = the big text agreed with the label (safe to learn).
/// x, y = centre of the name's label on screen (0-1, origin bottom-left).
struct Target { let text: String; let left: String; var x = 0.0; var y = 0.0; var verified = false
    var started: Bool { left.trimmingCharacters(in: .whitespaces).count < text.count }  // some letters already typed
}

/// OCR the game panel. gameOn = the TIME label is visible.
/// mult = score multiplier shown in the HUD ("X1.5"), 0 if not read this frame.
/// seen = names whose label is on screen with lit letters still under it, even if the cursor couldn't be read
/// (tells "finished" from "unreadable this frame").
/// unreadable = lit text the bot couldn't turn into a name it trusts (sent to GPT for the spelling).
func readWords(_ pb: CVPixelBuffer) -> (gameOn: Bool, cands: [Cand], targets: [Target], seen: Set<String>, mult: Double, unreadable: [String]) {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate  // .fast cut words short (HYPERSTONE -> HYPERST) on real frames
    req.usesLanguageCorrection = false
    req.regionOfInterest = ROI
    try? VNImageRequestHandler(cvPixelBuffer: pb).perform([req])

    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let px = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(pb), w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)

    /// The yellow cursor letter in a pixel rect: the leftmost run of yellow columns whose yellow spans >= minH rows
    /// (the yellow points like "10" under each word are less than half a letter tall, so they're skipped), merged
    /// with the runs that follow within one glyph width. Returns a point 30% into the glyph from its left edge (the
    /// centre over-shoots on narrow letters like J, F, I).
    func cursorRun(_ x0: Int, _ x1: Int, _ y0: Int, _ y1: Int, minH: Int, glyphW: Int) -> Double? {
        let x0 = max(0, x0), x1 = min(w, x1), y0 = max(0, y0), y1 = min(h, y1)
        guard x1 > x0, y1 > y0 else { return nil }
        var runs: [(l: Int, r: Int, ext: Int)] = [], cur: (l: Int, top: Int, bot: Int)? = nil, gap = 0
        for x in x0...x1 {
            var top = Int.max, bot = -1, n = 0
            if x < x1 {
                for y in y0..<y1 {
                    let i = y * bpr + x * 4
                    let bl = Int(px[i]), g = Int(px[i + 1]), r = Int(px[i + 2])
                    if min(r, g) >= 140 && bl < min(r, g) * 6 / 10 { n += 1; top = min(top, y); bot = max(bot, y) }
                }
            }
            if n >= 2 { cur = cur.map { ($0.l, min($0.top, top), max($0.bot, bot)) } ?? (x, top, bot); gap = 0 }
            else if let c = cur { gap += 1; if gap > 2 || x == x1 { runs.append((c.l, x - gap, c.bot - c.top + 1)); cur = nil; gap = 0 } }
        }
        guard let first = runs.first(where: { $0.ext >= minH }) else { return nil }
        let right = runs.filter { $0.l >= first.l && $0.l <= first.l + glyphW }.map { $0.r }.max() ?? first.r
        return Double(first.l) + 0.3 * Double(right - first.l)
    }

    var gameOn = false, cands: [Cand] = [], mult = 0.0
    for o in req.results ?? [] {
        guard let raw = o.topCandidates(1).first?.string else { continue }
        let up = raw.uppercased().trimmingCharacters(in: .whitespaces)
        if up == "TIME" { gameOn = true }
        if up.count <= 5, up.hasPrefix("X"), let m = Double(up.dropFirst()) { mult = m >= 10 ? m / 10 : m }  // "X20" = x2.0
        if HUD.contains(up) { continue }
        // Drop number tokens ("50 TANGO" = points next to a name); the clock/score lines end up empty.
        // Punctuation becomes a space ("NEGATIVE, BE" must not turn into NEGATIVEBE); apostrophes just go.
        let isIO = ["IO", "10", "I0", "1O", "LO"].contains(up)  // the hero IO reads like the points "10"
        let spaced = isIO ? "IO" : String(up.map { ",.!?:;".contains($0) ? " " : $0 }).split(separator: " ")
            .filter { !$0.contains(where: { $0.isNumber }) }.joined(separator: " ")
        let text = known(snap(spaced.filter { $0.isLetter || " -'".contains($0) }))
        guard text.filter({ $0.isLetter }).count >= 2 else { continue }

        // Box is relative to ROI; convert to pixel coords (buffer rows start at the top).
        let b = o.boundingBox
        let x0 = max(0, Int((ROI.minX + b.minX * ROI.width) * Double(w))), x1 = min(w, Int((ROI.minX + b.maxX * ROI.width) * Double(w)))
        let y0 = max(0, Int((1 - ROI.minY - b.maxY * ROI.height) * Double(h))), y1 = min(h, Int((1 - ROI.minY - b.minY * ROI.height) * Double(h)))
        guard x1 > x0, y1 > y0 else { continue }
        var white = 0, yellow = 0, whiteTop = Int.max, whiteBot = -1
        var litCols = [Int](repeating: 0, count: x1 - x0)
        for y in y0..<y1 {
            for x in x0..<x1 {
                let i = y * bpr + x * 4  // BGRA
                let bl = Int(px[i]), g = Int(px[i + 1]), r = Int(px[i + 2])
                let lo = min(r, g, bl), hi = max(r, g, bl)
                if lo >= WHITE_MIN && hi - lo < 40 { white += 1; litCols[x - x0] += 1; whiteTop = min(whiteTop, y); whiteBot = max(whiteBot, y) }  // untyped
                else if min(r, g) >= 140 && bl < min(r, g) * 6 / 10 { yellow += 1; litCols[x - x0] += 1 }  // cursor
            }
        }
        // First letter still to type: the yellow cursor letter's left edge, else the leftmost lit column
        // (cursor sitting on a space). Mapped to a letter index via estimated letter widths (midpoint rule).
        var start = 0
        // Letter height: rows the white letters span; else most of the box (the box may include the points below).
        let letterH = white >= 30 ? whiteBot - whiteTop + 1 : (y1 - y0) * 6 / 10
        let glyphW = (x1 - x0) / max(1, text.count)
        let run = yellow > 0 ? cursorRun(x0, x1, y0, y1, minH: letterH / 2, glyphW: glyphW) : nil
        if let c = run {
            start = letterIndex(text, (c - Double(x0)) / Double(x1 - x0))
            // The box can start a letter early. Typed letters leave grey ink before the cursor; no ink = the cursor
            // is on the first letter.
            if start > 0 {
                var ink = 0
                for y in y0..<y1 {
                    for x in x0..<max(x0, Int(c) - glyphW / 3) {
                        let i = y * bpr + x * 4
                        let bl = Int(px[i]), g = Int(px[i + 1]), r = Int(px[i + 2])
                        let lo = min(r, g, bl), hi = max(r, g, bl)
                        if lo >= 40 && lo < WHITE_MIN && hi - lo < 45 { ink += 1 }  // typed letters: dim grey 40-80, background < 40
                    }
                }
                if ink < letterH { start = 0 }
            }
        } else if let col = litCols.firstIndex(where: { $0 >= 2 }) {
            start = letterIndex(text, (Double(col) + Double(glyphW) / 2 - Double(x0)) / Double(x1 - x0))
        }
        let rect = CGRect(x: ROI.minX + b.minX * ROI.width, y: ROI.minY + b.minY * ROI.height,
                          width: b.width * ROI.width, height: b.height * ROI.height)
        // The points "10" are small; the big IO is ~3% of the frame tall with a cursor or some white (I is yellow, O thin).
        if isIO && !(Double(y1 - y0) / Double(h) >= 0.027 && (run != nil || white >= 5)) { continue }
        cands.append(Cand(text: text, start: start, score: white + yellow, yellow: yellow, rect: rect, cursor: run != nil))
    }

    // Every name on screen has a small clean label (never overlapped) above its big text. Labels = unlit lines.
    // For each, find the big text under it and where its yellow cursor is: that gives the letters still to type.
    // Labels whose big text has no cursor (faded, finished) are skipped.
    /// Dim grey "ink" left by typed letters in a normalized rect (background is darker, untyped text is white).
    func inkIn(_ r: CGRect) -> Int {
        let x0 = max(0, Int(r.minX * Double(w))), x1 = min(w, Int(r.maxX * Double(w)))
        let y0 = max(0, Int((1 - r.maxY) * Double(h))), y1 = min(h, Int((1 - r.minY) * Double(h)))
        var n = 0
        for y in y0..<max(y0, y1) {
            for x in x0..<max(x0, x1) {
                let i = y * bpr + x * 4
                let bl = Int(px[i]), g = Int(px[i + 1]), r = Int(px[i + 2])
                let lo = min(r, g, bl), hi = max(r, g, bl)
                if lo >= 40 && lo < WHITE_MIN && hi - lo < 45 { n += 1 }
            }
        }
        return n
    }
    /// White (untyped) pixels in a normalized rect.
    func whiteIn(_ r: CGRect) -> Int {
        let x0 = max(0, Int(r.minX * Double(w))), x1 = min(w, Int(r.maxX * Double(w)))
        let y0 = max(0, Int((1 - r.maxY) * Double(h))), y1 = min(h, Int((1 - r.minY) * Double(h)))
        var n = 0
        for y in y0..<max(y0, y1) {
            for x in x0..<max(x0, x1) {
                let i = y * bpr + x * 4
                let lo = min(Int(px[i]), Int(px[i + 1]), Int(px[i + 2])), hi = max(Int(px[i]), Int(px[i + 1]), Int(px[i + 2]))
                if lo >= WHITE_MIN && hi - lo < 40 { n += 1 }
            }
        }
        return n
    }
    /// cursorRun on a normalized rect (origin bottom-left), normalized x.
    func cursorX(_ r: CGRect, letterH: Double, glyphW: Double) -> Double? {
        cursorRun(Int(r.minX * Double(w)), Int(r.maxX * Double(w)), Int((1 - r.maxY) * Double(h)), Int((1 - r.minY) * Double(h)),
                  minH: Int(letterH * Double(h) / 2), glyphW: Int(glyphW * Double(w))).map { $0 / Double(w) }
    }
    var targets: [Target] = [], names = Set<String>(), seen = Set<String>(), unreadable: [String] = []
    for k in cands.indices where cands[k].cursor {
        let r = cands[k].rect
        let above = CGRect(x: r.minX, y: r.maxY, width: r.width, height: r.height)
        cands[k].typedAbove = inkIn(above); cands[k].whiteAbove = whiteIn(above)
    }
    // Labels are green-yellow: (almost) no white; big text always has white or a yellow cursor.
    // Name: de-doubled ("HEADDRESS HEADDRESS"), expanded if cut off ("CLAYMO"), and split when two overlapping
    // labels read as one line ("PHASE BOOTSBOTTLE" -> PHASE BOOTS + BOTTLE, each with its share of the box).
    var labels: [(name: String, rect: CGRect, split: Bool)] = []
    for L in cands where L.score - L.yellow < 15 {
        let toks = L.text.filter { $0 != "'" }.split(separator: " ").map(String.init)
        let half = toks.count / 2
        let raw = toks.count % 2 == 0 && half > 0 && Array(toks[..<half]) == Array(toks[half...]) ? toks[..<half].joined(separator: " ") : toks.joined(separator: " ")
        if !knownPhrases.contains(raw), raw.count >= 8 {
            var parts: [(Int, String)] = [], used = [Bool](repeating: false, count: raw.count)
            for k in knownPhrases.filter({ $0.count >= 4 }).sorted(by: { $0.count > $1.count }) {
                guard let r = raw.range(of: k) else { continue }
                let o = raw.distance(from: raw.startIndex, to: r.lowerBound)
                if used[o..<(o + k.count)].contains(true) { continue }
                for i in o..<(o + k.count) { used[i] = true }
                parts.append((o, k))
            }
            // Leftover pieces may be cut-off names ("ROSHA" -> ROSHAN); anything else is garbage and dropped.
            if !parts.isEmpty {
                var o = 0
                for piece in raw.split(separator: " ", omittingEmptySubsequences: false).map(String.init) {
                    if piece.count >= 4, !used[o..<(o + piece.count)].contains(true), expand(piece) != piece {
                        for i in o..<(o + piece.count) { used[i] = true }
                        parts.append((o, expand(piece)))
                    }
                    o += piece.count + 1
                }
            }
            if !parts.isEmpty && used.filter({ $0 }).count < raw.count - 1 || parts.count >= 2 {
                for (o, k) in parts {
                    labels.append((k, CGRect(x: L.rect.minX + L.rect.width * Double(o) / Double(raw.count), y: L.rect.minY,
                                              width: L.rect.width * Double(min(k.count, raw.count - o)) / Double(raw.count), height: L.rect.height), true))
                }
                continue
            }
        }
        labels.append((expand(raw), L.rect, false))
    }
    for (label, lr, split) in labels {
        let lh = lr.height, letterH = lh * 1.2  // big letters are ~1.5x the label box, letters ~80% of a box
        // Big lines under this label: with white or a cursor, overlapping it, starting within ~1.5 label heights below.
        let bigs = cands.filter { b in
            // Bright text only: a faded name (locked out while another word is half typed) has no white letters.
            // A cursor alone counts only for the last letter or two.
            (b.score - b.yellow >= 15 || (b.cursor && b.text.count - b.start <= 2)) && b.rect.maxY <= lr.minY + 0.6 * lh && b.rect.maxY > lr.minY - 5 * lh
                && min(b.rect.maxX, lr.maxX + lh) > max(b.rect.minX, lr.minX - lh)
        }.sorted { $0.rect.maxY > $1.rect.maxY }
        // A label the reader garbled ("JAELIN", "CHAL") is replaced by the big text under it when that's known.
        var name = label
        // Only for a label of about the same length: a phrase line ("TRADE" under "TRICKS OF THE TRADE") isn't the name.
        if !split, !knownPhrases.contains(name),
           let b = bigs.first(where: { knownPhrases.contains($0.text) && Double($0.text.count) >= 0.7 * Double(name.count) }) { name = b.text }
        guard name.filter({ $0.isLetter }).count >= 2, !names.contains(name) else { continue }
        if !bigs.isEmpty { seen.insert(name) }  // still has lit (untyped) letters; a finished name is all grey
        // Clean read of (part of) the big text: its position in the name + its cursor = what's left.
        var idx: Int? = nil
        for (k, b) in bigs.enumerated() where b.score > 0 {
            let r = k == 0 ? name.range(of: b.text) : name.range(of: b.text, options: .backwards)
            guard let r = r else { continue }
            let off = name.distance(from: name.startIndex, to: r.lowerBound)
            var at = off + b.start, hasCursor = b.cursor
            // The reader often drops letters at the front: faded typed ones, the yellow cursor letter, or ones
            // covered by something ("ULLIFIER" for a fresh NULLIFIER, "OTS" for BOOTS). Look left of the read text
            // for the cursor and count back letters from how far left it is.
            if !b.cursor, off > 0 {
                let pitch = b.rect.width / Double(max(1, b.text.count))
                let look = CGRect(x: b.rect.minX - (Double(off) + 0.6) * pitch, y: b.rect.minY, width: (Double(off) + 0.6) * pitch, height: b.rect.height)
                if let cx = cursorX(look, letterH: letterH, glyphW: pitch) {
                    at = max(0, off - max(1, Int(((b.rect.minX - cx) / pitch + 0.5).rounded()))); hasCursor = true
                } else if b.start == 0 && inkIn(look) < Int(letterH * Double(h)) {
                    // No cursor found and no grey ink where the "typed" letters would be: the reader just missed
                    // the first letters of a fresh word (JAKIRO read as AKIRO). Type it all.
                    at = 0
                }
            }
            if hasCursor || idx == nil { idx = at }
            if hasCursor { break }
        }
        // Learnable only if the big text spells the whole name (a partial match may be a cut-off label).
        let verified = idx != nil && (bigs.contains { $0.text == name } || bigs.map { $0.text }.joined(separator: " ") == name)
        // Overlapping words merge into garbage: estimate the big word's extent from the label and find its cursor.
        // Only for names already in words.txt; an unknown, unconfirmed label may itself be misread.
        // Only from a real label (label-sized): a faded big name has no white either and would pass as a "label"
        // (AEON DISK, faded while another word was locked, kept being typed into the lock).
        if idx == nil, knownPhrases.contains(name), lh < 0.035 {
            let bw = lr.width * BIG_RATIO, bx0 = lr.midX - bw / 2
            let band = CGRect(x: bx0 - 0.05 * bw, y: lr.minY - 2 * lh, width: bw * 1.05, height: 2 * lh)
            if let cx = cursorX(band, letterH: letterH, glyphW: bw / Double(name.count)) { idx = letterIndex(name, (cx - bx0) / bw) }
        }
        guard let i = idx, i < name.count else { continue }
        // An unknown name the big text doesn't confirm is probably a misread ("SO WIZARD HAT"): typing it means wrong
        // keys. GPT (below) adds the real spelling to the list, and the next frame types it.
        if !knownPhrases.contains(name) && !verified { unreadable.append(name); continue }
        names.insert(name)
        var left = String(name.dropFirst(i))
        if i > 0, name[name.index(name.startIndex, offsetBy: i - 1)] == " " { left = " " + left }  // cursor on a space
        targets.append(Target(text: name, left: left, x: lr.midX, y: lr.midY, verified: verified))
    }
    // Label hidden under another word: a known name whose big text (white, with a cursor) is read cleanly.
    for b in cands where b.cursor && (b.score - b.yellow >= 15 || b.text == "IO") && knownPhrases.contains(b.text) && !names.contains(b.text) {
        if names.contains(where: { $0.contains(b.text) }) { continue }  // part of a longer name already handled
        names.insert(b.text); seen.insert(b.text)
        targets.append(Target(text: b.text, left: String(b.text.dropFirst(b.start)), x: b.rect.midX, y: b.rect.midY))
    }
    targets.sort { ($0.started ? 0 : 1, $0.x) < ($1.started ? 0 : 1, $1.x) }  // final order is set in the live loop
    // Lit big text with a cursor that isn't part of any name we're typing: also unreadable.
    for b in cands where b.cursor && b.score - b.yellow >= 15 && !targets.contains(where: { $0.text.contains(b.text) }) {
        unreadable.append(b.text)
    }
    return (gameOn, cands, targets, seen, mult, unreadable)
}


/// Target = the biggest lit line (the hero label above is smaller; faded words aren't lit), plus the lines stacked
/// directly above/below it at the same size, so a phrase wrapped over 2+ lines is typed as one, with spaces.
/// Fallback when no label could be used: only for names on the list. Anything else is likely two overlapping
/// words read as one ("GUARDIAN GRBAVES") and typing it would leave a word half done.
func fallback(_ cands: [Cand]) -> Target? {
    guard let p = pick(true, cands), knownPhrases.contains(p.text) else { return nil }
    return p
}

/// Stuck word: nothing trusted to type, but bright text with a cursor is on screen. That's the word the game is
/// locked on (everything else fades), even if its name can't be read. Finish it from the visible letters: the
/// cursor line from the cursor on, plus the bright lines stacked under it.
func rescue(_ cands: [Cand]) -> Target? {
    // Only with evidence the word is half typed: the cursor mid-line, or grey typed text right above it
    // ("SANGE AND YASHA" above "AND KAYA"). A fresh word has neither; two overlapped fresh words read as garbage.
    guard let c = cands.filter({ $0.cursor && $0.score - $0.yellow >= 15
            && ($0.start > 0 || ($0.typedAbove >= Int($0.rect.height * 540) && $0.whiteAbove < 15)) }).max(by: { $0.score < $1.score }) else { return nil }
    var lines = [c]
    while let next = cands.first(where: { b in
        let a = lines[lines.count - 1]
        return !lines.contains(where: { $0.rect == b.rect }) && b.score - b.yellow >= 15
            && abs(b.rect.height - a.rect.height) < 0.25 * a.rect.height
            && b.rect.maxY <= a.rect.minY + 0.2 * a.rect.height && a.rect.minY - b.rect.maxY < a.rect.height
            && min(b.rect.maxX, a.rect.maxX) > max(b.rect.minX, a.rect.minX)
    }) { lines.append(next) }
    var left = ([String(c.text.dropFirst(c.start))] + lines.dropFirst().map { $0.text }).joined(separator: " ")
    guard left.contains(where: { $0.isLetter }) else { return nil }
    // A garbled read of a known name with the cursor on its first letter ("NUKDTALISMAN"): type the known name.
    if c.start == 0, lines.count == 1, !knownPhrases.contains(c.text) {
        let sq = c.text.replacingOccurrences(of: " ", with: "")
        if let k = knownPhrases.first(where: { k in
            let ks = k.replacingOccurrences(of: " ", with: "")
            return abs(ks.count - sq.count) <= 2 && editDistance(ks, sq, max: 3) <= 3
        }) { left = k }
    }
    return Target(text: "~" + lines.map { $0.text }.joined(separator: " "), left: left, x: c.rect.midX, y: c.rect.midY)  // "~": big text only
}

func pick(_ gameOn: Bool, _ cands: [Cand]) -> Target? {
    guard gameOn, let anchor = cands.filter({ $0.score >= MIN_SCORE }).max(by: { $0.rect.height < $1.rect.height }) else { return nil }
    func stacked(_ a: Cand, _ b: Cand) -> Bool {  // b is the next line under a
        abs(a.rect.height - b.rect.height) < 0.2 * max(a.rect.height, b.rect.height)  // hero label is ~30% smaller
            && min(a.rect.maxX, b.rect.maxX) > max(a.rect.minX, b.rect.minX)
            && b.rect.maxY <= a.rect.minY + 0.2 * a.rect.height && a.rect.minY - b.rect.maxY < a.rect.height
    }
    var lines = [anchor]
    while let up = cands.first(where: { stacked($0, lines[0]) }) { lines.insert(up, at: 0) }
    while let down = cands.first(where: { stacked(lines[lines.count - 1], $0) }) { lines.append(down) }

    let text = lines.map { $0.text }.joined(separator: " ")
    guard let first = lines.firstIndex(where: { $0.score >= MIN_SCORE }) else { return nil }
    var left = ([String(lines[first].text.dropFirst(lines[first].start))] + lines[(first + 1)...].map { $0.text }).joined(separator: " ")
    // Cursor on a space shows no yellow anywhere: the space itself is still to type.
    let typedSome = first > 0 || lines[first].start > 0
    if typedSome && lines.allSatisfy({ $0.yellow == 0 }) { left = " " + left }
    return left.trimmingCharacters(in: .whitespaces).isEmpty ? nil : Target(text: text, left: left)
}

// MARK: typing

/// System Events: slow (~230ms/letter) but the path proven to register in Dota.
func typeSE(_ w: String) {
    var err: NSDictionary?
    NSAppleScript(source: "tell application \"System Events\" to keystroke \"\(w)\"")?.executeAndReturnError(&err)
    if let err = err { log("typing blocked: \(err[NSAppleScript.errorMessage] ?? err) (give Terminal Accessibility + Automation permission)") }
}

/// Busy-wait: exact even when macOS coalesces timers for a background process (usleep(8ms) took ~100ms there).
func spin(ms: Double) {
    let end = DispatchTime.now().uptimeNanoseconds + UInt64(ms * 1_000_000)
    while DispatchTime.now().uptimeNanoseconds < end {}
}

/// Raw key events with the character attached, `gap` ms after each down and each up.
func typeCG(_ w: String, gap: Double) {
    for ch in w {
        guard let k = KEYS[ch] else { continue }
        for down in [true, false] {
            let e = CGEvent(keyboardEventSource: nil, virtualKey: k, keyDown: down)
            var u = Array(String(ch).utf16)
            e?.keyboardSetUnicodeString(stringLength: u.count, unicodeString: &u)
            e?.post(tap: .cghidEventTap)
            if gap > 0 { spin(ms: gap) }
        }
    }
}

/// mode: "se" or "cg:<ms>"
func type(_ word: String, mode: String) {
    let w = word.lowercased().filter { KEYS[$0] != nil }
    if mode.hasPrefix("cg") { typeCG(w, gap: Double(mode.split(separator: ":").last ?? "") ?? 10) } else { typeSE(w) }
}

/// Front-most normal window's owner, straight from the window server.
func frontIsDota() -> Bool {
    let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
    let front = list.first { ($0[kCGWindowLayer as String] as? Int) == 0 }
    return (front?[kCGWindowOwnerName as String] as? String)?.lowercased().contains("dota") ?? false
}

/// Continuous 60fps stream of the Dota window; always holds the newest frame, so there's no per-shot setup cost.
final class Grabber: NSObject, SCStreamOutput, SCStreamDelegate {
    private let lock = NSLock()
    private var frame: (pb: CVPixelBuffer, at: Date, seq: Int)?
    private var seq = 0
    var stream: SCStream?

    func start() async -> Bool {
        guard let content = try? await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true),
              let win = content.windows
                  .filter({ $0.windowLayer == 0 && ($0.owningApplication?.applicationName.lowercased().contains("dota") ?? false) })
                  .max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height })
        else { return false }
        let cfg = SCStreamConfiguration()
        cfg.width = Int(CAPTURE_WIDTH)
        cfg.height = Int(win.frame.height * CAPTURE_WIDTH / win.frame.width)
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)
        cfg.queueDepth = 5
        cfg.showsCursor = false
        let s = SCStream(filter: SCContentFilter(desktopIndependentWindow: win), configuration: cfg, delegate: self)
        do {
            try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: DispatchQueue(label: "frames"))
            try await s.startCapture()
        } catch { log("capture failed: \(error)"); return false }
        stream = s
        return true
    }

    func stop() { stream?.stopCapture(); stream = nil }

    func stream(_ s: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        // Only complete frames carry pixels; idle frames (nothing changed) are skipped.
        guard type == .screen,
              let info = (CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])?.first,
              let raw = info[.status] as? Int, SCFrameStatus(rawValue: raw) == .complete,
              let pb = sb.imageBuffer else { return }
        lock.withLock { seq += 1; frame = (pb, Date(), seq) }
    }

    func stream(_ s: SCStream, didStopWithError error: Error) { stream = nil }  // window closed/recreated: restart

    func newest(after s: Int) -> (pb: CVPixelBuffer, at: Date, seq: Int)? {
        lock.withLock { frame.flatMap { $0.seq > s ? $0 : nil } }
    }
}

/// "Is the game on?" decision from Jev (TypeSafe AI). Called from the main loop without blocking it.
final class Jev {
    private let lock = NSLock()
    private var _active = false, _busy = false
    private(set) var calls = 0, tokens = 0
    var active: Bool { lock.withLock { _active } }
    var cost: Double { lock.withLock { Double(tokens) * JEV_PRICE } }
    let key: String
    let ci = CIContext()

    init?() {
        let envFile = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent(".env")
        let fileVars = ((try? String(contentsOf: envFile, encoding: .utf8)) ?? "").split(separator: "\n")
            .reduce(into: [String: String]()) { d, l in let kv = l.split(separator: "=", maxSplits: 1); if kv.count == 2 { d[String(kv[0])] = String(kv[1]) } }
        guard let k = ENV["JEV_API_KEY"] ?? fileVars["JEV_API_KEY"] else { return nil }
        key = k.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Starts a check unless one is already running. Any error leaves the game "off", so a flaky call never types.
    func checkInBackground(_ pb: CVPixelBuffer) {
        guard lock.withLock({ () -> Bool in if _busy { return false }; _busy = true; return true }) else { return }
        guard let jpg = ci.jpegRepresentation(of: CIImage(cvPixelBuffer: pb), colorSpace: CGColorSpaceCreateDeviceRGB()) else {
            lock.withLock { _busy = false }; return
        }
        let body: [String: Any] = [
            "model": "jev-1.14.0", "state": ["screen": "shot"],
            "assets": ["shot": ["type": "image", "data_base64": jpg.base64EncodedString(), "format": "jpg"]],
            "questions": ["active": ["type": "noul", "instructions":
                "Is the Dota 2 typing minigame open with words to type on screen (count it as yes even if the timer shows 0:00)?"]],
        ]
        var req = URLRequest(url: URL(string: "https://api.typesafe.ai/v1/systemone")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("true", forHTTPHeaderField: "TypeSafe-Danger-Zone-Images")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        req.timeoutInterval = 5
        let t = Date()
        URLSession.shared.dataTask(with: req) { data, _, error in
            var on = false, used = 0, p = 0.0
            if let error = error { log("jev error: \(error.localizedDescription)") }
            if let data = data, let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                p = ((j["answers"] as? [String: Any])?["active"] as? [String: Any])?["noul"] as? Double ?? 0
                used = (j["usage"] as? [String: Any])?["input_tokens"] as? Int ?? 0
                on = p >= 0.5
                if j["answers"] == nil { log("jev error: \(String(data: data, encoding: .utf8) ?? "")") }
            }
            let (changed, n) = self.lock.withLock { () -> (Bool, Int) in
                let ch = on != self._active; self._active = on; self.calls += 1; self.tokens += used; self._busy = false
                return (ch, self.calls)
            }
            if changed || n == 1 {
                log("jev: game \(on ? "ON" : "OFF") (\(String(format: "%.2f", p)), \(ms(t))ms)  calls \(n)  jev cost $\(String(format: "%.4f", self.cost))")
            }
        }.resume()
    }
}

/// GPT vision supplies spellings the local reader can't get (overlapping names, long wrapped phrases). It's slow
/// (1-2s) and bad at cursor positions, so it only adds names to the list; the local reader still decides what to type.
final class GPTSpeller {
    private let lock = NSLock()
    private var busy = false, found: [String] = []
    private(set) var calls = 0, inTok = 0, outTok = 0
    var cost: Double { lock.withLock { Double(inTok) * 0.40 / 1e6 + Double(outTok) * 1.60 / 1e6 } }  // gpt-4.1-mini $/token
    let key: String
    let ci = CIContext()

    init?() {
        let envFile = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent(".env")
        let vars = ((try? String(contentsOf: envFile, encoding: .utf8)) ?? "").split(separator: "\n")
            .reduce(into: [String: String]()) { d, l in let kv = l.split(separator: "=", maxSplits: 1); if kv.count == 2 { d[String(kv[0])] = String(kv[1]) } }
        guard let k = ENV["OPENAI_API_KEY"] ?? vars["OPENAI_API_KEY"] else { return nil }
        key = k.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Names read since the last call (drained by the main loop, which owns the name list).
    func take() -> [String] { lock.withLock { let f = found; found = []; return f } }

    func ask(_ pb: CVPixelBuffer) {
        guard lock.withLock({ () -> Bool in if busy { return false }; busy = true; return true }),
              let jpg = ci.jpegRepresentation(of: CIImage(cvPixelBuffer: pb), colorSpace: CGColorSpaceCreateDeviceRGB()) else { return }
        let prompt = "Dota 2 typing minigame screenshot. List the full text of every BIG white word or phrase floating in the "
            + "scene (ignore the small green labels, numbers and the HUD). Phrases may wrap over several lines: join them. "
            + "Uppercase, no punctuation. JSON: {\"words\":[\"TEXT\", ...]}"
        let body: [String: Any] = ["model": "gpt-4.1-mini", "response_format": ["type": "json_object"],
            "messages": [["role": "user", "content": [["type": "text", "text": prompt],
                ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + jpg.base64EncodedString()]]]]]]
        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        req.timeoutInterval = 6
        let t = Date()
        URLSession.shared.dataTask(with: req) { data, _, error in
            var words: [String] = [], i = 0, o = 0
            if let data = data, let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let u = j["usage"] as? [String: Any]; i = u?["prompt_tokens"] as? Int ?? 0; o = u?["completion_tokens"] as? Int ?? 0
                if let c = ((j["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String,
                   let w = (try? JSONSerialization.jsonObject(with: Data(c.utf8))) as? [String: Any] {
                    words = (w["words"] as? [String] ?? []).map { raw in
                        String(raw.uppercased().map { ",.!?:;".contains($0) ? " " : $0 }).split(separator: " ").joined(separator: " ")
                            .filter { $0.isLetter || " -".contains($0) }
                    }.filter { $0.filter { $0.isLetter }.count >= 3 }
                } else if j["error"] != nil { log("gpt error: \(j["error"]!)") }
            } else if let e = error { log("gpt error: \(e.localizedDescription)") }
            self.lock.withLock { self.found += words; self.calls += 1; self.inTok += i; self.outTok += o; self.busy = false }
            log("  gpt (\(ms(t))ms): \(words.joined(separator: " | "))")
        }.resume()
    }
}

func ms(_ since: Date) -> Int { Int(Date().timeIntervalSince(since) * 1000) }

/// Everything also goes to dotatyper.log (next to the binary) so a run can be diagnosed afterwards.
let LOG_URL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("dotatyper.log")
let logLock = NSLock()
var logHandle: FileHandle? = nil
func log(_ s: String) {
    print(s)
    let f = DateFormatter(); f.dateFormat = "HH:mm:ss.SSS"
    logLock.withLock { logHandle?.write(Data("\(f.string(from: Date())) \(s)\n".utf8)) }
}
/// The app that actually has keyboard focus (window order is unreliable with fullscreen apps on other monitors).
let FRONT_SCRIPT = NSAppleScript(source: "tell application \"System Events\" to get name of first application process whose frontmost is true")
func frontOwner() -> String {
    var err: NSDictionary?
    if let name = FRONT_SCRIPT?.executeAndReturnError(&err).stringValue { return name }
    return "? (\(err?[NSAppleScript.errorMessage] ?? "System Events blocked: allow Automation for your terminal"))"
}

// MARK: test mode: ./dotatyper image.png  (draws the image into a BGRA buffer, same as a captured frame)
if CommandLine.arguments.count > 1 {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL, nil),
          let img0 = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print("can't read image"); exit(1) }
    let W = Int(CAPTURE_WIDTH), H = img0.height * W / img0.width  // same size the live capture uses
    var pbo: CVPixelBuffer?
    CVPixelBufferCreate(nil, W, H, kCVPixelFormatType_32BGRA, nil, &pbo)
    let pb = pbo!
    CVPixelBufferLockBaseAddress(pb, [])
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: W, height: H, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.interpolationQuality = .high
    ctx.draw(img0, in: CGRect(x: 0, y: 0, width: W, height: H))
    CVPixelBufferUnlockBaseAddress(pb, [])
    _ = readWords(pb)  // warm-up
    let t = Date()
    let r = readWords(pb)
    let ts = r.targets.isEmpty ? [rescue(r.cands) ?? fallback(r.cands)].compactMap { $0 } : r.targets  // as in live
    print("\(ms(t))ms  gameOn=\(r.gameOn)  would type: \(ts.isEmpty ? "(nothing)" : ts.map { $0.left == $0.text ? $0.left : "\($0.left) (of \($0.text))" }.joined(separator: ", "))")
    if ENV["JEV"] == "1", let jev = Jev() {
        jev.checkInBackground(pb)
        while jev.calls == 0 { usleep(20_000) }
    }
    exit(0)
}

// MARK: live
FileManager.default.createFile(atPath: LOG_URL.path, contents: nil)
logHandle = try? FileHandle(forWritingTo: LOG_URL)
try? FileManager.default.createDirectory(atPath: "debug", withIntermediateDirectories: true)
let jev: Jev? = ENV["JEV"] == "1" ? Jev() : nil
if ENV["JEV"] == "1" && jev == nil { print("JEV=1 but no JEV_API_KEY in .env"); exit(1) }
let DEBUG = ENV["DEBUG"] == "1", DRY = ENV["DRY"] == "1"
let MODE = ENV["KEYMODE"] ?? "se"
var typed = 0, totalMs = 0, completed = 0, retried = 0, learned = 0
// Session = one stretch with Dota in front. Snapshot of the counters when it started.
var session: (at: Date, typed: Int, completed: Int, retried: Int, calls: Int, cost: Double)? = nil
func sessionSummary() {
    guard let s = session else { return }
    let calls = (jev?.calls ?? 0) - s.calls, cost = (jev?.cost ?? 0) - s.cost
    log(String(format: "=== session over (%.0fs in Dota): %d typed, %d completed, %d retries, %d Jev calls, cost $%.4f  |  run total so far $%.4f ===",
               Date().timeIntervalSince(s.at), typed - s.typed, completed - s.completed, retried - s.retried, calls, cost, jev?.cost ?? 0))
    session = nil
}
let gpt: GPTSpeller? = ENV["NOGPT"] == "1" ? nil : GPTSpeller()  // NOGPT=1 turns the GPT spelling helper off
func costNow() -> String { String(format: "$%.4f", (jev?.cost ?? 0) + (gpt?.cost ?? 0)) }
signal(SIGINT) { _ in
    sessionSummary()
    let api = (jev.map { "\($0.calls) Jev calls, \($0.tokens) tokens" } ?? "no Jev")
        + (gpt.map { String(format: ", %d GPT calls ($%.4f)", $0.calls, $0.cost) } ?? ", no GPT")
    log("--- multiplier drops: \(resets), names lost without being typed: \(lost) ---")
    log("--- words: \(completed) completed, \(retried) retries, \(learned) new learned (words.txt has \(knownPhrases.count)) ---")
    log("--- run total: \(typed) typed, avg \(typed > 0 ? totalMs / typed : 0)ms each, cost \(costNow()) (\(api)) ---")
    exit(0)
}

log("args: \(ENV.filter { ["KEYMODE", "JEV", "DEBUG", "DRY"].contains($0.key) })")
if DRY { log("DRY RUN: nothing will be typed") }
log("known words: \(knownPhrases.count) in words.txt")
log("Watching for the Dota 2 typing minigame. Typing via \(MODE)\(jev != nil ? ", Jev decides game on/off" : ""). Ctrl+C to stop.")
if MODE == "se" { typeSE("") }  // warm up System Events (first call ~0.8s, then ~35ms)
let grab = Grabber()
var dota = false, checkedAt = Date.distantPast, seen = 0, jevAt = Date.distantPast, timeSeenAt = Date.distantPast, gameSeenAt = Date.distantPast
var doneAt = Date.distantPast
var recent: [String: (left: String, at: Date, verified: Bool)] = [:]  // what we last typed per name, and when
var sightings: [String: Int] = [:]  // verified completions of names not yet in words.txt
// When each name on screen first appeared. All names follow the same path (spawn low at the sides, arc up, drop
// into the bottom centre), so the oldest one is the closest to being lost. A name gone > 1.5s starts over.
var firstSeen: [String: Date] = [:], lastSeen: [String: Date] = [:]
var lastMult = 0.0, resets = 0, lastTypedLog = ""  // multiplier tracking: every drop is logged with frames
var lost = 0, typedThisVisit = Set<String>()  // names that left the screen without ever being typed
/// Last ~3s of frames the bot acted on (copies, so the capture stream's buffers aren't held), dumped on a reset.
var history: [(pb: CVPixelBuffer, at: Date, note: String)] = []
func copyPB(_ src: CVPixelBuffer) -> CVPixelBuffer? {
    var dst: CVPixelBuffer?
    CVPixelBufferCreate(nil, CVPixelBufferGetWidth(src), CVPixelBufferGetHeight(src), kCVPixelFormatType_32BGRA, nil, &dst)
    guard let d = dst else { return nil }
    CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(d, [])
    let rows = CVPixelBufferGetHeight(src), sb = CVPixelBufferGetBytesPerRow(src), db = CVPixelBufferGetBytesPerRow(d)
    for y in 0..<rows {
        memcpy(CVPixelBufferGetBaseAddress(d)! + y * db, CVPixelBufferGetBaseAddress(src)! + y * sb, min(sb, db))
    }
    CVPixelBufferUnlockBaseAddress(d, []); CVPixelBufferUnlockBaseAddress(src, .readOnly)
    return d
}
var typedFrom: [String: CVPixelBuffer] = [:]  // frame each name was last typed from, saved if it needs a retry
func saveJPG(_ pb: CVPixelBuffer, _ path: String) {
    if let jpg = CIContext().jpegRepresentation(of: CIImage(cvPixelBuffer: pb), colorSpace: CGColorSpaceCreateDeviceRGB()) {
        try? jpg.write(to: URL(fileURLWithPath: path))
    }
}
var beatAt = Date(), beatFrames = 0, beatN = 0, lastFront = ""
while true {
    if Date().timeIntervalSince(checkedAt) > 0.5 {  // ~2x/s is plenty
        let o = DRY ? "dry run" : frontOwner(); dota = DRY || o.lowercased().contains("dota"); checkedAt = Date()
        if o != lastFront { log("front window: \(o)\(dota ? "  (Dota, watching)" : "  (not Dota, idle)")"); lastFront = o }
        if dota && session == nil { session = (Date(), typed, completed, retried, jev?.calls ?? 0, jev?.cost ?? 0) }
        if !dota { sessionSummary() }
    }
    guard dota else { grab.stop(); try? await Task.sleep(nanoseconds: 300_000_000); continue }
    if grab.stream == nil {
        if await grab.start() { log("capture started") } else { log("capture: no Dota window found / no Screen Recording permission"); try? await Task.sleep(nanoseconds: 1_000_000_000); continue }
    }
    guard let f = grab.newest(after: seen) else { try? await Task.sleep(nanoseconds: 2_000_000); continue }
    seen = f.seq
    if let jev = jev, Date().timeIntervalSince(jevAt) > JEV_EVERY { jevAt = Date(); jev.checkInBackground(f.pb) }
    for w in gpt?.take() ?? [] where !knownPhrases.contains(w) {
        if let why = gptReject(w) { log("  gpt ignored: \(w) (\(why))"); continue }
        for frag in gptNames where w.contains(frag) && !savedPhrases.contains(frag) {  // "HOLDING UP" once the full phrase is in
            knownPhrases.remove(frag); gptNames.remove(frag); log("  gpt dropped fragment: \(frag)")
        }
        addVocab(w); gptNames.insert(w); log("  gpt added to list: \(w)")
    }
    // Frames captured before the game had time to redraw after our typing still show the old state: skip them.
    if f.at < doneAt.addingTimeInterval(SETTLE) { continue }

    let t = Date()
    let r = readWords(f.pb)
    if r.gameOn { timeSeenAt = f.at }
    // Lit text the local reader can't trust: ask GPT for the spelling (one request at a time, in the background).
    if !r.unreadable.isEmpty, let gpt = gpt, f.at.timeIntervalSince(timeSeenAt) < OVERTIME { gpt.ask(f.pb) }
    let timerOn = jev?.active ?? (f.at.timeIntervalSince(timeSeenAt) < GAME_STICKY)
    if timerOn { gameSeenAt = f.at }
    let ocrMs = ms(t)
    var targets = r.targets
    // No usable label: first the stuck-word rescue (bright text with a cursor), then the old fallback.
    if targets.isEmpty, let p = rescue(r.cands) ?? fallback(r.cands) { targets = [p] }

    // When the clock hits 0:00 words still on screen can be typed: keep going while there are any (overtime).
    let overtime = !timerOn && !targets.isEmpty && f.at.timeIntervalSince(gameSeenAt) < OVERTIME
    let gameOn = timerOn || overtime
    if !gameOn { targets = [] }
    // Heartbeat every 2s: what the bot sees, plus a frame snapshot (debug/live_XX.jpg, last 30 kept).
    beatFrames += 1
    if Date().timeIntervalSince(beatAt) >= 2 {
        log("status: \(beatFrames / 2) fps, ocr \(ocrMs)ms, game \(overtime ? "overtime" : gameOn ? "ON" : "off"), on screen: \(targets.isEmpty ? "none" : targets.map { "\($0.text)[\($0.left)]" }.joined(separator: " "))")
        if let jpg = CIContext().jpegRepresentation(of: CIImage(cvPixelBuffer: f.pb), colorSpace: CGColorSpaceCreateDeviceRGB()) {
            try? jpg.write(to: URL(fileURLWithPath: String(format: "debug/live_%02d.jpg", beatN % 30)))
        }
        beatAt = Date(); beatFrames = 0; beatN += 1
    }
    guard gameOn else { continue }
    if let c = copyPB(f.pb) {
        history.append((c, f.at, "on screen: " + r.targets.map { "\($0.text)[\($0.left)]" }.joined(separator: " ")))
        history.removeAll { f.at.timeIntervalSince($0.at) > 3 }
    }
    if r.mult > 0 {
        if r.mult < lastMult && timerOn {  // a drop on the end screen is just the round ending
            resets += 1
            log("  MULTIPLIER DROP x\(lastMult) -> x\(r.mult)  (last typed: \(lastTypedLog))  -> debug/drop\(resets)_*.jpg")
            // The last 3s of frames, oldest first, so the cause (a lost name or a wrong key) is visible.
            for (i, hf) in history.enumerated() {
                let dt = String(format: "%.1f", hf.at.timeIntervalSince(f.at))
                saveJPG(hf.pb, "debug/drop\(resets)_\(i)_\(dt)s.jpg")
                log("     drop\(resets)_\(i) at \(dt)s: \(hf.note)")
            }
        }
        lastMult = r.mult
    }

    // Bookkeeping: a name we typed whose label is gone (or that has no cursor any more) was accepted.
    let onScreen = Set(targets.map { $0.text })
    for (name, rec) in recent where !onScreen.contains(name) && f.at > rec.at.addingTimeInterval(SETTLE)
        && (!r.seen.contains(name) || f.at > rec.at.addingTimeInterval(LOCK_WAIT)) {
        completed += 1
        if gptNames.contains(name) && !savedPhrases.contains(name) && !rec.verified {
            // From GPT but not confirmed by the big text: save only after it's completed twice.
            sightings[name, default: 0] += 1
            if sightings[name]! >= 2 { learned += 1; learn(name); log("  learned (from GPT, completed twice): \(name)") }
        } else if rec.verified && !savedPhrases.contains(name) {
            sightings[name, default: 0] += 1
            if sightings[name]! >= 2 { learned += 1; learn(name); log("  learned: \(name)") }  // twice = not a one-off misread
        }
        recent[name] = nil; typedFrom[name] = nil
    }
    // Type every name on screen back to back, in priority order:
    //  half-typed first (the game is locked onto it), then the bottom-centre danger zone, then oldest first.
    // Skipped: a name we just typed that still shows the same leftover (screen not caught up yet; retried after
    // REPEAT_GAP). Held: everything, while a name we just typed is on screen but unreadable (it may still be
    // locked, and other letters would go into it).
    let now = Date()
    for name in r.seen.union(targets.map { $0.text }) { if firstSeen[name] == nil { firstSeen[name] = f.at }; lastSeen[name] = f.at }
    for (name, at) in lastSeen where f.at.timeIntervalSince(at) > 1.5 {
        if !typedThisVisit.contains(name), knownPhrases.contains(name), let since = firstSeen[name] {
            lost += 1
            log(String(format: "  LOST: %@ (on screen %.1fs, never typed; meanwhile typed: %@)", name, at.timeIntervalSince(since), lastTypedLog))
        }
        firstSeen[name] = nil; lastSeen[name] = nil; typedThisVisit.remove(name)
    }
    targets.sort { a, b in
        func key(_ t: Target) -> (Int, Int, Double) {
            (t.started ? 0 : 1, DANGER.contains(CGPoint(x: t.x, y: t.y)) ? 0 : 1, firstSeen[t.text]?.timeIntervalSince1970 ?? 0)
        }
        return key(a) < key(b)
    }
    let unreadable = recent.contains { name, rec in r.seen.contains(name) && !onScreen.contains(name) && now.timeIntervalSince(rec.at) < LOCK_WAIT }
    if unreadable { continue }
    // Only the bot types, so a name can only be half-typed if the bot typed part of it: that one is the lock (the
    // game fades every other name until it's finished), so type only it. A "half-typed" name the bot never touched
    // is a misread (often a faded name): retype it whole.
    if let lock = targets.first(where: { $0.started && recent[$0.text] != nil }) {
        targets = [lock]
    } else {
        targets = targets.map { $0.started ? Target(text: $0.text, left: $0.text, x: $0.x, y: $0.y, verified: $0.verified) : $0 }
    }
    let todo = targets.filter { tg in !(recent[tg.text].map { $0.left == tg.left && now.timeIntervalSince($0.at) < REPEAT_GAP } ?? false) }
    if todo.isEmpty { continue }
    let tt = Date()
    for tg in todo {
        if let rec = recent[tg.text] {
            retried += 1; log("  retry: \(tg.text) (last typed \(rec.left), still shows \(tg.left))  -> debug/retry_\(retried)_*.jpg")
            // A retry means a wrong key went in: keep the frame we typed from and the one showing the result.
            if let before = typedFrom[tg.text] { saveJPG(before, "debug/retry_\(retried)_\(tg.text)_1typed-\(rec.left).jpg") }
            saveJPG(f.pb, "debug/retry_\(retried)_\(tg.text)_2now-\(tg.left).jpg")
        }
        if DRY { log("would type: \(tg.left)  (\(tg.text))") } else { type(tg.left, mode: MODE) }
        recent[tg.text] = (tg.left, Date(), tg.verified || (recent[tg.text]?.verified ?? false))
        typedThisVisit.insert(tg.text)
        typedFrom[tg.text] = f.pb
        typed += 1
    }
    totalMs += ms(f.at) * todo.count
    if DEBUG, let jpg = CIContext().jpegRepresentation(of: CIImage(cvPixelBuffer: f.pb), colorSpace: CGColorSpaceCreateDeviceRGB()) {
        try? jpg.write(to: URL(fileURLWithPath: "debug/\(String(format: "%03d", typed))_\(todo.map { $0.left }.joined(separator: "+")).jpg"))
    }
    if !DRY {
        log("typed: \(todo.map { $0.left == $0.text ? $0.left : "\($0.left) (finishing \($0.text))" }.joined(separator: ", "))  [\(MODE)]  total \(ms(f.at))ms (frame wait \(Int(t.timeIntervalSince(f.at) * 1000)) + ocr \(ocrMs) + typing \(ms(tt)))  #\(typed)  cost \(costNow())")
    }
    lastTypedLog = todo.map { $0.left }.joined(separator: ", ")
    doneAt = Date()
}
