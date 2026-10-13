# How SuperConductor Atlas picks chords

## The short version (what to tell judges)

You're sitting on a chord. The app answers: **"What chord should come next, and
where does each choice lead?"**

- Every chord is a **node** in a graph. Every allowed move between chords is an
  **edge**.
- A Jac **walker** starts on your current chord and walks outward along the
  edges, a few chords deep.
- Every move it tries gets a **penalty score**. **Lower is better.**
- After each step outward, it throws away all but the best few paths, then
  keeps walking from the survivors.
- You see the survivors as a map and can click any path to hear it.

The rest of this document explains the score, then the search, then where Jac
fits in.

---

## 1. The score

Every move from one chord to the next gets a number. It is the sum of three
penalties:

```
move score  =  1.5 × grammar  +  voice leading  +  phrase
path score  =  sum of its move scores
```

| Part | The question it answers | Example |
|---|---|---|
| **Grammar** | Is this chord change normal in classical music? | V → I is the most normal ending there is, so it costs 0.2. Going *backwards* (V → IV) costs 4.0. |
| **Voice leading** | Can the four singers (bass, tenor, alto, soprano) get to the next chord smoothly, without breaking the classic rules? | Moving everyone in parallel fifths costs 20 per pair. Holding a shared note costs −0.2. |
| **Phrase** | Does this move make sense given the whole path so far? | Bouncing back to a chord you just left costs +1.6. Landing a proper ending (a *cadence*) *subtracts* up to 0.9. |

Grammar is multiplied by 1.5 because a wrong chord is worse than an awkward
voicing of the right one.

### Worked example: C → G → C (I → V → I), real numbers from the app

We start on a C major chord with the four voices on **C3 E3 C4 G4**.

**Move 1: C → G (I → V)**

| Part | Value | Why |
|---|---|---|
| Grammar | 0.6 | "tonic → dominant", a normal move |
| Voice leading | 1.1 | The voices move to **G2 D3 B3 G4**. The soprano stays on G (a held common tone). The others move a little, and each semitone of movement costs a bit. |
| Phrase | −0.45 | Stopping on V is a *half cadence*, a legitimate place to pause, so it gets a bonus. |
| **Move score** | 1.5 × 0.6 + 1.1 − 0.45 = **1.55** | |

**Move 2: G → C (V → I)**

| Part | Value | Why |
|---|---|---|
| Grammar | 0.2 | "authentic resolution", the strongest ending in the language |
| Voice leading | 1.1 | Back to **C3 E3 C4 G4**. The B in the alto rises to C, which is exactly what the rules require of that note (the *leading tone*). |
| Phrase | −0.5 | Ends the phrase with an authentic cadence, so it gets a bonus. |
| **Move score** | 1.5 × 0.2 + 1.1 − 0.5 = **0.9** | |

**Path score: 1.55 + 0.9 = 2.45**

### What a bad move looks like

Same starting chord, moving to D minor (I → ii). The grammar is fine (0.45).
What changes is *how* the four voices get there:

| Voicing of D minor | Voice-leading penalty | Why |
|---|---|---|
| Everyone steps up together: **D3 F3 D4 A4** | **63.2** | parallel octaves (20) + two sets of parallel fifths (20 + 20) + all voices moving the same direction (1.5) |
| The best one the app finds: **D3 F3 A3 D4** | **3.0** | no rule broken, only the cost of moving |

Same chord change, two very different scores. That's why the app can't score a
chord change once and reuse the number: the score depends on **exactly which
notes you're on**, and that depends on every choice made before. This is the
reason the search is done by a walker that carries each path's notes with it.

Every penalty and bonus also comes with a one-line explanation ("leading tone in
the alto (B3→C4) resolves", "parallel 5ths (B–S)"). The app shows these, so
every point of a score can be traced to a specific rule.

---

## 2. The search: explore, score, cut, repeat

Scoring *every* possible 4-chord continuation isn't practical. In C major the
app knows 56 chords, each chord has about 20 allowed next chords, and every
move has many possible voicings. That's roughly **430,000 paths** four chords
deep.

So the walker works one layer at a time:

1. **Explore:** from every surviving path, try every allowed next chord and
   score it.
2. **Cut:** keep the best few and throw the rest away.
3. **Repeat** from the survivors.

Here's a real run from C major (the numbers the app's "Walker log" panel shows):

| Step | Paths tried | Kept | Thrown away |
|---|---|---|---|
| 1 chord ahead | 54 | 8 | 46 |
| 2 chords ahead | 220 | 12 | 205 |
| 3 chords ahead | 366 | 12 | 351 |
| 4 chords ahead | 320 | 12 | 306 |

That's **960 moves scored instead of ~430,000**. It takes about half a second.

(Kept + thrown away is sometimes less than tried. A few paths that land a
cadence are set aside as "destinations" instead. See the third rule below.)

### How it decides what to keep

A plain "keep the top 8" would show 8 variations of the same idea, so the cut
has three extra rules:

- **One version of each chord.** V⁶₅ scored 1.68 and was *cut*, even though it
  beat two chords that stayed. The reason is that V⁴₃ is the same chord (G7)
  with a different note in the bass, and V⁴₃ scored better (1.46). Showing both
  is redundant.
- **At least one option of each kind.** B♭ major (♭VII) scored 3.76, the worst
  on the list, but it *stays*. It's the only "borrowed colour" chord, and
  without this rule you would never be offered one.
- **Every option shows a future.** At every later step, the best continuation
  of each of the 8 first choices survives, so every option on the menu shows
  where it leads.

The best path to each **type of ending** (full close, half close, deceptive
ending, and so on) is also saved, even if it would otherwise be cut. If you
want a deceptive ending, you see the best one.

The menu from that run (score in brackets):
**V⁴₂/IV (1.11), vi (1.34), iii (1.44), V⁴₃ (1.46), V (1.55), IV⁶ (1.61), V⁷/ii (1.69), ♭VII (3.76)**

---

## 3. Where the graph comes in

```
   [C] ──0.45──▶ [F] ──0.40──▶ [G] ──0.20──▶ [C]
    │                           ▲
    └──────────0.60─────────────┘
         (number on each arrow = grammar penalty)
```

- **Nodes** are chords (`node Chord` in `graph/harmony.jac`).
- **Edges** are allowed moves (`edge Progression`). Each edge stores its grammar
  penalty and a sentence explaining it, for example
  `cost=0.2, rule="authentic resolution (D → T)"`.
- **Missing edges are banned moves.** If grammar says a move is illegal (for
  example, a chord that *must* resolve to G going somewhere else), no edge is
  ever made, so the walker can't take it. Of the 3,136 possible chord pairs in
  C major, only 1,123 become edges.
- **The graph builds itself as you explore.** At the start it holds a single
  chord. The first time the walker arrives at a chord, a helper walker
  (`Expand`) creates that chord's outgoing edges. After one search, 25 of the 56
  chords have been filled in. A second search reuses them and runs about 4× faster.

The **grammar** part of the score lives on the edges, because it depends only
on the two chords. The **voice-leading** and **phrase** parts depend on the path,
so the walker computes them as it walks.

---

## 4. Where Jac comes in

Jac is a language where graphs and "walkers that travel through graphs" are
built in, rather than something you'd build out of classes and loops. Each
feature below maps to one thing the app does:

| What the app needs | The Jac feature | Code |
|---|---|---|
| Chords and moves stored as a graph | `node Chord`, `edge Progression` with fields | `graph/harmony.jac:19, 33` |
| Create a move with its penalty attached | `here +>:Progression(cost=…, rule=…):+> next_chord` | `graph/harmony.jac:77` |
| "Which chords can I go to from here?" | `[here ->:Progression:->]` (follows only move edges) | `graph/beam.jac:108` |
| "What's the grammar penalty of each move?" | `[edge here ->:Progression:->]` (returns the edges themselves) | `graph/beam.jac:109` |
| Something that travels the graph carrying each path's notes and scores | `walker BeamSearch` | `graph/beam.jac:74` |
| Do different work at different stops | `can extend with Chord entry` scores moves, `can checkpoint with BeamGate entry` does the cutting | `graph/beam.jac:99, 153` |
| Go to all surviving chords, then come back to cut | `visit stops; visit here;` (the walker's to-do list) | `graph/beam.jac:179` |
| Build the graph only where we've explored | the search walker launches a second walker: `here spawn Expand()` | `graph/beam.jac:100` |
| Stop when done | `disengage` | `graph/beam.jac:164` |
| Remember the graph and the user's progression between clicks | attach them to `root` | `graph/harmony.jac:93`, `services/atlas.jac:189` |
| The browser calls the search directly | `def:pub explore` on the server, `sv import` in the UI | `services/atlas.jac:398`, `frontend.cl.jac:14` |
| Describe a path in words (optional) | `by llm` with `sem` hints. The LLM only *describes* a path the search already chose. It never picks chords. | `services/atlas.jac:165` |

### The one idea worth remembering

**The graph stores what never changes; the walker carries what does.**

- *What never changes:* which chord moves are allowed and how normal they are.
  That's stored once, on the edges.
- *What changes per path:* which exact notes you're on, what you've already
  played, and the running score. The walker carries that as it moves.

That split is what lets 960 scored moves stand in for 430,000.

---

## Honest limitations

- Cutting early means the app can miss a path that starts badly but ends
  brilliantly. The "keep one of each kind" and "every option shows a future"
  rules reduce this but don't eliminate it.
- The penalty numbers were set by hand from standard music-theory textbook
  rules. They weren't learned from data.
- The music-theory rules themselves (`theory/`) are ordinary functions written
  in Jac. The graph and walker features are in `graph/` and `services/`.
