"""How to type chords: every spelling the parser (theory/vocab.jac parse_chord) accepts,
shown in the side panel's Typing tab."""

"""(what to type, what it means) - every suffix uses plain keyboard characters."""
glob GUIDE_QUALITIES: list[list[str]] = [
    ["C", "major"],
    ["Cm", "minor"],
    ["Cdim  or  Co", "diminished"],
    ["Caug  or  C+", "augmented"],
    ["C7", "dominant 7th"],
    ["Cmaj7  or  CM7", "major 7th"],
    ["Cm7", "minor 7th"],
    ["Cm7b5", "half-diminished 7th (ø7)"],
    ["Cdim7  or  Co7", "fully diminished 7th"],
    ["Csus4  /  Csus2", "suspended"],
    ["C7sus4", "dominant 7th, suspended"],
    ["C6  /  Cm6", "added 6th"]
];

glob GUIDE_ROMAN: list[list[str]] = [
    ["I  ii  iii  IV  V  vi  vii", "upper case major, lower case minor"],
    ["viio  /  vii°", "diminished"],
    ["viiø7", "half-diminished 7th"],
    ["viio7", "fully diminished 7th"],
    ["V+", "augmented"],
    ["V7  ii7  IV7", "the key's own 7th (IV7 = IVmaj7)"],
    ["IVmaj7  /  IVM7", "major 7th, explicitly"],
    ["I6  I64", "1st / 2nd inversion"],
    ["V65  V43  V42", "7th chord inversions"],
    ["bVI  bVII  #iv", "chromatic roots"],
    ["V/V  V7/ii  viio7/V", "applied chords"],
    ["N6  It6  Fr6  Ger6", "Neapolitan, augmented 6ths"],
    ["Cad64", "cadential 6/4"]
];

glob GUIDE_BASS: list[list[str]] = [
    ["C/E", "1st inversion (3rd in the bass)"],
    ["C/G", "2nd inversion (5th in the bass)"],
    ["C7/Bb", "3rd inversion (7th in the bass)"]
];


def:pub TypingGuide -> JsxElement {
    return <div className="guide">
        <div>
            <h4>Root</h4>
            <p>A letter A–G, then <code>b</code> for flat or <code>#</code> for sharp: <code>Bb</code>, <code>F#</code>, <code>Ebb</code>.
            The letter matters: <code>D#</code> and <code>Eb</code> are different chords to a theorist.</p>
            <h4>Bass note (inversions)</h4>
            <table><tbody>
                {for (i, row) in enumerate(GUIDE_BASS) {
                    <tr key={"b" + str(i)}><td><code>{row[0]}</code></td><td>{row[1]}</td></tr>
                }}
            </tbody></table>
            <h4>Separating chords</h4>
            <p>Spaces, commas or arrows: <code>{"Gm/D -> Cm/Eb, D/A"}</code>.</p>
        </div>
        <div>
            <h4>Chord type (root C shown)</h4>
            <table><tbody>
                {for (i, row) in enumerate(GUIDE_QUALITIES) {
                    <tr key={"q" + str(i)}><td><code>{row[0]}</code></td><td>{row[1]}</td></tr>
                }}
            </tbody></table>
        </div>
        <div>
            <h4>Or Roman numerals (in the current key)</h4>
            <table><tbody>
                {for (i, row) in enumerate(GUIDE_ROMAN) {
                    <tr key={"r" + str(i)}><td><code>{row[0]}</code></td><td>{row[1]}</td></tr>
                }}
            </tbody></table>
            <p>Superscripts work too (<code>ii⁶₅</code>), and you can mix symbols and numerals.</p>
        </div>
        <div>
            <h4>What happens</h4>
            <p>Each chord is analysed in the current key (in F major, <code>Gm/D</code> becomes ii⁶₄ and
            <code>D/A</code> becomes V⁶₄/ii) and voiced to lead smoothly from its neighbour.</p>
            <p><b>Before first</b> works backwards from your first chord, so you can set a reference
            chord and build in both directions. Moves the walkers' rules don't cover are still allowed;
            they're labelled as typed.</p>
        </div>
    </div>;
}
