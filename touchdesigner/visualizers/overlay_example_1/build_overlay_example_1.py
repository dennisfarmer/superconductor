"""overlay_example_1: one example visualizer. Everything /inputs receives, drawn over
the webcam.

overlay_example_1.toe (next to this file) already contains it. This script rebuilds the
`overlay_example_1` component from scratch, e.g. after you change it here. Run it in
overlay_example_1.toe's Textport (Dialogs > Textport and DATs), then save the project
(Cmd+S):

    exec(open('/path/to/superconductor_touchdesigner/touchdesigner/visualizers/overlay_example_1/build_overlay_example_1.py').read())

What it draws:

  webcam      camera1, as the background
  hands       MediaPipe's 21 points per hand (fingertips bigger; orange = right hand,
              blue = left), growing with the music's loudness
  boxes       the camera page's tracked objects (their colour) and hand boxes (cyan)
  bars        the frequency bands from OSC, bottom left
  beat        one dot per beat of the bar, top right; the current beat is big
  flash       a white flash on every detected hit (onset)
  text        tempo, beat, loudness, the conductor's status, the camera page's message

How it's built: Script CHOPs turn the tables / OSC channels into one sample per shape
(tx, ty, sx, sy, r, g, b); Geometry COMPs draw one shape per sample (instancing); a
Render TOP with an orthographic camera draws them in 0-1 image coordinates; Over TOPs
put that, the text and the flash over the webcam. `out` is the result; `window`
shows it full screen (pulse its Open parameter; Esc closes it).
"""
W, H = 1280, 720
ASPECT = H / W  # world units: x 0..1 across the image, y 0..ASPECT up it

# ---- Script CHOP callbacks: data -> one sample per shape ------------------------------

COMMON = '''ASPECT = %r


def onSetupParameters(scriptOp):
    return


def onGetCookLevel(scriptOp):
    return CookLevel.ALWAYS  # follow the live data every frame


def channels(scriptOp, columns):
    """columns: {name: [values]}, all the same length"""
    scriptOp.clear()
    n = len(next(iter(columns.values()))) if columns else 0
    scriptOp.numSamples = n
    for name, values in columns.items():
        scriptOp.appendChan(name).vals = values


def num(cell, default=0.0):
    try:
        return float(cell.val)
    except (AttributeError, ValueError):
        return default


def osc(name, default=0.0):
    c = op('../inputs/osc')
    ch = c[name] if c is not None else None
    return ch.eval() if ch is not None else default
''' % ASPECT

HANDS = COMMON + '''

def onCook(scriptOp):
    t = op('../inputs/hand_points')
    level = osc('sc/level')
    cols = {k: [] for k in ('tx', 'ty', 'sx', 'r', 'g', 'b')}
    for i in range(1, t.numRows):
        point = int(num(t[i, 'point']))
        right = t[i, 'handedness'].val == 'Right'
        size = (0.012 if point in (4, 8, 12, 16, 20) else 0.006) * (1.0 + 0.6 * level)
        cols['tx'].append(num(t[i, 'x']))
        cols['ty'].append((1.0 - num(t[i, 'y'])) * ASPECT)
        cols['sx'].append(size)
        cols['r'].append(1.0 if right else 0.3)
        cols['g'].append(0.45 if right else 0.8)
        cols['b'].append(0.2 if right else 1.0)
    channels(scriptOp, cols)
'''

BOXES = COMMON + '''

def hex_rgb(text, fallback):
    text = (text or '').lstrip('#')
    try:
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return fallback


def onCook(scriptOp):
    cols = {k: [] for k in ('tx', 'ty', 'sx', 'sy', 'r', 'g', 'b')}

    def box(x1, y1, x2, y2, rgb):
        cols['tx'].append((x1 + x2) / 2)
        cols['ty'].append((1.0 - (y1 + y2) / 2) * ASPECT)
        cols['sx'].append(x2 - x1)
        cols['sy'].append((y2 - y1) * ASPECT)
        for k, v in zip('rgb', rgb):
            cols[k].append(v)

    objects = op('../inputs/objects')
    for i in range(1, objects.numRows):
        box(*(num(objects[i, k]) for k in ('x1', 'y1', 'x2', 'y2')),
            hex_rgb(objects[i, 'color'].val, (1.0, 1.0, 1.0)))
    hands = op('../inputs/hand_boxes')
    for i in range(1, hands.numRows):
        box(*(num(hands[i, k]) for k in ('x1', 'y1', 'x2', 'y2')), (0.3, 0.9, 1.0))
    channels(scriptOp, cols)
'''

BARS = COMMON + '''

def onCook(scriptOp):
    c = op('../inputs/osc')
    bands = [ch.eval() for ch in c.chans('sc/bands*')] if c is not None else []
    left, width, gap, tallest = 0.03, 0.30, 0.004, 0.14
    w = width / max(1, len(bands))
    cols = {k: [] for k in ('tx', 'ty', 'sx', 'sy', 'r', 'g', 'b')}
    for i, v in enumerate(bands):
        h = max(0.002, v * tallest)
        cols['tx'].append(left + (i + 0.5) * w)
        cols['ty'].append(0.03 + h / 2)
        cols['sx'].append(w - gap)
        cols['sy'].append(h)
        f = i / max(1, len(bands) - 1)  # low = magenta, high = yellow
        cols['r'].append(1.0)
        cols['g'].append(0.25 + 0.7 * f)
        cols['b'].append(0.8 * (1 - f))
    channels(scriptOp, cols)
'''

BEAT = COMMON + '''

def onCook(scriptOp):
    beats = int(osc('sc/tempo/beats_per_bar', 4)) or 4
    known = osc('sc/tempo/beat_known') > 0.5 and osc('sc/tempo/playing') > 0.5
    beat, frac = int(osc('sc/tempo/beat')), osc('sc/tempo/beat_frac')
    cols = {k: [] for k in ('tx', 'ty', 'sx', 'r', 'g', 'b')}
    for i in range(beats):
        now = known and i == beat
        cols['tx'].append(0.97 - (beats - 1 - i) * 0.035)
        cols['ty'].append(ASPECT - 0.035)
        cols['sx'].append(0.008 + (0.010 * (1.0 - frac) if now else 0.0))
        bright = 1.0 if now else 0.35
        cols['r'].append(bright)
        cols['g'].append(bright)
        cols['b'].append(bright if i else 0.3 * bright)  # beat one is yellow
    channels(scriptOp, cols)
'''

PULSE = COMMON + '''

def onCook(scriptOp):
    # a hit sets it to 1; it fades by 15 percent a frame
    v = scriptOp.fetch('v', 0.0) * 0.85
    if osc('sc/onset') > 0:
        v = 1.0
    scriptOp.store('v', v)
    channels(scriptOp, {'flash': [v]})
'''

INFO_TEXT = '''"""The text line. The `info` Text TOP shows mod('info_text').line()."""


def value(chop, name, default=0.0):
    c = op(chop)
    ch = c[name] if c is not None else None
    return ch.eval() if ch is not None else default


def cell(table, row):
    t = op(table)
    c = t[row, 1] if t is not None else None
    return c.val if c is not None else ''


def line():
    osc = '../inputs/osc'
    return '{:.0f} BPM   beat {}/{}   level {:.2f}   conductor: {}   {}'.format(
        value(osc, 'sc/tempo/bpm'),
        int(value(osc, 'sc/tempo/beat')) + 1,
        int(value(osc, 'sc/tempo/beats_per_bar', 4)),
        value(osc, 'sc/level'),
        cell('../inputs/conduct', 'status') or '-',
        cell('../inputs/tracking_info', 'message'))
'''


def _script_chop(parent, name, code, x, y):
    dat = parent.create(textDAT, name + '_callbacks')
    dat.text = code
    dat.nodeX, dat.nodeY = x, y - 120
    chop = parent.create(scriptCHOP, name)
    chop.par.callbacks = dat
    chop.nodeX, chop.nodeY = x, y
    return chop


def _material(parent, name, alpha, x, y):
    mat = parent.create(constantMAT, name)
    mat.par.alpha = alpha
    mat.par.blending = True
    mat.par.depthtest = False  # flat shapes: later ones simply draw over earlier ones
    mat.nodeX, mat.nodeY = x, y
    return mat


def _shapes(parent, name, sop_type, chop, material, x, y, scale_y=None):
    """a Geometry COMP drawing one `sop_type` per sample of `chop`"""
    geo = parent.create(geometryCOMP, name)
    for child in list(geo.children):
        child.destroy()
    sop = geo.create(sop_type, 'shape')
    if sop_type == circleSOP:
        sop.par.divs = 24
        sop.par.radx = sop.par.rady = 1
    else:
        sop.par.sizex = sop.par.sizey = 1
    sop.display = sop.render = True
    geo.par.instancing = True
    geo.par.instanceop = chop
    geo.par.instancetx, geo.par.instancety = 'tx', 'ty'
    geo.par.instancesx = 'sx'
    geo.par.instancesy = scale_y or 'sx'
    geo.par.instancecolorop = chop
    geo.par.instancecolormode = 'replace'
    geo.par.instancer, geo.par.instanceg, geo.par.instanceb = 'r', 'g', 'b'
    geo.par.material = material
    geo.nodeX, geo.nodeY = x, y
    return geo


def build_overlay_example_1(parent=None):
    parent = parent or op('/project1') or op('/')
    if parent.op('inputs') is None:
        raise RuntimeError('load superconductor_inputs.tox first: the overlay reads from /inputs')
    old = parent.op('overlay_example_1')
    if old is not None:
        old.destroy()
    overlay = parent.create(containerCOMP, 'overlay_example_1')

    # data -> shapes
    hands = _script_chop(overlay, 'hand_shapes', HANDS, 0, 0)
    boxes = _script_chop(overlay, 'box_shapes', BOXES, 200, 0)
    bars = _script_chop(overlay, 'bar_shapes', BARS, 400, 0)
    beat = _script_chop(overlay, 'beat_shapes', BEAT, 600, 0)
    pulse = _script_chop(overlay, 'flash', PULSE, 800, 0)

    solid = _material(overlay, 'solid', 1.0, 0, -500)
    glass = _material(overlay, 'glass', 0.3, 200, -500)

    geos = [
        _shapes(overlay, 'boxes_geo', rectangleSOP, boxes, glass, 200, -300, scale_y='sy'),
        _shapes(overlay, 'bars_geo', rectangleSOP, bars, solid, 400, -300, scale_y='sy'),
        _shapes(overlay, 'hands_geo', circleSOP, hands, solid, 0, -300),
        _shapes(overlay, 'beat_geo', circleSOP, beat, solid, 600, -300),
    ]

    cam = overlay.create(cameraCOMP, 'ortho')
    cam.par.projection = 'ortho'
    cam.par.orthoorigin = 'bottomleft'
    cam.par.orthowidth = 1
    cam.par.tx, cam.par.ty, cam.par.tz = 0, 0, 5
    cam.nodeX, cam.nodeY = 800, -300

    render = overlay.create(renderTOP, 'render')
    render.par.camera = cam
    render.par.geometry = ' '.join(g.name for g in geos)
    render.par.outputresolution = 'custom'
    render.par.resolutionw, render.par.resolutionh = W, H
    render.par.bgcolora = 0
    render.nodeX, render.nodeY = 1000, -300

    # the picture: webcam, shapes, text, flash
    webcam = overlay.create(selectTOP, 'webcam')
    webcam.par.top = '../inputs/camera1'
    webcam.nodeX, webcam.nodeY = 1000, 0

    text = overlay.create(textDAT, 'info_text')
    text.text = INFO_TEXT
    text.nodeX, text.nodeY = 1000, -620
    info = overlay.create(textTOP, 'info')
    info.par.text.expr = "mod('info_text').line()"
    info.par.outputresolution = 'custom'
    info.par.resolutionw, info.par.resolutionh = W, H
    info.par.alignx, info.par.aligny = 'left', 'top'
    info.par.fontsizex = 22
    info.par.bgalpha = 0
    info.nodeX, info.nodeY = 1000, -500

    flash = overlay.create(constantTOP, 'flash_top')
    flash.par.outputresolution = 'custom'
    flash.par.resolutionw, flash.par.resolutionh = W, H
    flash.par.alpha.expr = "op('flash')['flash'] * 0.35"
    flash.nodeX, flash.nodeY = 1000, -700

    def over(name, top, under, x):
        o = overlay.create(overTOP, name)
        o.inputConnectors[0].connect(top)
        o.inputConnectors[1].connect(under)
        o.nodeX, o.nodeY = x, -300
        return o

    picture = over('over_shapes', render, webcam, 1200)
    picture = over('over_text', info, picture, 1400)
    picture = over('over_flash', flash, picture, 1600)
    out = overlay.create(nullTOP, 'out')
    out.inputConnectors[0].connect(picture)
    out.nodeX, out.nodeY = 1800, -300

    window = overlay.create(windowCOMP, 'window')
    window.par.winop = out
    window.par.size = 'fill'
    window.par.borders = False
    window.par.closeescape = True  # Esc leaves full screen
    window.nodeX, window.nodeY = 2000, -300

    # the container shows the result in the Network Editor and in perform mode
    overlay.par.w, overlay.par.h = W, H
    overlay.par.top = out
    print(f'built {overlay.path}: open {window.path} (pulse Open) for full screen')
    return overlay


build_overlay_example_1()
