"""Builds the inputs component: everything SuperConductor sends, in TouchDesigner.
Run it once from the Textport (Dialogs > Textport and DATs):

    exec(open('/path/to/superconductor_touchdesigner/touchdesigner/build_inputs.py').read())

It creates /project1/inputs (replacing an earlier one); save it as
touchdesigner/superconductor_inputs.tox (TOUCHDESIGNER_SETUP.md). The full list of what
it provides is in TOUCHDESIGNER_PROTOCOL_GETTING_STARTED.md. Inside:

  audio     SuperConductor Playback VST3 Plugin (Audio VST CHOP): the music, as two audio channels.
            Starts as a monitor (plays along with the DAW's copy); pulse
            "Display Plugin GUI" on it for its settings.
  osc       OSC In CHOP on port 7000: loudness, bands, hits, tempo from the one plugin
            copy that has "Send OSC" on (playback/docs/osc.md).
  tracking  WebSocket DAT on the client's ws://localhost:8467/ws/tracking: what the
            Tracking page tracked, as tables (objects, hand_boxes, combos, tracking_info)
            and a CHOP with one sample per object (objects_chop).
  hands     WebSocket DAT on the conducting server's ws://localhost:8480/ws/hands/monitor:
            MediaPipe's 21 points per hand (hand_points, hand_points_chop) and the
            conductor's reply (conduct).
  camera1   Video Device In TOP. add_camera() in the Textport adds camera2, camera3, ...

Coordinates are 0-1 of the camera image, x to the right and y DOWN (image rows), not
mirrored. For TouchDesigner's y-up space use 1 - y.

Change the addresses with the variables below before running, or on the operators.
"""
import os
import sys

CLIENT_HOST, CLIENT_PORT = "localhost", 8467       # sc-collab: /ws/tracking
CONDUCT_HOST, CONDUCT_PORT = "localhost", 8480     # Conducting page's server: /ws/hands/monitor
OSC_PORT = 7000

if sys.platform == "darwin":
    PLUGIN = os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/SuperConductor Playback VST3 Plugin.vst3")
    CAMERA_DRIVER = "avfoundation"
else:
    PLUGIN = "C:/Program Files/Common Files/VST3/SuperConductor Playback VST3 Plugin.vst3"
    CAMERA_DRIVER = "mediafoundation"


TRACKING_CALLBACKS = '''"""/ws/tracking: one overlay per camera frame the client handled
(CAMERA_PROTOCOL.md on client-web). Fills the tables next to this DAT."""
import json


def onConnect(dat):
    op('tracking_info')['connected', 1] = 1


def onDisconnect(dat):
    op('tracking_info')['connected', 1] = 0


def onReceiveText(dat, rowIndex, message):
    try:
        o = json.loads(message)
    except ValueError:
        return
    w = float(o.get('width') or 1)
    h = float(o.get('height') or 1)

    objects = op('objects')
    objects.clear()
    objects.appendRow(['identity', 'track_id', 'class_name', 'label', 'x1', 'y1', 'x2', 'y2',
                       'cx', 'cy', 'coasted', 'combined', 'color'])
    for ob in o.get('objects', []):
        x1, y1, x2, y2 = ob.get('box', [0, 0, 0, 0])
        objects.appendRow([ob.get('identity', ''), ob.get('track_id', ''), ob.get('class_name', ''),
                           ob.get('label', ''), x1 / w, y1 / h, x2 / w, y2 / h,
                           (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                           int(bool(ob.get('coasted'))), int(bool(ob.get('combined'))), ob.get('color', '')])

    boxes = op('hand_boxes')
    boxes.clear()
    boxes.appendRow(['x1', 'y1', 'x2', 'y2'])
    for x1, y1, x2, y2 in o.get('hands', []):
        boxes.appendRow([x1 / w, y1 / h, x2 / w, y2 / h])

    combos = op('combos')
    combos.clear()
    combos.appendRow(['active', 'label'])
    for c in o.get('combos', []):
        combos.appendRow([int(bool(c.get('active'))), c.get('label', '')])

    info = op('tracking_info')
    info['seq', 1] = o.get('seq', '')
    info['width', 1] = int(w)
    info['height', 1] = int(h)
    info['message', 1] = o.get('message') or ''
    status = o.get('status') or {}
    info['vision_fps', 1] = status.get('vision_fps', '')
    info['music', 1] = status.get('music', '')
'''

HANDS_CALLBACKS = '''"""/ws/hands/monitor: each frame's MediaPipe hands from the Conducting page, plus the
conductor's reply (README.md on conducting_sendtoserver). Fills hand_points and conduct."""
import json


def onConnect(dat):
    op('conduct')['connected', 1] = 1


def onDisconnect(dat):
    op('conduct')['connected', 1] = 0


def onReceiveText(dat, rowIndex, message):
    try:
        m = json.loads(message)
    except ValueError:
        return
    points = op('hand_points')
    points.clear()
    points.appendRow(['hand', 'handedness', 'point', 'x', 'y', 'z'])
    handedness = m.get('handedness') or []
    for hand, landmarks in enumerate(m.get('hands') or []):
        side = (handedness[hand] if hand < len(handedness) else None) or ''  # 'Left' / 'Right'
        for i, p in enumerate(landmarks):
            points.appendRow([hand, side, i, p.get('x', 0), p.get('y', 0), p.get('z', 0)])

    conduct = op('conduct')
    conduct['seq', 1] = m.get('seq', '')
    for key, value in (m.get('conduct') or {}).items():
        if isinstance(value, (dict, list)):
            continue
        if conduct.row(key) is None:
            conduct.appendRow([key, ''])
        conduct[key, 1] = '' if value is None else value
'''

# one sample per row of a table, one channel per numeric column (for instancing)
TABLE_TO_CHOP = '''"""Turns the table named by the custom parameter Table into channels: one sample per
row, one channel per column listed in Columns. Re-cooks whenever the table changes."""


def onSetupParameters(scriptOp):
    return


def onCook(scriptOp):
    table = op(scriptOp.fetch('table', ''))
    columns = scriptOp.fetch('columns', [])
    scriptOp.clear()
    if table is None or table.numRows < 1:
        scriptOp.numSamples = 0
        return
    rows = table.numRows - 1
    scriptOp.numSamples = rows
    for name in columns:
        values = []
        for i in range(rows):
            cell = table[i + 1, name]
            try:
                values.append(float(cell.val) if cell is not None else 0.0)
            except ValueError:
                values.append(0.0)
        scriptOp.appendChan(name).vals = values
'''


def _table(parent, name, header, rows=(), x=0, y=0):
    t = parent.create(tableDAT, name)
    t.clear()
    t.appendRow(header)
    for r in rows:
        t.appendRow(r)
    t.nodeX, t.nodeY = x, y
    return t


def _text(parent, name, text, x=0, y=0):
    t = parent.create(textDAT, name)
    t.text = text
    t.nodeX, t.nodeY = x, y
    return t


def _table_chop(parent, name, table, columns, x=0, y=0):
    callbacks = _text(parent, name + '_callbacks', TABLE_TO_CHOP, x, y - 120)
    chop = parent.create(scriptCHOP, name)
    chop.par.callbacks = callbacks
    chop.store('table', table.path)
    chop.store('columns', list(columns))
    chop.nodeX, chop.nodeY = x, y
    return chop


def add_camera(parent=None, device=None):
    """adds the next Video Device In TOP (camera2, camera3, ...); pick its Device on the op"""
    parent = parent or op('/project1/inputs') or op('/inputs')
    n = 1
    while parent.op(f'camera{n}') is not None:
        n += 1
    cam = parent.create(videodeviceinTOP, f'camera{n}')
    cam.par.driver = CAMERA_DRIVER
    if device is not None:
        cam.par.device = device
    cam.nodeX, cam.nodeY = 1200, -200 * (n - 1)
    return cam


def build(parent_path=None):
    # /project1 in a normal project; the root when TouchDesigner was opened another way
    parent = op(parent_path) if parent_path else (op('/project1') or op('/'))
    old = parent.op('inputs')
    linked = old.par.externaltox.eval() if old is not None else ''
    if old is not None:
        old.destroy()
    sc = parent.create(baseCOMP, 'inputs')
    if linked:
        # keep the link to the .tox file (superconductor.toe loads the component from it)
        sc.par.externaltox = linked
        sc.par.enableexternaltox = True

    # audio: the plugin, as a monitor (its own default inside TouchDesigner)
    audio = sc.create(audiovstCHOP, 'audio')
    audio.par.file = PLUGIN
    audio.par.rate = 48000  # the stream's rate: the plugin then passes it through unchanged
    audio.nodeX, audio.nodeY = 0, 0

    osc = sc.create(oscinCHOP, 'osc')
    osc.par.port = OSC_PORT
    osc.par.active = True
    osc.nodeX, osc.nodeY = 0, -200

    # tracking from the Tracking page
    objects = _table(sc, 'objects', ['identity', 'track_id', 'class_name', 'label', 'x1', 'y1', 'x2', 'y2',
                                     'cx', 'cy', 'coasted', 'combined', 'color'], x=400, y=0)
    _table(sc, 'hand_boxes', ['x1', 'y1', 'x2', 'y2'], x=400, y=-150)
    _table(sc, 'combos', ['active', 'label'], x=400, y=-300)
    _table(sc, 'tracking_info', ['key', 'value'],
           [[k, ''] for k in ('connected', 'seq', 'width', 'height', 'message', 'vision_fps', 'music')], x=400, y=-450)
    tracking_cb = _text(sc, 'tracking_callbacks', TRACKING_CALLBACKS, x=200, y=-120)
    tracking = sc.create(websocketDAT, 'tracking')
    tracking.par.netaddress = f'{CLIENT_HOST}/ws/tracking'
    tracking.par.port = CLIENT_PORT
    tracking.par.callbacks = tracking_cb
    tracking.par.active = True
    tracking.nodeX, tracking.nodeY = 200, 0
    _table_chop(sc, 'objects_chop', objects, ['identity', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'coasted', 'combined'],
                x=600, y=0)

    # MediaPipe hands from the Conducting page
    points = _table(sc, 'hand_points', ['hand', 'handedness', 'point', 'x', 'y', 'z'], x=900, y=0)
    _table(sc, 'conduct', ['key', 'value'], [['connected', ''], ['seq', '']], x=900, y=-150)
    hands_cb = _text(sc, 'hands_callbacks', HANDS_CALLBACKS, x=800, y=-120)
    hands = sc.create(websocketDAT, 'hands')
    hands.par.netaddress = f'{CONDUCT_HOST}/ws/hands/monitor'
    hands.par.port = CONDUCT_PORT
    hands.par.callbacks = hands_cb
    hands.par.active = True
    hands.nodeX, hands.nodeY = 800, 0
    _table_chop(sc, 'hand_points_chop', points, ['hand', 'point', 'x', 'y', 'z'], x=1000, y=-300)

    add_camera(sc)
    print(f'built {sc.path}: audio (SuperConductor Playback VST3 Plugin, {PLUGIN}), osc :{OSC_PORT}, '
          f'tracking ws://{CLIENT_HOST}:{CLIENT_PORT}/ws/tracking, '
          f'hands ws://{CONDUCT_HOST}:{CONDUCT_PORT}/ws/hands/monitor, camera1')
    return sc


build()
