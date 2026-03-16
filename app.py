import os
import uuid
import io
import base64
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from main import apply_preset, load_image, load_presets

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

PRESETS = load_presets(os.path.join(os.path.dirname(__file__), 'presets.json'))

PREVIEW_SIZE = 1000
THUMB_SIZE = 150


def session_path(sid):
    return os.path.join(UPLOAD_DIR, sid + '.meta.json')


def save_session(sid, data):
    with open(session_path(sid), 'w') as f:
        json.dump(data, f)


def load_session(sid):
    p = session_path(sid)
    if not os.path.exists(p):
        return None
    with open(p, 'r') as f:
        return json.load(f)


def img_to_data_uri(img_float, quality=80):
    uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(uint8)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def resize_array(img_float, max_size):
    h, w = img_float.shape[:2]
    if max(h, w) <= max_size:
        return img_float
    scale = max_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    pil = Image.fromarray((np.clip(img_float, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def apply_rotation(img, rotation):
    rot = rotation % 360
    if rot == 90:
        return np.rot90(img, k=3)
    elif rot == 180:
        return np.rot90(img, k=2)
    elif rot == 270:
        return np.rot90(img, k=1)
    return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify(error='No file'), 400
    f = request.files['image']
    if not f.filename:
        return jsonify(error='No file selected'), 400

    sid = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_DIR, sid + '.jpg')
    f.save(img_path)

    save_session(sid, {'rotation': 0, 'path': img_path})

    img = load_image(img_path, max_size=PREVIEW_SIZE)
    main_uri = img_to_data_uri(img, quality=82)
    return jsonify(session=sid, main=main_uri)


@app.route('/presets-stream')
def presets_stream():
    sid = request.args.get('session')
    s = load_session(sid)
    if not s:
        return jsonify(error='Invalid session'), 400

    def generate():
        img = load_image(s['path'], max_size=PREVIEW_SIZE)
        rotated = apply_rotation(img, s['rotation'])
        preview_src = resize_array(rotated, PREVIEW_SIZE)
        thumb_src = resize_array(rotated, THUMB_SIZE)

        for name, params in PRESETS.items():
            thumb = img_to_data_uri(apply_preset(thumb_src, params), quality=65)
            preview = img_to_data_uri(apply_preset(preview_src, params), quality=82)

            payload = json.dumps({'name': name, 'thumb': thumb, 'preview': preview})
            yield f"data: {payload}\n\n"

        yield "data: {\"done\": true}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/set-rotation', methods=['POST'])
def set_rotation():
    data = request.json
    sid = data.get('session')
    s = load_session(sid)
    if not s:
        return jsonify(error='Invalid session'), 400
    s['rotation'] = data.get('rotation', 0) % 360
    save_session(sid, s)
    return jsonify(ok=True)


@app.route('/download', methods=['POST'])
def download():
    data = request.json
    sid = data.get('session')
    preset_name = data.get('preset', 'Original')
    s = load_session(sid)
    if not s:
        return jsonify(error='Invalid session'), 400

    full_img = load_image(s['path'], max_size=4000)
    rotated = apply_rotation(full_img, s['rotation'])

    intensity = data.get('intensity', 1.0)

    if preset_name in PRESETS and preset_name != 'Original':
        preset_result = apply_preset(rotated, PRESETS[preset_name])
        result = rotated * (1 - intensity) + preset_result * intensity
    else:
        result = rotated

    uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(uint8, 'RGB')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)

    filename = f"edited_{preset_name.replace(' ', '_')}.png"
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
