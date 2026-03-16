import os
import uuid
import json
import io
import base64
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, render_template, request, jsonify, send_file
from main import apply_preset, load_image, load_presets

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

PRESETS = load_presets(os.path.join(os.path.dirname(__file__), 'presets.json'))

# In-memory store: session_id -> { 'original': np.array, 'rotation': int }
sessions = {}


def img_to_data_uri(img_float, quality=85):
    uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(uint8)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def get_rotated(session):
    img = session['original']
    rot = session['rotation'] % 360
    if rot == 90:
        img = np.rot90(img, k=3)
    elif rot == 180:
        img = np.rot90(img, k=2)
    elif rot == 270:
        img = np.rot90(img, k=1)
    return img


def make_thumbnail(img_float, max_size=200):
    h, w = img_float.shape[:2]
    if max(h, w) <= max_size:
        return img_float
    scale = max_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    pil = Image.fromarray((np.clip(img_float, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


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
    path = os.path.join(UPLOAD_DIR, sid + '.jpg')
    f.save(path)

    img = load_image(path, max_size=1200)
    sessions[sid] = {'original': img, 'rotation': 0, 'path': path}

    rotated = get_rotated(sessions[sid])
    main_uri = img_to_data_uri(rotated)

    # Generate thumbnails for all presets
    thumb_src = make_thumbnail(rotated)
    thumbnails = {}
    for name, params in PRESETS.items():
        result = apply_preset(thumb_src, params)
        thumbnails[name] = img_to_data_uri(result, quality=70)

    return jsonify(session=sid, main=main_uri, thumbnails=thumbnails)


@app.route('/rotate', methods=['POST'])
def rotate():
    data = request.json
    sid = data.get('session')
    direction = data.get('direction', 'cw')
    if sid not in sessions:
        return jsonify(error='Invalid session'), 400

    sessions[sid]['rotation'] = (sessions[sid]['rotation'] + (90 if direction == 'cw' else -90)) % 360
    rotated = get_rotated(sessions[sid])

    main_uri = img_to_data_uri(rotated)

    thumb_src = make_thumbnail(rotated)
    thumbnails = {}
    for name, params in PRESETS.items():
        result = apply_preset(thumb_src, params)
        thumbnails[name] = img_to_data_uri(result, quality=70)

    return jsonify(main=main_uri, thumbnails=thumbnails)


@app.route('/apply', methods=['POST'])
def apply():
    data = request.json
    sid = data.get('session')
    preset_name = data.get('preset', 'Original')
    if sid not in sessions:
        return jsonify(error='Invalid session'), 400

    rotated = get_rotated(sessions[sid])
    if preset_name in PRESETS:
        result = apply_preset(rotated, PRESETS[preset_name])
    else:
        result = rotated

    main_uri = img_to_data_uri(result, quality=92)
    return jsonify(main=main_uri)


@app.route('/download', methods=['POST'])
def download():
    data = request.json
    sid = data.get('session')
    preset_name = data.get('preset', 'Original')
    if sid not in sessions:
        return jsonify(error='Invalid session'), 400

    # Load full-size original for download
    full_img = load_image(sessions[sid]['path'], max_size=99999)
    session_copy = {'original': full_img, 'rotation': sessions[sid]['rotation']}
    rotated = get_rotated(session_copy)

    if preset_name in PRESETS:
        result = apply_preset(rotated, PRESETS[preset_name])
    else:
        result = rotated

    uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(uint8)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=95)
    buf.seek(0)

    filename = f"edited_{preset_name.replace(' ', '_')}.jpg"
    return send_file(buf, mimetype='image/jpeg', as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
