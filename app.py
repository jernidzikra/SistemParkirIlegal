import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Default: webcam (index 0)
current_video_path = 0

# 1. KONFIGURASI FOLDER UPLOAD VIDEO
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 2. LOAD MODEL AI HASIL TRAINING (YOLOv8)
model = YOLO('best.pt')

# Tracker Waktu menggunakan posisi koordinat kotak (bukan ID lagi agar anti-gagal)
tracker_kendaraan = {}
BATAS_WAKTU_DETIK = 5

# Warna (BGR)
MERAH = (0, 0, 255)
HIJAU = (0, 255, 0)


def buka_sumber(path):
    """Buka webcam (int) atau file video (str)."""
    if isinstance(path, int):
        # CAP_DSHOW lebih stabil untuk webcam di Windows
        return cv2.VideoCapture(path, cv2.CAP_DSHOW)
    return cv2.VideoCapture(path)


def frame_pesan(teks):
    """Frame hitam berisi pesan, supaya error terlihat di layar, bukan layar hitam kosong."""
    img = np.zeros((480, 854, 3), np.uint8)
    cv2.putText(img, teks, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    ret, buf = cv2.imencode('.jpg', img)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


def generate_frames():
    global current_video_path, tracker_kendaraan

    sumber = current_video_path
    camera = buka_sumber(sumber)

    if not camera.isOpened():
        yield frame_pesan("Gagal membuka sumber video / webcam")
        camera.release()
        return

    frame_count = 0
    frame_berhasil = 0

    while True:
        success, frame = camera.read()
        if not success:
            # File video: ulang dari awal, tapi hanya jika sebelumnya pernah berhasil baca frame.
            # Kalau belum pernah (codec tidak didukung), berhenti agar tidak loop tanpa akhir.
            if isinstance(sumber, str) and frame_berhasil > 0:
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker_kendaraan.clear()
                continue
            if isinstance(sumber, str):
                yield frame_pesan("Video tidak bisa dibaca (coba format .mp4 H.264)")
            break

        frame_berhasil += 1
        frame_count += 1
        # OPTIMASI ANTI-LAG: Proses setiap 2 frame sekali
        if frame_count % 2 != 0:
            continue

        h_img, w_img, _ = frame.shape

        # ====================================================================
        # ROI: seluruh sisi kiri jalan, dari ATAS sampai BAWAH
        # Naikkan/turunkan angka 0.33 & 0.40 untuk melebarkan/menyempitkan zona
        # ====================================================================
        roi_polygon = np.array([
            [0, 0],
            [int(w_img * 0.33), 0],
            [int(w_img * 0.40), h_img],
            [0, h_img]
        ], np.int32)

        # Gambar zona merah transparan
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], MERAH)
        frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
        cv2.polylines(frame, [roi_polygon], True, MERAH, 1)

        # DETEKSI STANDAR
        results = model(frame, verbose=False, classes=[2, 4])

        waktu_sekarang = time.time()
        id_terdeteksi_sekarang = []

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for index, (box, cls) in enumerate(zip(boxes, clss)):
                    x1, y1, x2, y2 = map(int, box)
                    label_ai = model.names[cls]

                    # Posisi koordinat konstan sebagai pengganti ID virtual
                    key_id = f"{label_ai}_{int(x1/10)}_{int(y1/10)}"
                    id_terdeteksi_sekarang.append(key_id)

                    # Titik acuan: roda menyentuh aspal
                    titik_roda = (int((x1 + x2) / 2), y2)

                    # Cek zona kiri jalan
                    is_violating = cv2.pointPolygonTest(roi_polygon, titik_roda, False)

                    # Posisi label, dijaga agar tidak keluar frame atas
                    ty = max(y1 - 10, 15)

                    if is_violating >= 0:
                        if key_id not in tracker_kendaraan:
                            tracker_kendaraan[key_id] = waktu_sekarang

                        durasi_diam = waktu_sekarang - tracker_kendaraan[key_id]
                        waktu_tersisa = max(0, int(BATAS_WAKTU_DETIK - durasi_diam))

                        if durasi_diam >= BATAS_WAKTU_DETIK:
                            # PELANGGARAN -> Merah tebal
                            cv2.rectangle(frame, (x1, y1), (x2, y2), MERAH, 3)
                            cv2.putText(frame, f"ALERT: Illegal Parking ({label_ai})", (x1, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MERAH, 2)
                        else:
                            # WARNING -> Merah (lebih tipis)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), MERAH, 2)
                            cv2.putText(frame, f"Warning: {waktu_tersisa}s to Illegal ({label_ai})", (x1, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, MERAH, 2)
                    else:
                        if key_id in tracker_kendaraan:
                            tracker_kendaraan.pop(key_id)

                        # NORMAL -> Hijau
                        cv2.rectangle(frame, (x1, y1), (x2, y2), HIJAU, 2)
                        cv2.putText(frame, f"Normal: {label_ai}", (x1, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIJAU, 2)

        # Bersihkan memori tracker
        for id_lama in list(tracker_kendaraan.keys()):
            if id_lama not in id_terdeteksi_sekarang:
                tracker_kendaraan.pop(id_lama)

        # Encode ke biner
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


@app.route('/')
def index():
    mode_video = "VIDEO FILE UPLOADED" if isinstance(current_video_path, str) else "LIVE WEBCAM MONITORING"
    return render_template('index.html', mode=mode_video)


@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video_path, tracker_kendaraan

    file = request.files.get('file')
    if file is None or file.filename == '':
        return redirect(url_for('index'))

    nama = secure_filename(file.filename)
    ext = os.path.splitext(nama)[1].lower()
    if not nama or ext not in ALLOWED_EXT:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], nama)
    file.save(filepath)
    tracker_kendaraan.clear()
    current_video_path = filepath
    return redirect(url_for('index'))


@app.route('/reset')
def reset_camera():
    global current_video_path, tracker_kendaraan
    tracker_kendaraan.clear()
    current_video_path = 0
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # use_reloader=False: mencegah Flask restart otomatis (dan memuat ulang model) saat file tersimpan
    app.run(debug=True, port=5000, use_reloader=False)