import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import plotly.graph_objects as go
import numpy as np

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Floutage Vidéo Intelligent", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #e0f0ff, #a8d5ff); }
h1, h2, h3, h4 { color: #1b3a6b; }
[data-testid="stSidebar"] { background: linear-gradient(135deg, #cde0ff, #90bfff) !important; border-radius: 15px; padding: 15px; }
button[kind="primary"] { background: linear-gradient(135deg, #80bfff, #3399ff) !important; color: white !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# TITRE
# =========================
st.title("Floutage automatique de vidéo")
st.write("Floutez vos **visages**, **plaques** et **écrans** facilement")

# =========================
# CHARGEMENT MODÈLES
# =========================
@st.cache_resource
def charger_modeles():
    return {
        "face": YOLO("yolov8s-face-lindevs.pt"),
        "alpr": YOLO("best.pt"),
        "coco": YOLO("yolov8n.pt")
    }

models = charger_modeles()
OBJETS_COCO = ["laptop", "cell phone", "tv"]

# =========================
# OPTIONS
# =========================
st.sidebar.title("⚙️ Options")
use_face = st.sidebar.checkbox("Flouter les visages", True)
use_alpr = st.sidebar.checkbox("Flouter les plaques", True)
use_coco = st.sidebar.checkbox("Flouter les écrans", True)
intensite_flou = st.sidebar.slider("Intensité du flou", 1, 5, 3)

# =========================
# UPLOAD
# =========================
video_file = st.file_uploader("📤 Importer une vidéo", type=["mp4", "avi", "mov"])

# =========================
# FONCTIONS
# =========================
def clamp_bbox(bbox, shape):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(shape[1]-1, x2); y2 = min(shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def flouter_roi(frame, bbox, intensite, ellipse=False):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = max(15, ((x2-x1)//3)|1) * intensite
    flou = cv2.GaussianBlur(roi, (k, k), 0)
    if ellipse:
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cx, cy = (x2-x1)//2, (y2-y1)//2
        rx = int((x2-x1) * 0.95 / 2)  # largeur légèrement plus grande
        ry = int((y2-y1) * 0.95 / 2)  # hauteur légèrement plus grande
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        roi[mask == 255] = flou[mask == 255]
    else:
        roi[:] = flou

def cercle_progression(p):
    values = [p, 100-p]
    fig = go.Figure(go.Pie(
        values=values,
        hole=0.7,
        textinfo="none",
        marker_colors=["#2f87df", "#a8c4ed"],
        sort=False,
        direction="clockwise",
        rotation=90
    ))
    fig.update_layout(
        width=200, height=200,
        annotations=[dict(text=f"{p}%", x=0.5, y=0.5, showarrow=False, font_size=22)],
        margin=dict(t=0,b=0,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# =========================
# TRAITEMENT
# =========================
FRAME_STEP = 1
MAX_MISSED = 1

if video_file:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vidéo importée")
        st.video(video_file)  # affichage grand format importé

    if st.button("Lancer le floutage"):
        with st.spinner("Traitement en cours..."):
            temp_in = tempfile.NamedTemporaryFile(delete=False)
            temp_in.write(video_file.read())
            temp_in.close()

            cap = cv2.VideoCapture(temp_in.name)
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            trackers = {"face": [], "alpr": [], "coco": []}
            last_bboxes = {"face": [], "alpr": [], "coco": []}
            missed = {"face": [], "alpr": [], "coco": []}
            frame_id = 0
            progress = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                # 🔵 Détection YOLO
                if frame_id % FRAME_STEP == 0:
                    small = cv2.resize(frame, (640, int(640*h/w)))
                    for key, enabled, labels in [
                        ("face", use_face, None),
                        ("alpr", use_alpr, None),
                        ("coco", use_coco, OBJETS_COCO)
                    ]:
                        if not enabled:
                            continue
                        result = models[key](small, conf=0.4, verbose=False)[0]
                        trackers[key] = []
                        last_bboxes[key] = []
                        missed[key] = []
                        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                            if labels and models[key].names[int(cls)] not in labels:
                                continue
                            bbox = [
                                box[0]*w/640,
                                box[1]*h/small.shape[0],
                                box[2]*w/640,
                                box[3]*h/small.shape[0]
                            ]
                            safe = clamp_bbox(bbox, frame.shape)
                            if safe:
                                flouter_roi(frame, safe, intensite_flou, ellipse=(key=="face"))
                                tracker = cv2.legacy.TrackerKCF_create()
                                x1,y1,x2,y2 = safe
                                tracker.init(frame, (x1,y1,x2-x1,y2-y1))
                                trackers[key].append(tracker)
                                last_bboxes[key].append(safe)
                                missed[key].append(0)

                # 🔵 Tracking avec persistance
                else:
                    for key in trackers:
                        new_trackers = []
                        new_bboxes = []
                        new_missed = []
                        for i, tr in enumerate(trackers[key]):
                            ok, b = tr.update(frame)
                            if ok:
                                x, y, w0, h0 = map(int, b)
                                safe = clamp_bbox((x, y, x+w0, y+h0), frame.shape)
                                if safe:
                                    flouter_roi(frame, safe, intensite_flou, ellipse=(key=="face"))
                                    new_trackers.append(tr)
                                    new_bboxes.append(safe)
                                    new_missed.append(0)
                            else:
                                if i < len(last_bboxes[key]) and missed[key][i] < MAX_MISSED:
                                    safe = last_bboxes[key][i]
                                    flouter_roi(frame, safe, intensite_flou, ellipse=(key=="face"))
                                    new_trackers.append(tr)
                                    new_bboxes.append(safe)
                                    new_missed.append(missed[key][i]+1)
                        trackers[key] = new_trackers
                        last_bboxes[key] = new_bboxes
                        missed[key] = new_missed

                out.write(frame)
                if frame_id % 10 == 0:
                    p = round((frame_id / total) * 100)
                    progress.plotly_chart(cercle_progression(p), use_container_width=False)

            cap.release()
            out.release()

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.success("✅ Traitement terminé")

            st.download_button(
                "⬇️ Télécharger la vidéo floutée",
                video_bytes,
                "video_floutee.mp4"
            )
