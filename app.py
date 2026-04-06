import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import folium
from streamlit_folium import st_folium
from model_utils import RoadDamageCNN

# ── PAGE CONFIG (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RoadScan — AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── INJECT CSS ────────────────────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLASS_NAMES      = ['Good', 'Poor', 'Satisfactory', 'Very Poor']
SEVERITY_WEIGHTS = torch.tensor([0.0, 2.0, 1.0, 3.0])


SEVERITY_CONFIG = {
    'Good': {
        'color':  '#059669',
        'glow':   '#05966918',
        'bg':     '#f0fdf4',
        'border': '#6ee7b7',
        'icon':   '✓',
        'desc':   'No maintenance required',
    },
    'Satisfactory': {
        'color':  '#d97706',
        'glow':   '#d9770618',
        'bg':     '#fffbeb',
        'border': '#fcd34d',
        'icon':   '◑',
        'desc':   'Preventive care recommended',
    },
    'Poor': {
        'color':  '#ea580c',
        'glow':   '#ea580c18',
        'bg':     '#fff7ed',
        'border': '#fdba74',
        'icon':   '⚠',
        'desc':   'Repair within 2 weeks',
    },
    'Very Poor': {
        'color':  '#dc2626',
        'glow':   '#dc262618',
        'bg':     '#fef2f2',
        'border': '#fca5a5',
        'icon':   '✕',
        'desc':   'Immediate action required',
    },
}

ALERT_TEXT = {
    'Good':
        'Road surface is in good condition. No immediate action required. '
        'Schedule routine inspection in 6 months.',
    'Satisfactory':
        'Minor surface degradation detected. Consider preventive maintenance '
        'within 3 months to avoid further deterioration.',
    'Poor':
        'Significant damage detected. Cracks and surface deformation are '
        'present. Schedule repair within 2 weeks.',
    'Very Poor':
        'CRITICAL: Severe structural damage — potholes and deep deformation '
        'visible. Immediate repair required. High safety risk to road users.',
}

PROB_COLORS = {
    'Good':         '#22c55e',
    'Satisfactory': '#f59e0b',
    'Poor':         '#f97316',
    'Very Poor':    '#ef4444',
}

PRIORITY_COLOR = {
    'LOW':      '#22c55e',
    'MEDIUM':   '#f59e0b',
    'HIGH':     '#f97316',
    'CRITICAL': '#ef4444',
}

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    mdl  = RoadDamageCNN(num_classes=4)
    ckpt = torch.load('road_damage_test2_model.pth', map_location='cpu')
    mdl.load_state_dict(ckpt['model_state'])
    mdl.eval()
    return mdl

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_priority(si, conf):
    score = si * conf
    if score < 0.5:  return 'LOW'
    if score < 1.0:  return 'MEDIUM'
    if score < 2.0:  return 'HIGH'
    return 'CRITICAL'

def generate_gradcam(mdl, tensor, cls_idx):
    grads, acts = [], []

    def fhook(m, i, o): acts.append(o.detach())
    def bhook(m, gi, go): grads.append(go[0].detach())

    layer = mdl.features[-3]
    h1 = layer.register_forward_hook(fhook)
    h2 = layer.register_full_backward_hook(bhook)
    mdl.zero_grad()
    out = mdl(tensor)
    out[0, cls_idx].backward()
    h1.remove(); h2.remove()

    w   = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = (w * acts[0]).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam.numpy()

def make_overlay(pil_img, cam, alpha=0.45):
    img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.
    cam_r  = cv2.resize(cam, (224, 224))
    heat   = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heat   = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    blend  = alpha * heat + (1 - alpha) * img_np
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-logo">🛣️</div>
    <div>
      <div class="topbar-name">Road<em>Scan</em></div>
      <div class="topbar-sub">
        CNN-Based Road Surface Damage Detection &amp; Severity Classification
      </div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="topbar-chip">Project &nbsp;·&nbsp; Machine Vision</div>
    <div class="topbar-chip">
      <span class="live-dot"></span>MODEL READY
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD (top level so inference vars are available everywhere) ─────────
uploaded = st.file_uploader(
    "Upload a road surface image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ── RUN INFERENCE IMMEDIATELY so stats bar can use results ────────────────────
img       = None
pred_name = None
conf      = None
si        = None
priority  = None
cfg       = None
probs     = None
pred_idx  = None

if uploaded:
    img = Image.open(uploaded).convert('RGB')

    tensor = preprocess(img).unsqueeze(0)
    tensor.requires_grad_(True)

    with st.spinner("Running CNN inference..."):
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze(0)

    pred_idx  = torch.argmax(probs).item()
    pred_name = CLASS_NAMES[pred_idx]
    conf      = probs[pred_idx].item()
    si        = (probs * SEVERITY_WEIGHTS).sum().item()
    priority  = get_priority(si, conf)
    cfg       = SEVERITY_CONFIG[pred_name]

# ── STATS BAR (only when results exist) ──────────────────────────────────────
if pred_name:
    p_color = cfg['color']
    st.markdown(f"""
    <div class="stats-bar">
      <div class="stat-cell">
        <div class="stat-label">Detected Class</div>
        <div class="stat-value" style="color:{p_color}">{pred_name}</div>
        <div class="stat-sub">argmax(pᵢ)</div>
      </div>
      <div class="stat-cell">
        <div class="stat-label">Confidence</div>
        <div class="stat-value" style="color:{p_color}">{conf*100:.1f}%</div>
        <div class="stat-sub">C = max(pᵢ)</div>
      </div>
      <div class="stat-cell">
        <div class="stat-label">Severity Index</div>
        <div class="stat-value" style="color:{p_color}">{si:.2f}</div>
        <div class="stat-sub">SI = Σ wᵢ·pᵢ</div>
      </div>
      <div class="stat-cell">
        <div class="stat-label">Priority</div>
        <div class="stat-value" style="color:{p_color}">{priority}</div>
        <div class="stat-sub">SI × C threshold</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1.05, 1], gap="large")

# ═══════════════════════════════════════════════════
# LEFT COLUMN
# ═══════════════════════════════════════════════════
with col1:
    st.markdown("""
    <div class="col-header">
      <div class="col-label">Severity Classification</div>
      <div class="col-title">Image Input &amp; Severity Classification</div>
    </div>
    """, unsafe_allow_html=True)

    if img:
        # Image
        st.markdown('<div style="padding:0 28px">', unsafe_allow_html=True)
        st.image(img,
                 caption="Original Image · Road Image Acquisition",
                 use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline tags
        st.markdown("""
        <div class="pipe-row">
        <div class="pipe-tag">① Resize 224×224</div>
        <div class="pipe-tag">② ToTensor</div>
        <div class="pipe-tag">③ ImageNet Normalize</div>
        </div>
        """, unsafe_allow_html=True)


        # Severity banner
        st.markdown(f"""
        <div class="sev-banner"
             style="background:{cfg['bg']};
                    border-color:{cfg['border']};
                    --glow:{cfg['glow']}">
          <div class="sev-icon"
               style="background:{cfg['glow']};
                      color:{cfg['color']};
                      border-color:{cfg['border']}">
            {cfg['icon']}
          </div>
          <div>
            <div class="sev-class" style="color:{cfg['color']}">{pred_name}</div>
            <div class="sev-sub">
              {cfg['desc']} &nbsp;·&nbsp;
              Priority: <strong style="color:{cfg['color']}">{priority}</strong>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric tiles
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-tile">
            <div class="metric-tile-label">Confidence</div>
            <div class="metric-tile-val" style="color:{cfg['color']}">{conf*100:.1f}%</div>
            <div class="metric-tile-sub">C = max(pᵢ)</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label">Severity Index</div>
            <div class="metric-tile-val" style="color:{cfg['color']}">{si:.2f}</div>
            <div class="metric-tile-sub">SI = Σ wᵢ·pᵢ</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label">Priority</div>
            <div class="metric-tile-val"
                 style="color:{cfg['color']};font-size:18px;padding-top:5px">
              {priority}
            </div>
            <div class="metric-tile-sub">SI × C threshold</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars — native Streamlit to avoid HTML rendering bug
        st.markdown("""
        <div class="prob-section">
          <div class="prob-header">Class Probability Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            for i, name in enumerate(CLASS_NAMES):
                p      = probs[i].item()
                is_top = (i == pred_idx)
                color  = PROB_COLORS[name]

                label_col, bar_col, pct_col = st.columns([1.8, 5, 1])

                with label_col:
                    st.markdown(
                        f'<div style="font-family:Space Mono,monospace;'
                        f'font-size:11px;padding-top:6px;'
                        f'color:{"#e5e5e5" if is_top else "#404040"};'
                        f'font-weight:{"700" if is_top else "400"}">'
                        f'{name}</div>',
                        unsafe_allow_html=True
                    )
                with bar_col:
                    st.progress(float(p))
                with pct_col:
                    st.markdown(
                        f'<div style="font-family:Space Mono,monospace;'
                        f'font-size:11px;padding-top:6px;text-align:right;'
                        f'color:{"#e5e5e5" if is_top else "#404040"}">'
                        f'{p*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )

        # Alert
        st.markdown(f"""
        <div class="alert-wrap" style="border-left-color:{cfg['color']}">
          <div class="alert-head" style="color:{cfg['color']}">
            ▸ Maintenance Alert [{priority}]
          </div>
          <div class="alert-body">{ALERT_TEXT[pred_name]}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🛣️</div>
          <div class="empty-title">No image uploaded yet</div>
          <div class="empty-sub">Upload a road photo above to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# RIGHT COLUMN
# ═══════════════════════════════════════════════════
with col2:
    st.markdown("""
    <div class="col-header">
      <div class="col-label">Grad CAM & Location</div>
      <div class="col-title">Grad-CAM Explanation &amp; Location Alert</div>
    </div>
    """, unsafe_allow_html=True)

    if img:

        # ── Grad-CAM ──────────────────────────────────
        st.markdown('<div class="gcam-wrap">', unsafe_allow_html=True)
        try:
            cam_t = preprocess(img).unsqueeze(0)
            cam_t.requires_grad_(True)
            cam     = generate_gradcam(model, cam_t, pred_idx)
            overlay = make_overlay(img, cam)

            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(img.resize((224, 224)),
                         caption="Input",
                         use_container_width=True)
            with gc2:
                st.image(overlay,
                         caption="Grad-CAM Overlay",
                         use_container_width=True)

            st.markdown("""
            <div class="gcam-note">
              Red = highest CNN attention &nbsp;·&nbsp;
              Blue = low attention &nbsp;·&nbsp;
              Feature map
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

        # ── Map ───────────────────────────────────────
        

        def get_gps_from_image(pil_img):
            try:
                exif_data = pil_img._getexif()
                if not exif_data:
                    return None
                from PIL.ExifTags import TAGS, GPSTAGS
                exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
                gps_info = exif.get('GPSInfo')
                if not gps_info:
                    return None
                gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
                def to_deg(vals):
                    d, m, s = vals
                    return float(d) + float(m)/60 + float(s)/3600
                lat = to_deg(gps['GPSLatitude'])
                lon = to_deg(gps['GPSLongitude'])
                if gps.get('GPSLatitudeRef')  == 'S': lat = -lat
                if gps.get('GPSLongitudeRef') == 'W': lon = -lon
                return [lat, lon]
            except Exception:
                return None
            
        coords   = get_gps_from_image(img) or [12.9716, 77.5946]
        gps_note = "Live GPS from image EXIF" if get_gps_from_image(img) else "Simulated location (no EXIF data found)"

        m = folium.Map(
            location=coords,
            zoom_start=15,
            tiles='CartoDB positron'
        )
        st.markdown(f"""
        <div class="map-wrap">
          <div class="col-label">Location-Based Alert</div>
          <div class="col-title" style="margin-bottom:5px">
            GPS-Tagged Damage Report
          </div>
          <div class="map-meta">
            {gps_note}. Severity and priority are embedded in the marker.
            Simulates a dashcam logging GPS coordinates at the point of
            damage detection. Severity and priority are embedded in the marker.
          </div>
        </div>
        """, unsafe_allow_html=True)

        folium.CircleMarker(
            location=[coords[0], coords[1]],
            radius=14,
            color=cfg['color'],
            fill=True,
            fill_color=cfg['color'],
            fill_opacity=0.88,
            popup=folium.Popup(
                f"<b style='font-family:monospace'>{pred_name}</b><br>"
                f"SI: {si:.2f} &nbsp;·&nbsp; {priority}<br>"
                f"Confidence: {conf*100:.1f}%",
                max_width=180
            )
        ).add_to(m)

        folium.Marker(
            [coords[0], coords[1]],
            icon=folium.DivIcon(html=f"""
            <div style="font-family:monospace;font-size:10px;
                        background:{cfg['color']};color:#fff;
                        padding:4px 10px;border-radius:5px;
                        white-space:nowrap;font-weight:700;
                        box-shadow:0 2px 10px {cfg['glow']}">
              {pred_name.upper()} · {priority}
            </div>
            """)
        ).add_to(m)

        st.markdown('<div style="padding:0 28px">', unsafe_allow_html=True)
        st_folium(m, height=250, use_container_width=True)

        st.markdown(f"""
        <div class="callout">
          <div class="callout-icon">💡</div>
          <div class="callout-text">
            In production, GPS coordinates come from EXIF metadata or a
            phone sensor. SI = <strong>{si:.2f}</strong> with priority
            <strong>{priority}</strong> would trigger an automatic
            maintenance ticket for road authorities.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📍</div>
          <div class="empty-title">Awaiting image</div>
          <div class="empty-sub">
            Grad-CAM and map appear after analysis
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  <div class="footer-left">
    RoadScan &nbsp;·&nbsp; Road Damage Severity Detection &nbsp;·&nbsp;
    2025–2026
  </div>
  <div class="footer-tags">
    <span class="footer-tag">PyTorch</span>
    <span class="footer-tag">CNN from scratch</span>
    <span class="footer-tag">Grad-CAM</span>
    <span class="footer-tag">Streamlit</span>
    <span class="footer-tag">Folium</span>
  </div>
</div>
""", unsafe_allow_html=True)