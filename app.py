import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------
st.set_page_config(page_title="X-Ray Enhancer", page_icon="🏥", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(34,211,238,0.10), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(139,92,246,0.12), transparent 45%),
        #0B1220;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.85);
    border-right: 1px solid rgba(148,163,184,0.15);
}

.hero {
    padding: 28px 32px; border-radius: 20px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(139,92,246,0.14));
    border: 1px solid rgba(148,163,184,0.18);
}
.hero h1 {
    margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #22D3EE, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { margin: 8px 0 0 0; color: #94A3B8; font-size: 1rem; }
.badge {
    display: inline-block; margin-top: 14px; margin-right: 8px;
    padding: 4px 12px; border-radius: 999px; font-size: 0.78rem;
    color: #67E8F9; background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.30);
}

.card {
    background: rgba(30,41,59,0.55); border: 1px solid rgba(148,163,184,0.18);
    border-radius: 16px; padding: 18px 20px; height: 100%;
}
.card h4 { margin: 0 0 6px 0; color: #E2E8F0; font-size: 1.05rem; }
.card p  { margin: 0; color: #94A3B8; font-size: 0.88rem; line-height: 1.5; }
.card code { color: #67E8F9; background: rgba(34,211,238,0.10); padding: 2px 6px; border-radius: 6px; }

.flow { display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 10px; margin: 8px 0 18px 0; }
.step {
    padding: 10px 18px; border-radius: 12px; font-weight: 600; font-size: 0.9rem;
    color: #E2E8F0; background: rgba(34,211,238,0.10); border: 1px solid rgba(34,211,238,0.30);
}
.arrow { color: #A78BFA; font-size: 1.3rem; }

.img-label {
    font-weight: 600; font-size: 0.9rem; color: #E2E8F0;
    margin-bottom: 8px; padding-left: 8px; border-left: 3px solid #22D3EE;
}
div[data-testid="stImage"] img {
    border-radius: 14px; border: 1px solid rgba(148,163,184,0.20);
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
}
div[data-testid="stMetric"] {
    background: rgba(30,41,59,0.55); border: 1px solid rgba(148,163,184,0.18);
    border-radius: 16px; padding: 14px 18px;
}
div[data-testid="stMetricLabel"] { color: #94A3B8; }
button[data-baseweb="tab"] { font-weight: 600; }
div[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(34,211,238,0.35); border-radius: 16px; background: rgba(15,23,42,0.5);
}
.stDownloadButton button, .stButton button {
    border-radius: 12px; font-weight: 600; padding: 0.55rem 1.2rem;
}
.stDownloadButton button {
    background: linear-gradient(90deg, #06B6D4, #8B5CF6); color: white; border: none;
}
.stDownloadButton button:hover { filter: brightness(1.1); color: white; }
.footer-note { color: #64748B; font-size: 0.78rem; text-align: center; margin-top: 30px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------
def show_image(img, label):
    st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)
    try:
        st.image(img, width="stretch", clamp=True)
    except Exception:
        st.image(img, use_container_width=True, clamp=True)


def image_stats(img):
    """Kecerahan (mean), kontras (std), entropi."""
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return float(img.mean()), float(img.std()), entropy


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    """SSIM sederhana dengan jendela Gaussian (tanpa library tambahan)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    s11 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1 ** 2
    s22 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2 ** 2
    s12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def gamma_table(g):
    return np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255, 0, 255).astype(np.uint8)


def style_ax(ax, title, xlabel="Intensitas piksel", ylabel=""):
    ax.set_title(title, color="#E2E8F0", fontsize=12, fontweight="bold", pad=10)
    ax.set_facecolor("#0F172A")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.set_xlabel(xlabel, color="#94A3B8", fontsize=9)
    ax.set_ylabel(ylabel, color="#94A3B8", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color="#334155", alpha=0.35, linewidth=0.6)


def plot_hist(ax, img, color, title):
    h = np.bincount(img.ravel(), minlength=256)
    ax.fill_between(range(256), h, color=color, alpha=0.35)
    ax.plot(h, color=color, lw=1.3)
    ax.set_xlim(0, 255)
    style_ax(ax, title, ylabel="Jumlah piksel")


def cdf(img):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    c = np.cumsum(h)
    return c / c[-1]


def make_sample():
    """Buat gambar X-ray dada sintetis berkontras rendah untuk demo."""
    rng = np.random.default_rng(7)
    h, w = 512, 512
    img = np.full((h, w), 120, np.float32)
    cv2.ellipse(img, (w // 2, 270), (200, 235), 0, 0, 360, 150, -1)         # badan
    cv2.ellipse(img, (175, 250), (85, 150), 0, 0, 360, 85, -1)               # paru kiri
    cv2.ellipse(img, (337, 250), (85, 150), 0, 0, 360, 85, -1)               # paru kanan
    cv2.ellipse(img, (w // 2 + 25, 330), (55, 70), 0, 0, 360, 170, -1)       # jantung
    cv2.rectangle(img, (w // 2 - 14, 60), (w // 2 + 14, 480), 175, -1)       # tulang belakang
    for i in range(7):                                                       # tulang rusuk
        y = 130 + i * 40
        cv2.ellipse(img, (175, y), (105, 38), 15, 200, 340, 140, 3)
        cv2.ellipse(img, (337, y), (105, 38), -15, 200, 340, 140, 3)
    img = cv2.GaussianBlur(img, (0, 0), 5)
    img += rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255)
    img = 70 + (img - img.min()) / (img.max() - img.min()) * 70              # kontras rendah
    return img.astype(np.uint8)


# ---------------------------------------------------------------
# State & reset
# ---------------------------------------------------------------
DEFAULTS = {
    "method": "CLAHE (disarankan)",
    "clip": 2.0,
    "tile": 8,
    "gamma": 1.2,
    "invert": False,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("use_sample", False)


def reset_params():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>X-Ray Enhancer</h1>
        <p>Perbaikan kualitas citra sinar-X dada dengan Histogram Equalization, CLAHE, dan Gamma Correction.</p>
        <span class="badge">Histogram Equalization</span>
        <span class="badge">CLAHE</span>
        <span class="badge">Gamma Correction</span>
        <span class="badge">PSNR · SSIM</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Pengaturan")
method = st.sidebar.radio("Metode kontras", ["Histogram Equalization", "CLAHE (disarankan)"], key="method")

if method.startswith("CLAHE"):
    clip_limit = st.sidebar.slider("Clip limit", 1.0, 8.0, step=0.5, key="clip")
    tile = st.sidebar.select_slider("Ukuran tile CLAHE", options=[4, 8, 16], key="tile")
else:
    clip_limit, tile = 2.0, 8

gamma_val = st.sidebar.slider(
    "Nilai Gamma", 0.1, 3.0, step=0.1, key="gamma",
    help="Di atas 1 = lebih terang, di bawah 1 = lebih gelap.",
)
invert = st.sidebar.checkbox("Invert (negatif)", key="invert")
st.sidebar.button("↺ Reset pengaturan", on_click=reset_params, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Unggah gambar di halaman utama, lalu atur parameter di sini.")

# ---------------------------------------------------------------
# Sumber gambar
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Unggah foto X-Ray (JPG / PNG)", type=["jpg", "jpeg", "png"])

img_original = None
source_name = ""

if uploaded_file is not None:
    st.session_state["use_sample"] = False
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img_original = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    source_name = uploaded_file.name
    if img_original is None:
        st.error("File tidak dapat dibaca sebagai gambar. Coba file JPG/PNG lain.")
        st.stop()
elif st.session_state["use_sample"]:
    img_original = make_sample()
    source_name = "gambar_contoh_sintetis"
    st.caption("Menampilkan gambar contoh sintetis (bukan citra pasien).")
    if st.button("Tutup contoh"):
        st.session_state["use_sample"] = False
        st.rerun()

# ---------------------------------------------------------------
# Halaman awal (belum ada gambar)
# ---------------------------------------------------------------
if img_original is None:
    st.markdown("### Cara kerja")
    st.markdown(
        """
        <div class="flow">
            <div class="step">1 · Unggah citra</div><div class="arrow">➜</div>
            <div class="step">2 · Tingkatkan kontras (HE / CLAHE)</div><div class="arrow">➜</div>
            <div class="step">3 · Koreksi Gamma</div><div class="arrow">➜</div>
            <div class="step">4 · Bandingkan & unduh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        """<div class="card"><h4>Histogram Equalization</h4>
        <p>Meratakan sebaran intensitas ke seluruh rentang 0-255 sehingga area gelap dan terang lebih terpisah.
        Bekerja global pada seluruh gambar.</p></div>""",
        unsafe_allow_html=True,
    )
    c2.markdown(
        """<div class="card"><h4>CLAHE</h4>
        <p>Ekualisasi adaptif per kotak kecil dengan pembatas kontras (<code>clipLimit</code>).
        Lebih natural dan noise lebih terkendali untuk citra medis.</p></div>""",
        unsafe_allow_html=True,
    )
    c3.markdown(
        """<div class="card"><h4>Gamma Correction</h4>
        <p>Mengatur kecerahan non-linear: <code>out = 255 × (in/255)^(1/γ)</code>.
        γ &gt; 1 lebih terang, γ &lt; 1 lebih gelap.</p></div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1, 2])
    with left:
        if st.button("🧪 Pakai gambar contoh", use_container_width=True):
            st.session_state["use_sample"] = True
            st.rerun()
    with right:
        st.info("Belum punya foto? Klik tombol di kiri untuk mencoba dengan gambar contoh sintetis.")

    st.markdown(
        '<div class="footer-note">Hanya untuk keperluan edukasi, bukan alat diagnosis medis.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------
# Pemrosesan
# ---------------------------------------------------------------
table = gamma_table(gamma_val)

img_he_only = cv2.equalizeHist(img_original)
img_clahe_only = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(img_original)
img_gamma_only = cv2.LUT(img_original, table)

if method.startswith("CLAHE"):
    img_step2 = img_clahe_only
    step2_label = "2 · Setelah CLAHE"
else:
    img_step2 = img_he_only
    step2_label = "2 · Setelah Histogram Equalization"

img_final = cv2.LUT(img_step2, table)

# Metrik dihitung sebelum invert
m0, s0, e0 = image_stats(img_original)
m1, s1, e1 = image_stats(img_final)
psnr_val = psnr(img_original, img_final)
ssim_val = ssim(img_original, img_final)

if invert:
    img_final_disp = 255 - img_final
    img_step2_disp = 255 - img_step2
    img_orig_disp = 255 - img_original
else:
    img_final_disp, img_step2_disp, img_orig_disp = img_final, img_step2, img_original

# ---------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Kecerahan", f"{m1:.1f}", f"{m1 - m0:+.1f}")
k2.metric("Kontras (std)", f"{s1:.1f}", f"{s1 - s0:+.1f}")
k3.metric("Entropi", f"{e1:.2f}", f"{e1 - e0:+.2f}")
k4.metric("PSNR (dB)", f"{psnr_val:.1f}")
k5.metric("SSIM", f"{ssim_val:.3f}")
st.caption(
    "PSNR dan SSIM mengukur kemiripan hasil dengan citra asli. Nilai yang lebih rendah di sini wajar, "
    "karena tujuan peningkatan kontras memang mengubah citra."
)

st.write("")

# ---------------------------------------------------------------
# Tab
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🖼️ Perbandingan Citra", "↔️ Slider Sebelum/Sesudah", "🧪 Bandingkan Metode", "📊 Analisis"]
)

# --- Tab 1
with tab1:
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        show_image(img_orig_disp, "1 · Citra Asli")
    with c2:
        show_image(img_step2_disp, step2_label)
    with c3:
        show_image(img_final_disp, f"3 · Hasil Akhir (Gamma {gamma_val})")

    st.write("")
    ok, buf = cv2.imencode(".png", img_final_disp)

    report = (
        "parameter,nilai\n"
        f"sumber,{source_name}\n"
        f"metode,{method}\n"
        f"clip_limit,{clip_limit if method.startswith('CLAHE') else '-'}\n"
        f"tile,{tile if method.startswith('CLAHE') else '-'}\n"
        f"gamma,{gamma_val}\n"
        f"invert,{invert}\n"
        "\nmetrik,asli,hasil\n"
        f"kecerahan,{m0:.2f},{m1:.2f}\n"
        f"kontras_std,{s0:.2f},{s1:.2f}\n"
        f"entropi,{e0:.3f},{e1:.3f}\n"
        f"psnr_db,-,{psnr_val:.2f}\n"
        f"ssim,-,{ssim_val:.4f}\n"
    )

    d1, d2, _ = st.columns([1, 1, 2])
    if ok:
        d1.download_button("⬇️ Unduh hasil (PNG)", data=buf.tobytes(),
                           file_name="xray_enhanced.png", mime="image/png")
    d2.download_button("📄 Unduh laporan (CSV)", data=report.encode("utf-8"),
                       file_name="laporan_xray.csv", mime="text/csv")

# --- Tab 2
with tab2:
    st.write("Geser untuk membandingkan: sisi kiri **asli**, sisi kanan **hasil**.")
    pos = st.slider("Posisi pembatas", 0, 100, 50, key="split")
    h, w = img_orig_disp.shape
    x = int(w * pos / 100)
    comp = img_orig_disp.copy()
    comp[:, x:] = img_final_disp[:, x:]
    if 0 < x < w:
        comp[:, max(x - 1, 0):min(x + 2, w)] = 255
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        show_image(comp, f"Asli (kiri)  |  Hasil (kanan) · pembatas {pos}%")

# --- Tab 3
with tab3:
    st.write("Perbandingan tiap metode secara terpisah pada citra yang sama.")
    variants = [
        ("Asli", img_original),
        ("Hanya HE", img_he_only),
        (f"Hanya CLAHE (clip {clip_limit})", img_clahe_only),
        (f"Hanya Gamma ({gamma_val})", img_gamma_only),
        ("Kombinasi pilihanmu", img_final),
    ]
    cols = st.columns(len(variants), gap="small")
    for col, (name, im) in zip(cols, variants):
        with col:
            show_image(255 - im if invert else im, name)
            mm, ss, ee = image_stats(im)
            st.caption(f"std {ss:.1f} · entropi {ee:.2f}")

# --- Tab 4
with tab4:
    st.markdown("#### Histogram")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.patch.set_facecolor("#0B1220")
    plot_hist(ax[0], img_original, "#94A3B8", "Histogram Citra Asli")
    plot_hist(ax[1], img_final, "#22D3EE", "Histogram Hasil Perbaikan")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Kurva Gamma dan Histogram Kumulatif (CDF)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.patch.set_facecolor("#0B1220")

    xs = np.arange(256)
    ax[0].plot(xs, xs, color="#64748B", lw=1.2, ls="--", label="γ = 1 (tanpa perubahan)")
    ax[0].plot(xs, table, color="#A78BFA", lw=2.2, label=f"γ = {gamma_val}")
    ax[0].set_xlim(0, 255)
    ax[0].set_ylim(0, 255)
    style_ax(ax[0], "Kurva Transfer Gamma", "Intensitas input", "Intensitas output")
    ax[0].legend(facecolor="#0F172A", edgecolor="none", labelcolor="#E2E8F0", fontsize=8)

    ax[1].plot(xs, cdf(img_original), color="#94A3B8", lw=2, label="Asli")
    ax[1].plot(xs, cdf(img_final), color="#22D3EE", lw=2, label="Hasil")
    ax[1].plot(xs, xs / 255.0, color="#64748B", lw=1, ls="--", label="Ideal (merata)")
    ax[1].set_xlim(0, 255)
    ax[1].set_ylim(0, 1.02)
    style_ax(ax[1], "CDF Intensitas", "Intensitas piksel", "Proporsi kumulatif")
    ax[1].legend(facecolor="#0F172A", edgecolor="none", labelcolor="#E2E8F0", fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Histogram hasil yang menyebar di rentang 0-255 dan CDF yang mendekati garis lurus "
        "menandakan kontras yang lebih baik."
    )

st.markdown(
    '<div class="footer-note">Hanya untuk keperluan edukasi, bukan alat diagnosis medis.</div>',
    unsafe_allow_html=True,
)