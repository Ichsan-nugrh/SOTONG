import streamlit as st
import os
import uuid
import matplotlib.pyplot as plt
from openai import OpenAI

# Import semua fungsi dari file utils.py
import utils

# 1. KONFIGURASI HALAMAN & CSS
st.set_page_config(page_title="SOTONG AI", layout="centered")
st.markdown("""
    <style>
    .block-container { max-width: 800px; padding-top: 2rem; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer { visibility: hidden; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SOTONG AI")
st.subheader("Sistem Observasi Tumor Otak dan Neural Geometri")
st.info("ℹ️ DISCLAIMER : SOTONG adalah proyek akademis, bukan alat diagnosis resmi")

# 2. INISIALISASI DATABASE & API
utils.init_db()

PRIMARY_MODEL = "openai/gpt-oss-120b" 
VISION_MODEL = "qwen/qwen3.8-27b" 

@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY tidak ditemukan!")
        st.stop()
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

client = get_groq_client()
cls_models, model_seg, class_names, device = utils.load_models()

# 3. SIDEBAR
with st.sidebar:
    st.header("⚙️ Menu SOTONG")
    uploaded_file = st.file_uploader("Unggah Citra MRI", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        analyze_btn = st.button("🔍 Analisis Gambar", use_container_width=True)
        
    st.divider()
    if st.button("🗑️ Mulai Obrolan Baru", use_container_width=True):
        if "session_id" in st.query_params:
            utils.clear_messages(st.query_params["session_id"])
        st.query_params.clear()
        st.session_state.clear()
        st.rerun()

# 4. MANAJEMEN SESI CHAT
if "session_id" not in st.query_params:
    st.query_params["session_id"] = str(uuid.uuid4())
current_session_id = st.query_params["session_id"]

if "messages" not in st.session_state:
    db_messages = utils.load_messages(current_session_id)
    if not db_messages:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya SOTONG. Silakan unggah gambar MRI melalui menu di kiri, lalu klik 'Analisis Gambar'."}]
    else:
        st.session_state.messages = db_messages

# 5. LOGIKA ANALISIS GAMBAR
if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
    with st.chat_message("user"):
        st.write("Tolong analisis gambar MRI ini.")
        st.image(uploaded_file, width=200)
    
    st.session_state.messages.append({"role": "user", "content": "Tolong analisis gambar MRI ini."})
    utils.save_message(current_session_id, "user", "Tolong analisis gambar MRI ini.")

    with st.chat_message("assistant"):
        with st.spinner("Membaca gambar dan menganalisis jaringan otak..."):
            try:
                is_valid = utils.validate_image_with_ai(client, VISION_MODEL, uploaded_file)
                
                if not is_valid:
                    st.error("❌ Maaf, ini sepertinya bukan gambar MRI otak.")
                    st.session_state.messages.append({"role": "assistant", "content": "Maaf, gambar ini bukan MRI otak."})
                    utils.save_message(current_session_id, "assistant", "Maaf, gambar ini bukan MRI otak.")
                else:
                    label, conf, img_orig, img_overlay = utils.predict_image(uploaded_file, cls_models, model_seg, class_names, device)
                    
                    if label == "Tidak Diketahui":
                        st.error(f"❌ Sistem tidak yakin ini MRI (Keyakinan: {conf*100:.1f}%).")
                        msg = f"Sistem tidak yakin ini MRI (Keyakinan: {conf*100:.1f}%)."
                    else:
                        st.success(f"**Hasil Klasifikasi:** {label} ({conf*100:.1f}%)")
                        st.session_state['ml_images'] = {"original": img_orig, "overlay": img_overlay}
                        
                        msg = f"Berdasarkan analisis sistem, terdeteksi **{label}** dengan keyakinan {conf*100:.1f}%. Area tumor telah disegmentasi (warna merah)."
                    
                    st.session_state['ml_result'] = {"label": label, "confidence": conf}
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    utils.save_message(current_session_id, "assistant", msg)
                    st.session_state['trigger_auto_explain'] = True 
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal memproses gambar: {e}")

# 6. RENDER CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Cek jika pesan ini adalah hasil analisis, maka tampilkan gambarnya
        if message["role"] == "assistant" and "Berdasarkan analisis sistem" in message["content"]:
            if st.session_state.get('ml_images'):
                imgs = st.session_state['ml_images']
                fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                axes[0].imshow(imgs['original'], cmap='gray'); axes[0].set_title('MRI Original'); axes[0].axis('off')
                axes[1].imshow(imgs['overlay']); axes[1].set_title('Segmentasi Tumor'); axes[1].axis('off')
                st.pyplot(fig)

# 7. AUTO-EXPLAIN
if st.session_state.get('trigger_auto_explain', False):
    st.session_state['trigger_auto_explain'] = False 
    
    auto_prompt = "Tolong jelaskan secara detail hasil klasifikasi dan segmentasi tersebut, serta apa artinya bagi saya."
    st.session_state.messages.append({"role": "user", "content": auto_prompt})
    utils.save_message(current_session_id, "user", auto_prompt)
    
    with st.chat_message("user"):
        st.markdown(auto_prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Menyiapkan penjelasan..."):
            ml_result = st.session_state.get('ml_result', None)
            hasil_analisis = f"Hasil klasifikasi: {ml_result['label']} ({ml_result['confidence']*100:.1f}%)"
            system_context = utils.load_prompt("chat_system.txt", hasil_analisis=hasil_analisis)
            
            try:
                bot_reply = st.write_stream(utils.get_chat_response(client, PRIMARY_MODEL, system_context, st.session_state.messages[-10:]))
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                utils.save_message(current_session_id, "assistant", bot_reply)
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")

# 8. INPUT CHAT MANUAL
if prompt := st.chat_input("Tanya apa saja tentang hasil MRI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    utils.save_message(current_session_id, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("SOTONG sedang berpikir..."):
            ml_result = st.session_state.get('ml_result', None)
            hasil_analisis = f"Hasil klasifikasi: {ml_result['label']} ({ml_result['confidence']*100:.1f}%)" if ml_result else "Pengguna belum melakukan analisis gambar."
            system_context = utils.load_prompt("chat_system.txt", hasil_analisis=hasil_analisis)

            try:
                bot_reply = st.write_stream(utils.get_chat_response(client, PRIMARY_MODEL, system_context, st.session_state.messages[-10:]))
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                utils.save_message(current_session_id, "assistant", bot_reply)
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")