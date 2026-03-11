"""
CRAG Chatbot — Streamlit UI
Real-time mic input via MediaRecorder JS component → Whisper → RAG stream
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import re
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Get API URL from environment
# API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ─── Config ───────────────────────────────────────────────────────────────────

try:
    API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
except Exception:
    API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="CRAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background: #0f1117; }

    section[data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2a2f3e;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .msg-user {
        display:flex; justify-content:flex-end; margin:.75rem 0;
    }
    .msg-user .bubble {
        background:linear-gradient(135deg,#2563eb,#1d4ed8);
        color:#fff; padding:.75rem 1.1rem;
        border-radius:18px 18px 4px 18px;
        max-width:72%; font-size:.95rem; line-height:1.55;
        box-shadow:0 2px 8px rgba(37,99,235,.3);
        white-space:pre-wrap; word-wrap:break-word;
    }
    .msg-assistant {
        display:flex; justify-content:flex-start; margin:.75rem 0; align-items:flex-start;
    }
    .msg-assistant .avatar {
        width:32px; height:32px; border-radius:50%;
        background:linear-gradient(135deg,#7c3aed,#4f46e5);
        display:flex; align-items:center; justify-content:center;
        font-size:.85rem; flex-shrink:0; margin-right:.6rem; margin-top:2px;
    }
    .msg-assistant .bubble {
        background:#1e2330; color:#e2e8f0;
        padding:.75rem 1.1rem; border-radius:18px 18px 18px 4px;
        max-width:75%; font-size:.95rem; line-height:1.6;
        border:1px solid #2a3044; box-shadow:0 2px 8px rgba(0,0,0,.3);
        white-space:pre-wrap; word-wrap:break-word;
    }

    .sidebar-section {
        font-size:.72rem; font-weight:700; letter-spacing:.1em;
        text-transform:uppercase; color:#4b5563 !important;
        padding:1rem 0 .4rem;
    }
    .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:.72rem; font-weight:600; }
    .badge-green { background:#052e16; color:#4ade80; border:1px solid #166534; }
    .badge-blue  { background:#0c1a3a; color:#60a5fa; border:1px solid #1e3a8a; }
    .badge-red   { background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; }

    .welcome-hero { text-align:center; padding:3rem 0 1rem; }
    .welcome-hero h1 {
        font-size:2.2rem; font-weight:700;
        background:linear-gradient(135deg,#60a5fa,#a78bfa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .welcome-hero p { color:#6b7280; font-size:1rem; }
    .chip {
        display:inline-block; background:#1e2330; border:1px solid #2a3044;
        border-radius:8px; padding:.5rem .85rem; margin:.3rem;
        font-size:.85rem; color:#94a3b8;
    }
    .chip span { font-size:1.05rem; margin-right:4px; }

    .stTextInput > div > div > input {
        background:#1e2330 !important; border:1px solid #2a3044 !important;
        border-radius:12px !important; color:#e2e8f0 !important; padding:.7rem 1rem !important;
    }
    .stButton > button {
        border-radius:10px !important; border:1px solid #2a3044 !important;
        background:#1e2330 !important; color:#e2e8f0 !important; transition:all .2s !important;
    }
    .stButton > button:hover {
        background:#2563eb !important; border-color:#2563eb !important; color:white !important;
    }
    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:#2a3044; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────

DEFAULTS = {
    "thread_id": None,
    "thread_title": None,
    "messages": [],          # each message: {"role", "content", "image"(optional)}
    "threads_list": [],
    "api_base": API_BASE,
    "is_streaming": False,
    "pdf_uploaded": False,
    "last_error": None,
    "pending_voice_question": None,
    "_pending_question": None,
    "_last_sent_q": "",
    "_last_bridge_transcript": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── API helpers ──────────────────────────────────────────────────────────────

def api(method, path, **kwargs):
    url = f"{st.session_state.api_base}{path}"
    # (connect_timeout, read_timeout) — connect fast, allow slow reads for
    # PDF indexing and other heavy operations
    kwargs.setdefault("timeout", (10, 120))
    try:
        r = getattr(requests, method)(url, **kwargs)
        r.raise_for_status()
        return r
    except requests.exceptions.ConnectionError:
        st.session_state.last_error = "Cannot connect to backend. Is the server running?"
    except requests.exceptions.ReadTimeout:
        st.session_state.last_error = "Request timed out — the server is taking too long. Try again."
    except requests.exceptions.HTTPError as e:
        st.session_state.last_error = f"API {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        st.session_state.last_error = str(e)
    return None

def create_thread(title=None):
    p = {"title": title} if title else {}
    r = api("post", "/threads", params=p)
    if r:
        d = r.json()
        return d["thread_id"], d["title"]
    return None, None

def load_threads():
    r = api("get", "/threads?limit=30")
    if r:
        st.session_state.threads_list = r.json()

def delete_thread(tid):
    api("delete", f"/threads/{tid}")
    load_threads()
    if st.session_state.thread_id == tid:
        st.session_state.thread_id = None
        st.session_state.thread_title = None
        st.session_state.messages = []

def load_thread_history(tid):
    r = api("get", f"/threads/{tid}")
    if r:
        data = r.json()
        msgs = data.get("messages", [])
        rebuilt = []
        for m in msgs:
            if "question" in m:
                rebuilt.append({"role": "user", "content": m["question"]})
            if "answer" in m:
                rebuilt.append({"role": "assistant", "content": m["answer"]})
        st.session_state.messages = rebuilt

def stream_chat(question, thread_id):
    # url = f"{st.session_state.api_base}/threads/{thread_id}/chat/stream"
    url = f"{API_BASE}/threads/{thread_id}/chat/stream"
    try:
        with requests.post(
            url,
            json={"question": question},
            stream=True,
            # (connect_timeout, read_timeout)
            # read_timeout = max seconds of silence between two chunks.
            # The RAG pipeline (retrieve → rerank → evaluate → web search → generate)
            # can take 20-40s before the first token arrives, so set this high.
            timeout=(10, 300),
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.ReadTimeout:
        yield "\n\n❌ Response timed out — the pipeline took too long. Try a simpler question or check the backend."
    except Exception as e:
        yield f"\n\n❌ Stream error: {e}"

def upload_pdf(tid, file_bytes, filename):
    r = api("post", f"/threads/{tid}/upload",
            files={"file": (filename, file_bytes, "application/pdf")})
    return r is not None

def transcribe_bytes(audio_bytes: bytes, filename="audio.webm") -> Optional[str]:
    url = f"{st.session_state.api_base}/transcribe"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    mime_map = {"wav":"audio/wav","mp3":"audio/mpeg","webm":"audio/webm",
                "ogg":"audio/ogg","m4a":"audio/x-m4a","flac":"audio/flac"}
    mime = mime_map.get(ext, "audio/webm")
    try:
        r = requests.post(url, files={"file": (filename, audio_bytes, mime)}, timeout=60)
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except Exception as e:
        st.session_state.last_error = f"Transcription failed: {e}"
        return None

def health_check():
    r = api("get", "/health")
    return r.json() if r else None

def _extract_image_from_response(text: str):
    """
    Scan response text for the inline image marker appended by the backend:
        ![Page image](data:image/jpeg;base64,<b64data>)

    Returns:
        (clean_text, image_bytes | None)
        clean_text   — response with the image markdown line removed
        image_bytes  — raw JPEG bytes ready for st.image(), or None
    """
    pattern = r'!\[Page image\]\((data:image/jpeg;base64,[A-Za-z0-9+/=]+)\)'
    match = re.search(pattern, text)
    if not match:
        return text, None

    data_uri   = match.group(1)                          # data:image/jpeg;base64,...
    b64_data   = data_uri.split(",", 1)[1]               # strip the prefix
    image_bytes = base64.b64decode(b64_data)

    # Remove the image markdown line (and the separator before it) from the text
    clean = re.sub(r'\n*─{30,}\n*🖼️.*?\n?!\[Page image\][^\n]*', '', text, flags=re.DOTALL)
    clean = re.sub(pattern, '', clean).strip()

    return clean, image_bytes

# ─── Render helpers ───────────────────────────────────────────────────────────

def render_message(role, content, image_bytes=None):
    if role == "assistant":
        lines = content.split("\n")
        content = "\n".join(
            l for l in lines
            if not l.startswith("📝 **Thread:**") and not l.startswith("══")
        ).strip()
    if role == "user":
        st.markdown(f'<div class="msg-user"><div class="bubble">{content}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'''<div class="msg-assistant">
            <div class="avatar">🧠</div>
            <div class="bubble">{content}</div>
        </div>''', unsafe_allow_html=True)
        # Render page image below the bubble if this message has one
        if image_bytes:
            st.image(
                BytesIO(image_bytes),
                caption="📄 Relevant page from your PDF",
                use_column_width=True,
            )

def render_welcome():
    st.markdown("""
    <div class="welcome-hero">
        <h1>🧠 CRAG Chatbot</h1>
        <p>Corrective RAG · Hybrid Search · Real-time Voice · PDF Knowledge Base</p>
    </div>
    <div style="text-align:center;margin:1.5rem 0;">
        <div class="chip"><span>📄</span>Upload PDFs</div>
        <div class="chip"><span>🌐</span>Web Fallback</div>
        <div class="chip"><span>🎙️</span>Live Voice</div>
        <div class="chip"><span>⚡</span>Token Streaming</div>
        <div class="chip"><span>🔁</span>Context Resolution</div>
        <div class="chip"><span>🧩</span>BM25 + Vector</div>
    </div>
    <div style="text-align:center;color:#4b5563;font-size:.9rem;margin-top:2rem;">
        ← Create or select a thread in the sidebar to begin
    </div>
    """, unsafe_allow_html=True)

def mic_component(api_base: str) -> None:
    """
    Mic recorder that:
    1. Records audio via MediaRecorder
    2. POSTs directly to /transcribe (JS → backend, no Python middleman)
    3. Writes transcript into localStorage under key 'crag_transcript'
    4. A separate polling component reads localStorage every 400ms and
       writes into the Streamlit text_input (key='_voice_text_input')
       by simulating a React-compatible change event.
    """
    # ── Component 1: Mic button + waveform + fetch /transcribe ────────────────
    mic_html = f"""
<!DOCTYPE html><html><head>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:transparent;display:flex;flex-direction:column;
       align-items:center;gap:8px;padding:8px 0;font-family:system-ui,sans-serif;}}
  #micBtn{{width:64px;height:64px;border-radius:50%;border:none;cursor:pointer;
    background:linear-gradient(135deg,#1e2330,#2a3044);
    box-shadow:0 0 0 3px #374151;font-size:1.6rem;
    display:flex;align-items:center;justify-content:center;
    transition:all .25s;outline:none;}}
  #micBtn:hover{{box-shadow:0 0 0 4px #4b5563;}}
  #micBtn.rec{{background:linear-gradient(135deg,#dc2626,#991b1b);
    box-shadow:0 0 0 4px rgba(220,38,38,.5),0 0 20px rgba(220,38,38,.3);
    animation:pulse 1.3s ease-in-out infinite;}}
  @keyframes pulse{{0%,100%{{transform:scale(1);}}50%{{transform:scale(1.07);}}}}
  #stat{{font-size:.78rem;color:#6b7280;font-family:monospace;text-align:center;min-height:16px;}}
  #tmr{{font-size:.95rem;color:#60a5fa;font-family:monospace;font-weight:700;display:none;}}
  canvas{{width:210px;height:34px;border-radius:6px;background:#12161f;display:none;}}
</style></head>
<body>
  <button id="micBtn" onclick="toggle()">🎙️</button>
  <div id="stat">Click to record</div>
  <div id="tmr">00:00</div>
  <canvas id="wv" width=420 height=68></canvas>
<script>
const API="{api_base}";
const LS_KEY="crag_transcript";
let mr=null,chunks=[],tInt=null,secs=0,raf=null,analyser=null,dArr=null,actx=null;
const btn=document.getElementById('micBtn');
const stat=document.getElementById('stat');
const tmr=document.getElementById('tmr');
const cv=document.getElementById('wv');
const cx=cv.getContext('2d');

function toggle(){{if(mr&&mr.state==='recording')stopRec();else startRec();}}

async function startRec(){{
  try{{
    const stream=await navigator.mediaDevices.getUserMedia({{audio:true}});
    actx=new(window.AudioContext||window.webkitAudioContext)();
    const src=actx.createMediaStreamSource(stream);
    analyser=actx.createAnalyser();analyser.fftSize=256;
    dArr=new Uint8Array(analyser.frequencyBinCount);src.connect(analyser);
    const mt=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':
              MediaRecorder.isTypeSupported('audio/webm')?'audio/webm':'';
    mr=mt?new MediaRecorder(stream,{{mimeType:mt}}):new MediaRecorder(stream);
    chunks=[];
    mr.ondataavailable=e=>{{if(e.data.size>0)chunks.push(e.data);}};
    mr.onstop=async()=>{{
      stream.getTracks().forEach(t=>t.stop());
      cancelAnimationFrame(raf);if(actx)actx.close();cv.style.display='none';
      const blob=new Blob(chunks,{{type:mr.mimeType||'audio/webm'}});
      stat.textContent='⏳ Transcribing…';stat.style.color='#facc15';
      try{{
        const fd=new FormData();fd.append('file',blob,'recording.webm');
        const resp=await fetch(API+'/transcribe',{{method:'POST',body:fd}});
        if(!resp.ok)throw new Error('HTTP '+resp.status);
        const data=await resp.json();
        const text=(data.text||'').trim();
        if(!text)throw new Error('Empty transcript');
        stat.textContent='✅ '+text.slice(0,50)+(text.length>50?'…':'');
        stat.style.color='#4ade80';
        // Write to localStorage — the polling bridge will pick this up
        window.parent.localStorage.setItem(LS_KEY, text);
        window.parent.localStorage.setItem(LS_KEY+'_ts', Date.now().toString());
        setTimeout(()=>{{stat.textContent='Click to record';stat.style.color='#6b7280';}},4000);
      }}catch(err){{
        stat.textContent='❌ '+err.message;stat.style.color='#f87171';
        setTimeout(()=>{{stat.textContent='Click to record';stat.style.color='#6b7280';}},4000);
      }}
    }};
    mr.start(100);btn.classList.add('rec');btn.textContent='⏹️';
    stat.textContent='Recording… click to stop';stat.style.color='#ef4444';
    tmr.style.display='block';cv.style.display='block';
    secs=0;tmr.textContent='00:00';
    tInt=setInterval(()=>{{secs++;
      tmr.textContent=String(Math.floor(secs/60)).padStart(2,'0')+':'+String(secs%60).padStart(2,'0');
    }},1000);
    drawWave();
  }}catch(e){{stat.textContent='❌ Mic denied';stat.style.color='#f87171';}}
}}
function stopRec(){{
  if(mr&&mr.state!=='inactive')mr.stop();
  clearInterval(tInt);btn.classList.remove('rec');btn.textContent='🎙️';tmr.style.display='none';
}}
function drawWave(){{
  raf=requestAnimationFrame(drawWave);if(!analyser)return;
  analyser.getByteTimeDomainData(dArr);
  cx.fillStyle='#12161f';cx.fillRect(0,0,cv.width,cv.height);
  cx.lineWidth=2;cx.strokeStyle='#ef4444';cx.beginPath();
  const sw=cv.width/dArr.length;let x=0;
  for(let i=0;i<dArr.length;i++){{
    const y=(dArr[i]/128.0)*cv.height/2;i===0?cx.moveTo(x,y):cx.lineTo(x,y);x+=sw;
  }}
  cx.lineTo(cv.width,cv.height/2);cx.stroke();
}}
</script></body></html>
"""
    components.html(mic_html, height=185)


def voice_transcript_bridge() -> Optional[str]:
    """
    Polls localStorage for a new transcript every 500ms.
    When found, injects it into a Streamlit text_input and triggers a rerun
    by programmatically clicking a hidden Streamlit button.
    Returns the transcript string if one arrived this run, else None.
    """
    bridge_html = """
<!DOCTYPE html><html><head></head><body>
<script>
const LS_KEY = 'crag_transcript';
const LS_TS  = 'crag_transcript_ts';
let lastTs = null;

function tryInject() {
  try {
    const ts   = window.parent.localStorage.getItem(LS_TS);
    const text = window.parent.localStorage.getItem(LS_KEY);
    if (!text || !ts || ts === lastTs) return;
    lastTs = ts;

    // Clear immediately to prevent double-fire
    window.parent.localStorage.removeItem(LS_KEY);
    window.parent.localStorage.removeItem(LS_TS);

    // Find the hidden text input by its aria-label and inject value
    const doc = window.parent.document;
    const inputs = doc.querySelectorAll('input[type="text"]');
    let target = null;
    for (const inp of inputs) {
      const label = inp.getAttribute('aria-label') || inp.placeholder || '';
      if (label === '_voice_bridge_input') { target = inp; break; }
    }

    if (target) {
      // React-compatible value injection
      const nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
      ).set;
      nativeSetter.call(target, text);
      target.dispatchEvent(new Event('input', { bubbles: true }));
      target.dispatchEvent(new Event('change', { bubbles: true }));
    }
  } catch(e) { /* cross-origin safety */ }
}

setInterval(tryInject, 400);
</script>
</body></html>
"""
    components.html(bridge_html, height=0, scrolling=False)

    # Hidden text input that receives the injected transcript
    val = st.text_input(
        "_voice_bridge_input",
        value="",
        key="_voice_bridge_input",
        label_visibility="collapsed",
    )
    return val if val and val.strip() else None

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 .2rem;display:flex;align-items:center;gap:.5rem;">
        <span style="font-size:1.5rem;">🧠</span>
        <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">CRAG</span>
    </div>
    <div style="font-size:.75rem;color:#4b5563;margin-bottom:.5rem;">
        Corrective Retrieval-Augmented Generation
    </div>
    """, unsafe_allow_html=True)

    # Backend settings
    with st.expander("⚙️ Backend Settings", expanded=False):
        new_base = st.text_input("API URL", value=st.session_state.api_base,
                                 placeholder="http://localhost:8000",
                                 label_visibility="collapsed")
        if new_base != st.session_state.api_base:
            st.session_state.api_base = new_base
        if st.button("🔍 Check Connection", use_container_width=True):
            h = health_check()
            if h:
                st.markdown('<span class="badge badge-green">✅ Connected</span>',
                            unsafe_allow_html=True)
                st.caption(f"v{h.get('version','?')} · {h.get('database','?')}")
            else:
                st.markdown('<span class="badge badge-red">❌ Offline</span>',
                            unsafe_allow_html=True)
                if st.session_state.last_error:
                    st.caption(st.session_state.last_error)

    # ── Threads ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">💬 Threads</div>', unsafe_allow_html=True)

    ca, cb = st.columns([3, 1])
    with ca:
        new_title = st.text_input("title", placeholder="New conversation…",
                                  label_visibility="collapsed", key="new_thread_input")
    with cb:
        if st.button("＋", use_container_width=True, key="btn_new"):
            tid, ttitle = create_thread(new_title or None)
            if tid:
                st.session_state.thread_id    = tid
                st.session_state.thread_title = ttitle
                st.session_state.messages     = []
                st.session_state.pdf_uploaded = False
                load_threads()
                st.rerun()

    if st.button("🔄 Refresh", use_container_width=True, key="btn_refresh"):
        load_threads(); st.rerun()

    if not st.session_state.threads_list:
        load_threads()

    for t in st.session_state.threads_list[:20]:
        tid    = t.get("thread_id", "")
        ttitle = t.get("title", "Untitled")[:28]
        tmsg   = t.get("message_count", 0)
        prefix = "● " if tid == st.session_state.thread_id else ""
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(f"{prefix}{ttitle}\n{tmsg} msgs", key=f"t_{tid}",
                         use_container_width=True):
                st.session_state.thread_id    = tid
                st.session_state.thread_title = ttitle
                st.session_state.messages     = []
                load_thread_history(tid)
                st.rerun()
        with c2:
            if st.button("🗑", key=f"d_{tid}"):
                delete_thread(tid); st.rerun()

    if not st.session_state.threads_list:
        st.caption("No threads yet. Create one above ↑")

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📄 PDF Knowledge Base</div>',
                unsafe_allow_html=True)

    if st.session_state.thread_id:
        pdf_file = st.file_uploader("PDF", type=["pdf"],
                                    label_visibility="collapsed", key="pdf_up")
        if pdf_file and st.button("📤 Index PDF", use_container_width=True):
            with st.spinner(f"Indexing {pdf_file.name}…"):
                ok = upload_pdf(st.session_state.thread_id, pdf_file.read(), pdf_file.name)
            if ok:
                st.success(f"✅ '{pdf_file.name}' indexed!")
                st.session_state.pdf_uploaded = True
            else:
                st.error(f"❌ {st.session_state.last_error or 'Upload failed'}")
        if st.session_state.pdf_uploaded:
            st.markdown('<span class="badge badge-green">📄 PDF Active</span>',
                        unsafe_allow_html=True)
    else:
        st.caption("Select a thread first")

    # ── Cache ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🗄️ Cache</div>', unsafe_allow_html=True)
    if st.session_state.thread_id:
        if st.button("🧹 Clear Cache", use_container_width=True, key="btn_cache"):
            r = api("delete", f"/threads/{st.session_state.thread_id}/cache")
            st.success("Cache cleared!") if r else st.error("Failed")
    else:
        st.caption("Select a thread first")

    # ── 🎙️ Real-time Voice ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🎙️ Voice Input</div>', unsafe_allow_html=True)

    if st.session_state.thread_id:
        st.caption("Click 🎙️ → speak → click ⏹️ to stop. Whisper transcribes automatically.")

        # Mic button: records → POSTs to /transcribe → writes to localStorage
        mic_component(st.session_state.api_base)

        # Bridge: polls localStorage every 400ms → injects into hidden text_input
        incoming = voice_transcript_bridge()

        # New transcript arrived this run via the bridge
        if incoming and incoming != st.session_state.get("_last_bridge_transcript", ""):
            st.session_state["_last_bridge_transcript"] = incoming
            st.session_state.pending_voice_question = incoming
            st.rerun()

        # Transcript preview + confirm
        if st.session_state.pending_voice_question:
            pq = st.session_state.pending_voice_question
            st.info(f"🎙️ **Heard:**\n\n{pq}")
            col_s, col_c = st.columns(2)
            with col_s:
                if st.button("✅ Send", use_container_width=True, key="btn_send_voice"):
                    st.session_state.messages.append({"role": "user", "content": f"🎙️ {pq}"})
                    st.session_state["_pending_question"] = pq
                    st.session_state.pending_voice_question = None
                    st.rerun()
            with col_c:
                if st.button("✕ Cancel", use_container_width=True, key="btn_cancel_voice"):
                    st.session_state.pending_voice_question = None
                    st.rerun()
    else:
        st.caption("Select a thread first")

    # ── Pipeline info ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">ℹ️ Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:.78rem;color:#4b5563;line-height:1.8;">
🔍 Hybrid BM25 + Vector Search<br>
📊 CrossEncoder Reranking<br>
✅ LLM Doc Evaluation<br>
🌐 Tavily Web Fallback<br>
✂️ Sentence-level Refinement<br>
🔄 Context Resolution<br>
⚡ Token Streaming<br>
🧠 MongoDB Memory
</div>
""", unsafe_allow_html=True)

# ─── Main chat area ───────────────────────────────────────────────────────────

if not st.session_state.thread_id:
    render_welcome()

else:
    # Header
    hc1, hc2 = st.columns([4, 1])
    with hc1:
        st.markdown(f"""
        <div style="padding:.5rem 0 .3rem;border-bottom:1px solid #2a3044;margin-bottom:.5rem;">
            <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">
                💬 {st.session_state.thread_title or 'Conversation'}
            </span>
            <span style="font-size:.75rem;color:#4b5563;margin-left:.8rem;">
                {st.session_state.thread_id[:8]}…
            </span>
        </div>""", unsafe_allow_html=True)
    with hc2:
        turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f'<div style="text-align:right;padding-top:.5rem;"><span class="badge badge-blue">{turns} turns</span></div>',
                    unsafe_allow_html=True)

    # Messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;color:#374151;padding:3rem 0;font-size:.95rem;">
            🤖 Ask me anything — I'll search your PDFs and the web!<br>
            <span style="font-size:.82rem;color:#1f2937;">
                Type below ↓ or use the 🎙️ mic in the sidebar
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(
                msg["role"],
                msg["content"],
                image_bytes=msg.get("image"),   # bytes stored when message was saved
            )

    stream_placeholder = st.empty()
    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

    # Input row
    ic1, ic2 = st.columns([9, 1])
    with ic1:
        typed_q = st.text_input(
            "q", placeholder="Ask a question… (Enter to send)",
            label_visibility="collapsed", key="chat_input",
            disabled=st.session_state.is_streaming,
        )
    with ic2:
        send_btn = st.button("➤", use_container_width=True,
                             disabled=st.session_state.is_streaming, key="send_btn")

    # Determine question to send
    question_to_send = None

    if st.session_state.get("_pending_question"):
        question_to_send = st.session_state.pop("_pending_question")

    elif send_btn and typed_q and typed_q.strip():
        q_candidate = typed_q.strip()
        if q_candidate != st.session_state["_last_sent_q"]:
            question_to_send = q_candidate
            st.session_state["_last_sent_q"] = q_candidate
            st.session_state.messages.append({"role": "user", "content": question_to_send})

    # Stream response
    if question_to_send:
        st.session_state.is_streaming = True
        full_response = ""

        with stream_placeholder.container():
            response_box  = st.empty()
            image_slot    = st.empty()   # reserved slot for image, shown after stream ends
            try:
                for token in stream_chat(question_to_send, st.session_state.thread_id):
                    full_response += token

                    # Strip header lines for live display
                    display = full_response
                    if display.startswith("📝 **Thread:**"):
                        parts = display.split("\n", 2)
                        display = parts[-1] if len(parts) >= 2 else display

                    # Strip the image marker from the live bubble — show text only while streaming
                    display_clean = re.sub(
                        r'\n*─{30,}\n*🖼️.*?$', '', display, flags=re.DOTALL
                    ).strip()

                    response_box.markdown(f"""
                    <div class="msg-assistant">
                        <div class="avatar">🧠</div>
                        <div class="bubble">{display_clean}▌</div>
                    </div>""", unsafe_allow_html=True)
            except Exception as e:
                full_response = f"❌ Error: {e}"

        # Strip header
        clean = full_response
        if clean.startswith("📝 **Thread:**"):
            parts = clean.split("\n", 2)
            clean = parts[-1].strip() if len(parts) >= 2 else clean

        # Extract image from final response
        clean_text, image_bytes = _extract_image_from_response(clean)

        # Show image in the reserved slot immediately after streaming
        if image_bytes:
            with image_slot:
                st.image(
                    BytesIO(image_bytes),
                    caption="📄 Relevant page from your PDF",
                    use_column_width=True,
                )

        # Save message with image bytes for history rendering
        st.session_state.messages.append({
            "role":    "assistant",
            "content": clean_text,
            "image":   image_bytes,   # None if no image, bytes if visual chunk was cited
        })
        st.session_state.is_streaming = False
        st.session_state["_last_sent_q"] = ""
        st.rerun()

    # Error
    if st.session_state.last_error and not st.session_state.is_streaming:
        st.error(st.session_state.last_error)
        if st.button("✕ Dismiss", key="dismiss_err"):
            st.session_state.last_error = None
            st.rerun()