import streamlit as st
import io
import base64
from zipfile import ZipFile
import requests
from PIL import Image
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="AI 吳振畫室",
    layout="wide",
    page_icon="🎨",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — theme-aware, professional styling
# Reference: https://github.com/microsoft/Streamlit_UI_Template
# Reference: https://docs.streamlit.io/develop/concepts/configuration/theming
# ============================================================
st.markdown("""
<style>
    /* ---- Header ---- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 0.8rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    /* ---- Generate Button ---- */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.7rem 2rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* ---- Image cards ---- */
    .image-card {
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* ---- Sidebar refinements ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        font-weight: 600;
    }

    /* ---- Prompt template chips ---- */
    .prompt-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .prompt-chip:hover {
        background: rgba(102, 126, 234, 0.2);
    }

    /* ---- Stats cards ---- */
    .stat-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.15);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }

    /* ---- Hide default Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Smooth transitions on containers ---- */
    .stContainer {
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">🎨 AI 吳振畫室</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">使用頂尖 AI 模型，將您的想像化為現實</p>', unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
DEFAULTS = {
    "generated_images": [],       # list of dicts: {url, bytes, prompt, model, timestamp, format}
    "generation_count": 0,
    "total_gen_time": 0.0,
    "output_format": "png",
    "polished_prompt": "",        # stores the last MiniMax-polished prompt
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# MODEL DEFINITIONS
# ============================================================
MODEL_MAP = {
    "FLUX.2 Pro": {
        "owner": "black-forest-labs",
        "name": "flux-2-pro",
        "description": "最高質量，支持參考圖像",
        "icon": "🌟",
        "speed": "中速",
        "quality": "⭐⭐⭐⭐⭐",
        "cost": "$$",
        "badge": "",
    },
    "Nano Banana 2": {
        "owner": "google",
        "name": "nano-banana-2",
        "description": "Google 最新！基於 Gemini 3.1 Flash，精準文字渲染，支持多參考圖",
        "icon": "🍌",
        "speed": "快速",
        "quality": "⭐⭐⭐⭐⭐",
        "cost": "$$",
        "badge": "NEW",
    },
    "Nano Banana Pro": {
        "owner": "google",
        "name": "nano-banana-pro",
        "description": "Google 圖像生成（可能過載）",
        "icon": "🍌",
        "speed": "中速",
        "quality": "⭐⭐⭐⭐",
        "cost": "$$",
        "badge": "",
    },
    "Z-Image Turbo": {
        "owner": "prunaai",
        "name": "z-image-turbo",
        "description": "極速生成，適合快速迭代",
        "icon": "⚡",
        "speed": "極速",
        "quality": "⭐⭐⭐",
        "cost": "$",
        "badge": "",
    },
    "Qwen Image": {
        "owner": "qwen",
        "name": "qwen-image",
        "description": "阿里 Qwen 圖像模型",
        "icon": "🤖",
        "speed": "中速",
        "quality": "⭐⭐⭐⭐",
        "cost": "$$",
        "badge": "",
    },
}

ASPECT_RATIO_MAP = {
    "1:1 (方形)": (1024, 1024),
    "16:9 (橫屏)": (1024, 576),
    "9:16 (豎屏)": (576, 1024),
    "4:3 (橫屏)": (1024, 768),
    "3:4 (豎屏)": (768, 1024),
    "3:2 (橫屏)": (1152, 768),
    "2:3 (豎屏)": (768, 1152),
    "4:5 (豎屏)": (832, 1024),
    "5:4 (橫屏)": (1024, 832),
    "21:9 (超寬)": (1024, 448),
    "9:21 (超長)": (448, 1024),
}

MODEL_PARAMS = {
    "FLUX.2 Pro": {
        "aspect_ratio": True, "width": False, "height": False,
        "ref_image": True, "negative_prompt": False,
    },
    "Nano Banana 2": {
        "aspect_ratio": True, "width": False, "height": False,
        "ref_image": True, "negative_prompt": False,
    },
    "Nano Banana Pro": {
        "aspect_ratio": False, "width": True, "height": True,
        "ref_image": True, "negative_prompt": False,
    },
    "Z-Image Turbo": {
        "aspect_ratio": False, "width": False, "height": True,
        "ref_image": False, "negative_prompt": True,
    },
    "Qwen Image": {
        "aspect_ratio": False, "width": True, "height": True,
        "ref_image": False, "negative_prompt": False,
    },
}

# Prompt templates for quick start
PROMPT_TEMPLATES = {
    "📷 攝影風格": "professional photography, 8k uhd, high detail, sharp focus, studio lighting",
    "🎨 油畫風格": "oil painting style, rich colors, textured brushstrokes, classical art",
    "🌸 動漫風格": "anime style, vibrant colors, detailed illustration, studio ghibli inspired",
    "🏙️ 賽博朋克": "cyberpunk style, neon lights, futuristic city, rain-soaked streets, cinematic",
    "🖼️ 水彩風格": "watercolor painting, soft edges, pastel colors, artistic, delicate",
    "✏️ 素描風格": "pencil sketch, detailed line art, black and white, cross-hatching",
}

# ============================================================
# REPLICATE API TOKEN
# ============================================================
REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN", "")
MINIMAX_API_KEY = st.secrets.get("MINIMAX_API_KEY", "")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_model_version(owner: str, name: str, token: str) -> Optional[str]:
    """Get the latest model version from Replicate (cached for 1 hour)."""
    url = f"https://api.replicate.com/v1/models/{owner}/{name}"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        version = data.get("latest_version", {}).get("id")
        logger.info(f"Got version for {owner}/{name}: {version}")
        return version
    except Exception as e:
        logger.error(f"Failed to get model version for {owner}/{name}: {e}")
        return None


def image_to_base64(img_file) -> str:
    """Convert an uploaded image file to a base64 data URI."""
    img_file.seek(0)
    img_data = img_file.read()
    # Detect MIME type from extension
    name = getattr(img_file, "name", "image.png").lower()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        mime = "image/jpeg"
    elif name.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/png"
    b64 = base64.b64encode(img_data).decode()
    return f"data:{mime};base64,{b64}"


def download_image_bytes(img_url: str) -> Optional[bytes]:
    """Download an image URL and return raw bytes. Returns None on failure."""
    try:
        resp = requests.get(img_url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return None


def polish_prompt_with_minimax(prompt: str) -> dict:
    """
    Use MiniMax M2.5 to expand and enhance an image generation prompt.
    Returns {"polished": str} on success or {"error": str} on failure.
    """
    system_prompt = (
        "You are a professional AI image generation prompt engineer. "
        "When given a rough description, expand and optimize it into a high-quality "
        "English prompt for image generation models (FLUX, Stable Diffusion, etc.).\n"
        "Requirements:\n"
        "1. Preserve the original intent, subject, and theme\n"
        "2. Add specific visual details: lighting, texture, composition, color palette, perspective\n"
        "3. Append quality-boosting keywords such as: masterpiece, 8k uhd, highly detailed, "
        "cinematic lighting, sharp focus, photorealistic\n"
        "4. Output ONLY the optimized English prompt — no explanations, no prefixes, no quotes"
    )
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "MiniMax-M2.5",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Optimize this prompt: {prompt}"},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(
            "https://api.minimax.io/v1/text/chatcompletion_v2",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"MiniMax response keys: {list(data.keys())}")

        # Check base_resp for API-level errors (status_code != 0 means error)
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            msg = base_resp.get("status_msg", "未知錯誤")
            logger.error(f"MiniMax base_resp error: {base_resp}")
            return {"error": f"MiniMax 錯誤 ({base_resp.get('status_code')}): {msg}", "raw": data}

        choices = data.get("choices")
        if not choices:
            # Some MiniMax responses use "reply" (older API) or return the full body differently
            reply = data.get("reply") or data.get("output") or data.get("text")
            if reply:
                return {"polished": str(reply).strip()}
            logger.error(f"Unexpected MiniMax response: {str(data)[:400]}")
            return {"error": f"API 返回格式異常，請稍後重試。響應: {str(data)[:200]}", "raw": data}

        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            return {"error": "模型未返回內容，請重試", "raw": data}
        return {"polished": content}

    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.text[:300]
        except Exception:
            detail = str(e)
        logger.error(f"MiniMax API HTTP error: {detail}")
        return {"error": f"MiniMax API 錯誤: {detail}"}
    except Exception as e:
        logger.error(f"MiniMax error: {e}")
        return {"error": f"潤色失敗: {str(e)}"}


def build_input_params(
    prompt: str,
    model_key: str,
    model_info: dict,
    params: dict,
    ref_images: list,
) -> dict:
    """Build the input parameters dict for a specific model."""
    owner = model_info["owner"]
    name = model_info["name"]
    input_params = {"prompt": prompt}

    # --- FLUX.2 Pro ---
    if owner == "black-forest-labs" and name == "flux-2-pro":
        if params.get("aspect_ratio"):
            input_params["aspect_ratio"] = params["aspect_ratio"].split(" ")[0]
        else:
            input_params["width"] = params.get("width", 1024)
            input_params["height"] = params.get("height", 1024)
        if params.get("seed") and params["seed"] > 0:
            input_params["seed"] = int(params["seed"])
        if ref_images:
            ref_images[0].seek(0)
            input_params["image"] = image_to_base64(ref_images[0])

    # --- Nano Banana 2 (Gemini 3.1 Flash Image) ---
    elif owner == "google" and name == "nano-banana-2":
        if params.get("aspect_ratio"):
            input_params["aspect_ratio"] = params["aspect_ratio"].split(" ")[0]
        if params.get("seed") and params["seed"] > 0:
            input_params["seed"] = int(params["seed"])
        if ref_images:
            # nano-banana-2 supports multiple reference images
            if len(ref_images) == 1:
                ref_images[0].seek(0)
                input_params["image"] = image_to_base64(ref_images[0])
            else:
                input_params["images"] = [image_to_base64(f) for f in ref_images]

    # --- Nano Banana Pro ---
    elif owner == "google" and name == "nano-banana-pro":
        input_params["width"] = params.get("width", 1024)
        input_params["height"] = params.get("height", 1024)
        if params.get("seed") and params["seed"] > 0:
            input_params["seed"] = int(params["seed"])
        if ref_images:
            ref_images[0].seek(0)
            input_params["image"] = image_to_base64(ref_images[0])

    # --- Z-Image Turbo ---
    elif owner == "prunaai" and name == "z-image-turbo":
        input_params["image_size"] = params.get("height", 1024)
        if params.get("seed") and params["seed"] > 0:
            input_params["seed"] = int(params["seed"])
        if params.get("negative_prompt"):
            input_params["negative_prompt"] = params["negative_prompt"]

    # --- Qwen Image ---
    elif owner == "qwen" and name == "qwen-image":
        input_params["width"] = params.get("width", 1024)
        input_params["height"] = params.get("height", 1024)
        if params.get("seed") and params["seed"] > 0:
            input_params["seed"] = int(params["seed"])

    # Common params
    if params.get("num_outputs", 1) > 1:
        input_params["num_outputs"] = params["num_outputs"]
    if params.get("output_format"):
        input_params["output_format"] = params["output_format"]

    return input_params


def generate_image_replicate(
    prompt: str,
    model_key: str,
    model_info: dict,
    params: dict,
    ref_images: Optional[list] = None,
) -> dict:
    """
    Generate image using Replicate API with retry logic.

    Returns:
        dict with keys:
          - "urls": list[str]   on success
          - "error": str        on failure
    """
    owner = model_info["owner"]
    name = model_info["name"]
    max_retries = 3

    # Get model version (cached)
    version = get_model_version(owner, name, REPLICATE_API_TOKEN)
    if not version:
        return {"error": f"無法獲取模型版本: {owner}/{name}。請檢查模型名稱是否正確。"}

    input_params = build_input_params(prompt, model_key, model_info, params, ref_images or [])

    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    # Safe logging (truncate base64)
    log_params = {}
    for k, v in input_params.items():
        if isinstance(v, str) and len(v) > 200:
            log_params[k] = v[:80] + f"...({len(v)} chars)"
        else:
            log_params[k] = v
    logger.info(f"Input params for {owner}/{name}: {log_params}")

    if params.get("debug_mode"):
        st.write("### 🔧 調試信息")
        st.json(log_params)

    # Retry loop
    for retry in range(max_retries):
        if retry > 0:
            wait_time = 5 * (retry + 1)
            logger.info(f"Retry {retry}/{max_retries}, waiting {wait_time}s...")
            time.sleep(wait_time)

        try:
            payload = {"version": version, "input": input_params}
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            # Handle retryable errors
            if response.status_code == 503:
                if retry < max_retries - 1:
                    logger.warning("Service unavailable (503), retrying...")
                    continue
                return {"error": "❌ 服務暫時不可用，請稍後再試或換用其他模型"}

            if response.status_code == 429:
                if retry < max_retries - 1:
                    logger.warning("Rate limited (429), retrying...")
                    time.sleep(10)
                    continue
                return {"error": "❌ 請求過於頻繁，請稍後再試"}

            if response.status_code != 201:
                error_text = response.text
                if "E003" in error_text or "high demand" in error_text.lower():
                    if retry < max_retries - 1:
                        continue
                    return {"error": f"❌ {model_key} 目前過載，請換用其他模型或稍後再試"}
                logger.error(f"API error {response.status_code}: {error_text[:500]}")

            response.raise_for_status()
            result = response.json()
            prediction_id = result.get("id", "unknown")
            logger.info(f"Prediction created: {prediction_id}")

            # Poll for result
            prediction_url = result.get("urls", {}).get("get")
            if not prediction_url:
                return {"error": "API 未返回結果 URL"}

            max_polls = 180  # 6 minutes max
            for poll in range(max_polls):
                time.sleep(2)
                status_resp = requests.get(prediction_url, headers=headers, timeout=30)
                status_data = status_resp.json()
                status = status_data.get("status", "unknown")

                if status == "succeeded":
                    output = status_data.get("output")
                    if isinstance(output, list):
                        return {"urls": output}
                    elif isinstance(output, str):
                        return {"urls": [output]}
                    else:
                        return {"error": f"意外的輸出格式: {type(output)}"}

                elif status == "failed":
                    error_msg = status_data.get("error", "生成失敗，未提供詳細信息")
                    return {"error": f"❌ 生成失敗: {error_msg}"}

                elif status in ("starting", "processing"):
                    continue
                else:
                    return {"error": f"未知狀態: {status}"}

            return {"error": "⏱️ 生成超時 (超過6分鐘)，請稍後重試"}

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.text[:500]
            except Exception:
                error_detail = str(e)
            logger.error(f"HTTP error: {error_detail}")

            if "E003" in str(error_detail) and retry < max_retries - 1:
                continue
            return {"error": f"HTTP 錯誤: {error_detail}"}

        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                continue
            return {"error": "❌ 請求超時，請檢查網絡連接"}

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": f"未知錯誤: {str(e)}"}

    return {"error": "重試次數已用完，請稍後再試"}


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ 模型設置")

    # Model selection
    model_names = list(MODEL_MAP.keys())
    model_labels = [f"{MODEL_MAP[m]['icon']} {m}" for m in model_names]
    selected_idx = st.selectbox(
        "選擇 AI 模型",
        range(len(model_labels)),
        format_func=lambda x: model_labels[x],
        key="model_select",
    )
    model_key = model_names[selected_idx]
    model_info = MODEL_MAP[model_key]
    model_settings = MODEL_PARAMS[model_key]

    # Model info card
    badge_html = (
        f'<span style="background:#28a745; color:white; font-size:0.7rem; '
        f'font-weight:700; padding:0.1rem 0.4rem; border-radius:4px; '
        f'margin-left:0.4rem; vertical-align:middle;">{model_info["badge"]}</span>'
        if model_info.get("badge") else ""
    )
    st.markdown(f"""
    <div style="padding:0.8rem; border-radius:10px;
                background: rgba(102,126,234,0.08);
                border-left: 4px solid #667eea; margin-bottom:1rem;">
        <strong>{model_info['icon']} {model_key}</strong>{badge_html}<br/>
        <span style="font-size:0.85rem;">{model_info['description']}</span><br/>
        <span style="font-size:0.8rem;">
            速度: {model_info['speed']} · 質量: {model_info['quality']} · 成本: {model_info['cost']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if model_key == "Nano Banana 2":
        st.info("✨ Nano Banana 2 是 Google 最新模型，支持精準文字渲染與多圖參考")
    elif model_key == "Nano Banana Pro":
        st.warning("⚠️ Nano Banana Pro 可能過載，如遇錯誤請換用其他模型")

    st.divider()

    # --- Reference Images ---
    st.markdown("### 📷 參考圖像")
    ref_images = []

    if model_settings.get("ref_image", False):
        uploaded_refs = st.file_uploader(
            "上傳參考圖像 (可選)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="上傳一張或多張參考圖像來引導生成",
            key="ref_uploader",
        )
        if uploaded_refs:
            st.caption(f"已選擇 {len(uploaded_refs)} 張圖片")
            thumb_cols = st.columns(min(len(uploaded_refs), 3))
            for i, ref_file in enumerate(uploaded_refs):
                with thumb_cols[i % len(thumb_cols)]:
                    try:
                        ref_file.seek(0)
                        img = Image.open(ref_file)
                        st.image(img, width=80)
                    except Exception as e:
                        st.error(f"預覽失敗: {e}")
            ref_images = uploaded_refs
    else:
        st.caption("ℹ️ 此模型不支持參考圖像")

    st.divider()

    # --- Generation Settings ---
    st.markdown("### 🎛️ 生成設置")

    seed = st.number_input(
        "隨機種子 (0=隨機)",
        min_value=0,
        max_value=2**31,
        value=0,
        help="設為 0 則每次隨機。固定值可重現結果。",
        key="seed_input",
    )

    num_outputs = st.slider(
        "生成數量",
        min_value=1,
        max_value=4,
        value=1,
        help="每次生成的圖片數量",
        key="num_outputs_slider",
    )

    # Aspect ratio / dimensions
    aspect_ratio = None
    img_width = 1024
    img_height = 1024

    if model_settings.get("aspect_ratio"):
        aspect_ratio = st.selectbox(
            "寬高比",
            list(ASPECT_RATIO_MAP.keys()),
            index=0,
            key="aspect_ratio_select",
        )
        w, h = ASPECT_RATIO_MAP[aspect_ratio]
        st.caption(f"📐 尺寸: {w} × {h} px")
        img_width, img_height = w, h
    else:
        col_w, col_h = st.columns(2)
        with col_w:
            if model_settings.get("width"):
                img_width = st.number_input("寬度", 256, 2048, 1024, 64, key="width_input")
        with col_h:
            if model_settings.get("height"):
                img_height = st.number_input("高度", 256, 2048, 1024, 64, key="height_input")

    output_format = st.selectbox(
        "輸出格式",
        ["png", "webp", "jpg"],
        index=0,
        key="format_select",
    )
    st.session_state.output_format = output_format

    # Negative prompt (for models that support it)
    negative_prompt = ""
    if model_settings.get("negative_prompt"):
        negative_prompt = st.text_input(
            "負面提示詞",
            placeholder="不想出現的元素，如: blurry, low quality",
            key="negative_prompt_input",
        )

    debug_mode = st.checkbox("🔧 調試模式", value=False, key="debug_check")

    st.divider()

    # --- Batch Prompts ---
    st.markdown("### 📄 批量提示詞")
    uploaded_file = st.file_uploader(
        "上傳 .txt 文件 (每行一個提示詞)",
        type="txt",
        key="batch_uploader",
    )

    st.divider()

    # --- Session Stats ---
    st.markdown("### 📊 本次統計")
    stat_cols = st.columns(2)
    with stat_cols[0]:
        st.metric("生成數量", st.session_state.generation_count)
    with stat_cols[1]:
        avg_time = (
            st.session_state.total_gen_time / st.session_state.generation_count
            if st.session_state.generation_count > 0
            else 0
        )
        st.metric("平均耗時", f"{avg_time:.1f}s")

    st.divider()

    # Clear history
    if st.button("🗑️ 清除所有歷史記錄", type="secondary", key="clear_all"):
        st.session_state.generated_images = []
        st.session_state.generation_count = 0
        st.session_state.total_gen_time = 0.0
        st.rerun()


# ============================================================
# MAIN CONTENT
# ============================================================

# --- Prompt Templates ---
st.markdown("#### 💡 風格模板 (點擊添加到提示詞)")
template_cols = st.columns(len(PROMPT_TEMPLATES))
selected_template = ""
for i, (label, template_text) in enumerate(PROMPT_TEMPLATES.items()):
    with template_cols[i]:
        if st.button(label, key=f"tmpl_{i}", use_container_width=True):
            selected_template = template_text

# --- Prompt Input ---
prompt_label_col, polish_label_col = st.columns([3, 1])
with prompt_label_col:
    st.markdown("#### ✍️ 輸入提示詞")
with polish_label_col:
    st.markdown(
        '<p style="font-size:0.75rem; color:#888; margin-top:2.2rem; text-align:right;">'
        'Powered by MiniMax M2.5</p>',
        unsafe_allow_html=True,
    )

default_prompt = selected_template if selected_template else ""
prompt = st.text_area(
    "描述您想生成的圖像",
    value=default_prompt,
    height=120,
    placeholder="例如: 一隻可愛的橘貓坐在窗台上，陽光透過窗戶灑在它身上，溫馨的家庭氛圍，高清攝影風格...",
    key="prompt_input",
    label_visibility="collapsed",
)

# --- Prompt polish row ---
char_col, polish_col = st.columns([3, 1])
with char_col:
    if prompt:
        st.caption(f"📝 {len(prompt)} 字符")
with polish_col:
    polish_btn = st.button(
        "✨ 潤色提示詞",
        key="polish_btn",
        use_container_width=True,
        help="使用 MiniMax M2.5 AI 優化您的提示詞，使其更適合圖像生成",
    )

if polish_btn:
    if not prompt.strip():
        st.warning("⚠️ 請先輸入提示詞再進行潤色")
    elif not MINIMAX_API_KEY:
        st.error("❌ 請在 `.streamlit/secrets.toml` 中設置 `MINIMAX_API_KEY`")
    else:
        with st.spinner("✨ MiniMax M2.5 正在潤色提示詞..."):
            polish_result = polish_prompt_with_minimax(prompt.strip())
        if debug_mode and "raw" in polish_result:
            with st.expander("🔧 MiniMax 原始響應", expanded=True):
                st.json(polish_result["raw"])
        if "error" in polish_result:
            st.error(polish_result["error"])
        else:
            st.session_state.polished_prompt = polish_result["polished"]

# --- Polished prompt display ---
if st.session_state.get("polished_prompt"):
    st.markdown("""
    <div style="padding:0.6rem 0.8rem; border-radius:8px;
                background: rgba(40,167,69,0.08);
                border-left: 4px solid #28a745; margin-bottom:0.8rem;">
        <strong style="color:#28a745;">✨ 潤色後的提示詞</strong>
        <span style="font-size:0.8rem; color:#888; margin-left:0.5rem;">by MiniMax M2.5</span>
    </div>
    """, unsafe_allow_html=True)
    polished_display = st.text_area(
        "潤色結果",
        value=st.session_state.polished_prompt,
        height=100,
        key="polished_display_area",
        label_visibility="collapsed",
    )
    apply_col, cancel_col, _ = st.columns([2, 1, 2])
    with apply_col:
        if st.button("✅ 套用此提示詞", key="apply_polish_btn", use_container_width=True):
            st.session_state["prompt_input"] = st.session_state.polished_prompt
            st.session_state.polished_prompt = ""
            st.rerun()
    with cancel_col:
        if st.button("✖ 取消", key="cancel_polish_btn", use_container_width=True):
            st.session_state.polished_prompt = ""
            st.rerun()

# --- Generate Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button(
        "🎨 開始生成",
        type="primary",
        use_container_width=True,
        key="generate_btn",
    )

# ============================================================
# GENERATION LOGIC
# ============================================================
if generate_btn:
    # Gather prompts
    prompts = []
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        prompts = [line.strip() for line in content.splitlines() if line.strip()]
    elif prompt.strip():
        prompts = [prompt.strip()]

    # Validate
    if not REPLICATE_API_TOKEN:
        st.error("❌ 缺少 Replicate API Token。請在 `.streamlit/secrets.toml` 中設置 `REPLICATE_API_TOKEN`。")
    elif not prompts:
        st.error("❌ 請輸入提示詞或上傳提示詞文件。")
    else:
        total = len(prompts) * num_outputs
        progress_bar = st.progress(0, text="準備中...")
        status_placeholder = st.empty()
        start_time_total = time.time()

        new_images = []
        error_count = 0

        for p_idx, current_prompt in enumerate(prompts):
            status_placeholder.info(
                f"🎨 正在生成 ({p_idx + 1}/{len(prompts)}): "
                f"**{current_prompt[:60]}{'...' if len(current_prompt) > 60 else ''}**"
            )

            gen_params = {
                "aspect_ratio": aspect_ratio,
                "width": img_width,
                "height": img_height,
                "seed": seed if seed > 0 else None,
                "num_outputs": num_outputs,
                "output_format": output_format,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "debug_mode": debug_mode,
            }

            gen_start = time.time()
            result = generate_image_replicate(
                current_prompt, model_key, model_info, gen_params,
                ref_images if ref_images else None,
            )
            gen_elapsed = time.time() - gen_start

            if "error" in result:
                st.error(result["error"])
                error_count += 1
            elif "urls" in result:
                for url in result["urls"]:
                    # Download image bytes immediately for caching
                    img_bytes = download_image_bytes(url)
                    if img_bytes:
                        new_images.append({
                            "url": url,
                            "bytes": img_bytes,
                            "prompt": current_prompt,
                            "model": model_key,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "format": output_format,
                            "gen_time": gen_elapsed,
                        })
                    else:
                        # Store URL even if download fails
                        new_images.append({
                            "url": url,
                            "bytes": None,
                            "prompt": current_prompt,
                            "model": model_key,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "format": output_format,
                            "gen_time": gen_elapsed,
                        })

            progress_bar.progress(
                (p_idx + 1) / len(prompts),
                text=f"已完成 {p_idx + 1}/{len(prompts)}",
            )

        total_elapsed = time.time() - start_time_total

        # Save to session state
        st.session_state.generated_images = new_images + st.session_state.generated_images
        st.session_state.generation_count += len(new_images)
        st.session_state.total_gen_time += total_elapsed

        # Clear progress
        progress_bar.empty()
        status_placeholder.empty()

        if new_images:
            st.success(
                f"🎉 成功生成 {len(new_images)} 張圖片！"
                f"（耗時 {total_elapsed:.1f} 秒）"
            )
        if error_count > 0:
            st.warning(f"⚠️ {error_count} 個提示詞生成失敗")


# ============================================================
# IMAGE GALLERY
# ============================================================
if st.session_state.generated_images:
    st.divider()

    # Gallery header with count
    gallery_col1, gallery_col2 = st.columns([3, 1])
    with gallery_col1:
        st.markdown(f"## 🖼️ 圖片庫 ({len(st.session_state.generated_images)} 張)")
    with gallery_col2:
        if st.button("🗑️ 清除所有圖片", key="clear_gallery"):
            st.session_state.generated_images = []
            st.rerun()

    # Display in responsive grid
    cols_per_row = 3
    images = st.session_state.generated_images
    fmt = st.session_state.output_format

    for row_start in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            img_idx = row_start + col_idx
            if img_idx >= len(images):
                break

            img_data = images[img_idx]
            with col:
                with st.container(border=True):
                    # Display image
                    try:
                        if img_data["bytes"]:
                            img = Image.open(io.BytesIO(img_data["bytes"]))
                            st.image(img, use_container_width=True)
                        else:
                            st.image(img_data["url"], use_container_width=True)
                    except Exception as e:
                        st.error(f"無法加載: {e}")
                        continue

                    # Metadata
                    st.caption(
                        f"{img_data['model']}  ·  {img_data['timestamp']}  ·  "
                        f"⏱️ {img_data.get('gen_time', 0):.1f}s"
                    )

                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if img_data["bytes"]:
                            st.download_button(
                                "📥 下載",
                                data=img_data["bytes"],
                                file_name=f"ai_{img_idx + 1}_{img_data['timestamp'].replace(':', '').replace(' ', '_')}.{img_data['format']}",
                                mime=f"image/{img_data['format']}",
                                key=f"dl_{img_idx}",
                                use_container_width=True,
                            )
                    with btn_col2:
                        if st.button("🗑️", key=f"del_{img_idx}", use_container_width=True):
                            st.session_state.generated_images.pop(img_idx)
                            st.rerun()

                    # Expandable prompt
                    with st.expander("📝 提示詞", expanded=False):
                        st.code(img_data["prompt"], language=None)

    # --- Batch Download ---
    if len(images) > 1:
        st.divider()
        st.markdown("### 📦 批量下載")

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            for idx, img_data in enumerate(images):
                if img_data["bytes"]:
                    zf.writestr(
                        f"ai_image_{idx + 1}.{img_data['format']}",
                        img_data["bytes"],
                    )
        zip_buffer.seek(0)

        st.download_button(
            "📥 下載全部圖片 (ZIP)",
            data=zip_buffer,
            file_name=f"ai_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            key="zip_download",
            use_container_width=False,
        )


# ============================================================
# FOOTER — USAGE GUIDE
# ============================================================
st.divider()
with st.expander("📖 使用說明", expanded=False):
    st.markdown("""
    ### 🎯 快速開始
    1. **選擇模型** — 在左側邊欄選擇合適的 AI 模型
    2. **選擇風格** — 點擊風格模板快速填入提示詞風格
    3. **輸入提示詞** — 詳細描述您想生成的圖像（中英文均可）
    4. **✨ 潤色提示詞** — 點擊「潤色提示詞」讓 MiniMax M2.5 自動優化您的描述
    5. **調整設置** — 設置尺寸、數量、種子等參數
    6. **生成圖像** — 點擊「開始生成」按鈕
    7. **下載保存** — 單張下載或批量打包 ZIP

    ### 🤖 模型對比
    | 模型 | 質量 | 速度 | 參考圖 | 成本 | 狀態 |
    |------|------|------|--------|------|------|
    | 🌟 FLUX.2 Pro | ⭐⭐⭐⭐⭐ | 中速 | ✅ | $$ | 穩定 |
    | 🍌 Nano Banana 2 🆕 | ⭐⭐⭐⭐⭐ | 快速 | ✅ 多圖 | $$ | 新上線 |
    | 🍌 Nano Banana Pro | ⭐⭐⭐⭐ | 中速 | ✅ | $$ | 可能過載 |
    | ⚡ Z-Image Turbo | ⭐⭐⭐ | 極速 | ❌ | $ | 穩定 |
    | 🤖 Qwen Image | ⭐⭐⭐⭐ | 中速 | ❌ | $$ | 穩定 |

    ### 💡 提示詞技巧
    - **具體描述**：越詳細越好，包括主體、場景、光線、風格
    - **加入風格詞**：如 "cinematic", "8k uhd", "studio lighting"
    - **使用負面提示**：排除不需要的元素（Z-Image Turbo 支持）
    - **固定種子**：找到喜歡的效果後記下種子值，方便重現

    ### ⚠️ 常見問題
    | 問題 | 解決方案 |
    |------|---------|
    | E003 過載錯誤 | 換用 FLUX.2 Pro 或 Nano Banana 2 |
    | 生成超時 | 檢查網絡，或嘗試較小尺寸 |
    | API Token 錯誤 | 在 `.streamlit/secrets.toml` 中設置 `REPLICATE_API_TOKEN` 和 `MINIMAX_API_KEY` |
    | 圖片質量不佳 | 優化提示詞，使用 Nano Banana 2 或 FLUX.2 Pro |
    | 文字渲染模糊 | 改用 Nano Banana 2，專為精準文字渲染優化 |
    """)

# Subtle footer
st.markdown(
    '<p style="text-align:center; color:#aaa; font-size:0.8rem; margin-top:2rem;">'
    "AI 吳振畫室 · 振視科技 · tot@alexzhenwu.com · Powered by Replicate · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
