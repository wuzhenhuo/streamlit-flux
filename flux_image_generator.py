import streamlit as st
import replicate
import io
import base64
from zipfile import ZipFile
import requests
import asyncio
from PIL import Image
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置页面标题
st.set_page_config(page_title="Flux 图像生成器", layout="wide")

# 标题
st.title("Flux 吳振畫室")

# 选择模型
model = st.selectbox(
    "選擇模型",
    ["black-forest-labs/flux-schnell", "black-forest-labs/flux-dev"],
    help="選擇要使用的Flux模型"
)

# 输入提示词
prompt = st.text_area("輸入提示詞", help="描述您想生成的圖像")

# 上传txt文件
uploaded_file = st.file_uploader("上傳提示詞文件 (每行一個提示詞)", type="txt")

# 设置参数
with st.expander("高級設置"):
    seed = st.number_input("隨機種子", min_value=0, help="設置隨機種子以獲得可重復的生成結果")
    num_outputs = st.slider("每個提示詞的輸出數量", min_value=1, max_value=4, value=1, help="每個提示詞生成的圖像數量")
    aspect_ratio = st.selectbox(
        "寬高比",
        ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        index=0,
        help="生成圖像的寬高比"
    )
    output_format = st.selectbox("輸出格式", ["png", "webp", "jpg"], index=0, help="輸出圖像的格式")
    output_quality = st.slider("輸出質量", min_value=0, max_value=100, value=100, help="輸出圖像的質量,從0到100。100是最佳質量,0是最低質量。對於.png輸出不相關")
    disable_safety_checker = st.checkbox("禁用安全檢查器", help="禁用生成圖像的安全檢查器。此功能僅通過API可用。")

async def generate_image_async(prompt, model, input_data, timeout=45):
    try:
        prediction = await replicate.predictions.async_create(
            model=model,
            input=input_data
        )
        start_time = asyncio.get_event_loop().time()
        while prediction.status != "succeeded":
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise asyncio.TimeoutError(f"生成圖像超時（{timeout}秒）")
            await asyncio.sleep(1)
            prediction = await replicate.predictions.async_get(prediction.id)
        
        logger.info(f"API 響應: {prediction.output}")
        logger.info(f"完整的 API 響應: {prediction}")

        if prediction.output:
            if isinstance(prediction.output, list):
                return prediction.output[0] if prediction.output else None
            elif isinstance(prediction.output, str):
                return prediction.output
            else:
                logger.error(f"未知的 API 響應格式: {type(prediction.output)}")
                return None
        else:
            logger.error("API 響應為空")
            return None
    except asyncio.TimeoutError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"生成圖像時發生錯誤: {str(e)}")
        return None

def get_image_download_link(img_url, filename):
    response = requests.get(img_url)
    response.raise_for_status()
    img_data = response.content
    b64 = base64.b64encode(img_data).decode()
    file_size = len(img_data)
    logger.info(f"下載的圖像大小: {file_size} 字節")
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">下载图片 ({file_size/1024:.2f} KB)</a>'

# 生成按钮
if st.button("生成圖像"):
    prompts = []
    if uploaded_file:
        prompts = [line.decode("utf-8").strip() for line in uploaded_file]
    elif prompt:
        prompts = [prompt]
    
    if not prompts:
        st.error("請輸入提示詞或上傳提示詞文件")
    else:
        with st.spinner(f"正在生成 {len(prompts) * num_outputs} 張圖像..."):
            async def generate_all_images():
                tasks = []
                for current_prompt in prompts:
                    for _ in range(num_outputs):
                        input_data = {
                            "prompt": current_prompt,
                            "seed": seed if seed else None,
                            "aspect_ratio": aspect_ratio,
                            "output_format": output_format,
                            "output_quality": output_quality,
                            "disable_safety_checker": disable_safety_checker
                        }
                        input_data = {k: v for k, v in input_data.items() if v is not None}
                        tasks.append(generate_image_async(current_prompt, model, input_data, timeout=45))
                return await asyncio.gather(*tasks)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            generated_images = loop.run_until_complete(generate_all_images())
            
            # 显示生成的图像
            for i, image_url in enumerate(generated_images):
                if image_url:
                    logger.info(f"处理图像 URL: {image_url}")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        try:
                            if not image_url.startswith(('http://', 'https://')):
                                raise ValueError(f"无效的 URL: {image_url}")
                            
                            response = requests.get(image_url)
                            response.raise_for_status()
                            
                            image = Image.open(io.BytesIO(response.content))
                            logger.info(f"图像 {i+1} 尺寸: {image.size}")
                            st.image(image, caption=f"生成的图像 {i+1}", use_column_width=True)
                            logger.info(f"成功显示图像 {i+1}")
                        except requests.RequestException as e:
                            logger.error(f"下载图像时发生错误: {str(e)}")
                            st.error(f"无法加载图像 {i+1}: {str(e)}")
                        except ValueError as e:
                            logger.error(str(e))
                            st.error(f"无效的图像 URL {i+1}: {str(e)}")
                        except Exception as e:
                            logger.error(f"处理图像时发生未知错误: {str(e)}")
                            st.error(f"处理图像 {i+1} 时发生错误: {str(e)}")
                    with col2:
                        st.markdown(get_image_download_link(image_url, f"generated_image_{i+1}.{output_format}"), unsafe_allow_html=True)
                        st.text_area("提示词", prompts[i // num_outputs], height=100)
                else:
                    logger.warning(f"图像 {i+1} 的 URL 为空")
        
        # 批量下载
        if len(generated_images) > 1:
            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                for i, image_url in enumerate(generated_images):
                    if image_url:
                        response = requests.get(image_url)
                        zip_file.writestr(f"generated_image_{i+1}.{output_format}", response.content)
            
            zip_buffer.seek(0)
            b64 = base64.b64encode(zip_buffer.getvalue()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="generated_images.zip">下载所有图片</a>'
            st.markdown(href, unsafe_allow_html=True)

# 添加说明
st.markdown("""
## 使用說明
- 選擇要使用的Flux模型。
- 在文本框中輸入描述您想要生成的圖像的提示詞，或上傳包含多個提示詞的txt文件（每行一個提示詞）。
- 如果需要，可以展開“高級設置”調整其他參數。
- 點擊“生成圖像”按鈕開始生成過程。
- 生成的圖像將顯示在頁面上，您可以單獨下載每張圖片或批量下載所有圖片。

注意：確保您已經設置了REPLICATE_API_TOKEN環境變量，否則API調用將失敗。
""")
