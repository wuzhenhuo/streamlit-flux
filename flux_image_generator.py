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
st.title("Flux 图像生成器")

# 选择模型
model = st.selectbox(
    "选择模型",
    ["black-forest-labs/flux-schnell", "black-forest-labs/flux-dev"],
    help="选择要使用的Flux模型"
)

# 输入提示词
prompt = st.text_area("输入提示词", help="描述您想生成的图像")

# 上传txt文件
uploaded_file = st.file_uploader("上传提示词文件 (每行一个提示词)", type="txt")

# 设置参数
with st.expander("高级设置"):
    seed = st.number_input("随机种子", min_value=0, help="设置随机种子以获得可重复的生成结果")
    num_outputs = st.slider("每个提示词的输出数量", min_value=1, max_value=4, value=1, help="每个提示词生成的图像数量")
    aspect_ratio = st.selectbox(
        "宽高比",
        ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        index=0,
        help="生成图像的宽高比"
    )
    output_format = st.selectbox("输出格式", ["png", "webp", "jpg"], index=0, help="输出图像的格式")
    output_quality = st.slider("输出质量", min_value=0, max_value=100, value=100, help="输出图像的质量,从0到100。100是最佳质量,0是最低质量。对于.png输出不相关")
    disable_safety_checker = st.checkbox("禁用安全检查器", help="禁用生成图像的安全检查器。此功能仅通过API可用。")

async def generate_image_async(prompt, model, input_data, timeout=45):
    try:
        prediction = await replicate.predictions.async_create(
            model=model,
            input=input_data
        )
        start_time = asyncio.get_event_loop().time()
        while prediction.status != "succeeded":
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise asyncio.TimeoutError(f"生成图像超时（{timeout}秒）")
            await asyncio.sleep(1)
            prediction = await replicate.predictions.async_get(prediction.id)
        
        logger.info(f"API 响应: {prediction.output}")
        logger.info(f"完整的 API 响应: {prediction}")

        if prediction.output:
            if isinstance(prediction.output, list):
                return prediction.output[0] if prediction.output else None
            elif isinstance(prediction.output, str):
                return prediction.output
            else:
                logger.error(f"未知的 API 响应格式: {type(prediction.output)}")
                return None
        else:
            logger.error("API 响应为空")
            return None
    except asyncio.TimeoutError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"生成图像时发生错误: {str(e)}")
        return None

def get_image_download_link(img_url, filename):
    response = requests.get(img_url)
    response.raise_for_status()
    img_data = response.content
    b64 = base64.b64encode(img_data).decode()
    file_size = len(img_data)
    logger.info(f"下载的图像大小: {file_size} 字节")
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">下载图片 ({file_size/1024:.2f} KB)</a>'

# 生成按钮
if st.button("生成图像"):
    prompts = []
    if uploaded_file:
        prompts = [line.decode("utf-8").strip() for line in uploaded_file]
    elif prompt:
        prompts = [prompt]
    
    if not prompts:
        st.error("请输入提示词或上传提示词文件")
    else:
        with st.spinner(f"正在生成 {len(prompts) * num_outputs} 张图像..."):
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
## 使用说明
- 选择要使用的Flux模型。
- 在文本框中输入描述您想要生成的图像的提示词，或上传包含多个提示词的txt文件（每行一个提示词）。
- 如果需要，可以展开“高级设置”调整其他参数。
- 点击“生成图像”按钮开始生成过程。
- 生成的图像将显示在页面上，您可以单独下载每张图片或批量下载所有图片。

注意：确保您已经设置了REPLICATE_API_TOKEN环境变量，否则API调用将失败。
""")
