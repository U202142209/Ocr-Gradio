import os
import traceback
from datetime import datetime
from io import BytesIO

import gradio as gr
import requests
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# # 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 初始化 OCR 识别管道
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')


# 获取当前的系统时间,记录日志使用
def getCurrentTime(is_filename=False):
    if is_filename:
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Recognize from URL function
def ocr_from_url(url):
    if not str(url).startswith("http"):
        gr.Warning("请输入有效的URL")
        return
    # 记录日志,将入户输入的连接和当前的实践记录到 logs/log.txt
    with open("logs/log.txt", "a") as f:
        f.write(f"{getCurrentTime()}  URL:{url}\n")
    try:
        gr.Info("正在识别图片...")
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        result = ocr_recognition(img)
        gr.Info("识别完成")
        return "# 识别结果 :" + result['text'][0], img
    except Exception as error:
        err_str = f"# 出错了 :{str(error)},请检查输入的网址 [{url}]({url}) 是否有效 :![图片]({url})"
        traceback.print_exc()
        return err_str, None


def ocr_from_file(file):
    if not file:
        gr.Warning("请上传有效的图片")
        return
    try:
        gr.Info("正在识别图片...")
        img = Image.open(file.name)
        # 将图片保存起来,记录日志
        img.save(f"images/{getCurrentTime(is_filename=True)}.jpg")
        # 开始识别
        result = ocr_recognition(img)
        gr.Info("识别完成")
        return "# 识别结果 :" + result['text'][0], img
    except Exception as error:
        traceback.print_exc()
        return "# 出错了 :" + str(error), None


def handleExample(x):
    # return gr.Textbox(value=x, label="URL")
    return ocr_from_url(x)


# 创建 Gradio Blocks 界面
with gr.Blocks() as demo:
    with gr.Row():
        # 设置标题
        gr.Markdown(
            """
            # OCR Recognition
            **说明**:这是一个[OCR](https://www.modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo)识别模型，可以识别图片中的文字，支持**中文、英文、数字**等多种语言。

            **使用**:支持两种使用方式
            - 方式一：输入图片URL地址
            - 方式二：上传图片文件
            """)
    # 输入框
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(label="方式一:输入URL", placeholder='请输入图片的URL地址')
            # 添加示例 URL
            with gr.Accordion("Examples|示例", open=True):
                with gr.Row():
                    example_urls = [{
                        "text": "示例1.识别验证码",
                        "value": "http://blog.wobushidalao.top/random_img/"
                    }, {
                        "text": "示例2.识别中文",
                        "value": "http://blog.wobushidalao.top/static/img/decorate/logo.jpg"
                    }, {
                        "text": "示例3.识别英文",
                        "value": "http://modelcube.cn/assets/logolarge.pic-1df83cdb.jpg"
                    }, ]
                    for item in example_urls:
                        gr.Button(
                            value=item["text"], size="sm",
                        ).click(
                            fn=lambda x=item["value"]: gr.Textbox(value=x, label="URL"),
                            inputs=[],
                            outputs=[url_input]
                        )
            url_button = gr.Button("Recognize from URL", variant="primary")

            file_input = gr.File(label="方式二:上传图片")
            # file_input = gr.Image(label="方式二:上传图片", type="filepath")
            file_button = gr.Button("Recognize from File", variant="primary")

        with gr.Column():
            text_output = gr.Markdown(label="Recognized Text")
            img_output = gr.Image(label="Input Image", show_label=False)

    # 绑定按钮点击事件
    url_button.click(ocr_from_url, inputs=url_input, outputs=[text_output, img_output])
    file_button.click(ocr_from_file, inputs=file_input, outputs=[text_output, img_output])

os.makedirs("logs", exist_ok=True)  # 创建用于保存用户输入日志的目录
os.makedirs("images", exist_ok=True)  # 记录用户上传的图片

# 启动 Gradio 应用
demo.queue(
    default_concurrency_limit=10
).launch(
    server_port=7594
)
