import gradio as gr


def echo_image(image):
    return image


demo = gr.Interface(
    fn=echo_image,
    inputs=gr.Image(type="pil", label="상품 사진 업로드"),
    outputs=gr.Image(label="업로드된 이미지"),
    title="AI 쇼핑 어시스턴트",
)


if __name__ == "__main__":
    demo.launch()
