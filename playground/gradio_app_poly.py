import gradio as gr
from PIL import Image, ImageDraw


def draw_quad(image, quad_coords):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    
    coords = list(map(float, quad_coords.split(',')))
    all_polygons = []
    for i in range(len(coords) // 8):
        all_polygons.append(coords[i * 8: (i + 1) * 8])
    print(f"left {coords[(i + 1) * 8:]}")
    
    for polygon in all_polygons:
        draw.polygon(polygon, outline="red", width=3)
    return image


def load_img(img_path_input):
    image = Image.open(img_path_input)
    return image


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# Quad Box Drawer")
        
        image_input = gr.Image(type="pil", label="Upload Image")
        img_path_input = gr.Textbox(label="Image Path", placeholder="absolute path")
        quad_input = gr.Textbox(label="Quad Coordinates (comma separated, 8 values)", placeholder="x1,y1,x2,y2,x3,y3,x4,y4")
        img_button = gr.Button("Load Image")
        draw_button = gr.Button("Draw Quad Box")
        image_output = gr.Image(type="pil", label="Image with Quad Box")
        img_button.click(load_img, inputs=[img_path_input], outputs=[image_input])
        draw_button.click(draw_quad, inputs=[image_input, quad_input], outputs=[image_output])

    demo.launch(share=False)
