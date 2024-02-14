from simple_lama_inpainting import SimpleLama
from PIL import Image
import gradio as gr


def inpaint_image(original_image, mask_image):
    simple_lama = SimpleLama()
    # Load images using PIL
    original_image = Image.open(original_image)
    mask_image = Image.open(mask_image).convert('L')

    # Perform inpainting
    result = simple_lama(original_image, mask_image)
    return result


# Create Gradio interface
iface = gr.Interface(fn=inpaint_image,
                     inputs=["image", "image"],
                     outputs="image",
                     inputs_layout="horizontal",
                     outputs_layout="centered",
                     title="Image Inpainting Demo",
                     description="Upload an image and a mask image. The model will inpaint the masked region of the image.",
                     examples=[['data.png', 'data_mask.png']])
iface.launch()
