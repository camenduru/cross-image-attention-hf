import sys
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image
from diffusers.training_utils import set_seed

from appearance_transfer_model import AppearanceTransferModel
from run import run_appearance_transfer
from utils.latent_utils import load_latents_or_invert_images
from utils.model_utils import get_stable_diffusion_model

sys.path.append(".")
sys.path.append("..")

from config import RunConfig

DESCRIPTION = '''
<h1 style="text-align: center;"> Cross-Image Attention for Zero-Shot Appearance Transfer </h1>
<p style="text-align: center;">
    This is a demo for our <a href="https://arxiv.org/abs/2311.03335">paper</a>: 
    ''Cross-Image Attention for Zero-Shot Appearance Transfer''.
    <br>
    Given two images depicting a source structure and a target appearance, our method generates an image merging 
    the structure of one image with the appearance of the other. 
    <br> 
    We do so in a zero-shot manner, with no optimization or model training required while supporting appearance 
    transfer across images that may differ in size and shape.
</p>
'''

pipe = get_stable_diffusion_model()


def main_pipeline(app_image_path: str,
                  struct_image_path: str,
                  domain_name: str,
                  seed: int,
                  prompt: Optional[str] = None) -> Image.Image:
    if prompt == "":
        prompt = None
    config = RunConfig(
        app_image_path=Path(app_image_path),
        struct_image_path=Path(struct_image_path),
        domain_name=domain_name,
        prompt=prompt,
        seed=seed,
        load_latents=False
    )
    print(config)
    set_seed(config.seed)
    model = AppearanceTransferModel(config=config, pipe=pipe)
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=config)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=config)
    print("Done.")
    return [images[0]]


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML('''<a href="https://huggingface.co/spaces/yuvalalaluf/cross-image-attention?duplicate=true"><img src="https://bit.ly/3gLdBN6" 
            alt="Duplicate Space"></a>''')

    with gr.Row():
        with gr.Column():
            app_image_path = gr.Image(label="Upload appearance image", type="filepath")
            struct_image_path = gr.Image(label="Upload structure image", type="filepath")
            domain_name = gr.Text(label="Domain name", max_lines=1,
                                  info="Specifies the domain the objects are coming from (e.g., 'animal', 'building', etc).")
            prompt = gr.Text(label="Prompt to use for inversion.", value='',
                             info='If this kept empty, we will use the domain name to define '
                                  'the prompt as "A photo of a <domain_name>".')
            random_seed = gr.Number(value=42, label="Random seed", precision=0)
            run_button = gr.Button('Generate')

        with gr.Column():
            result = gr.Gallery(label='Result')
            inputs = [app_image_path, struct_image_path, domain_name, random_seed, prompt]
            outputs = [result]
            run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)

    with gr.Row():
        examples = [
            ['inputs/zebra.png', 'inputs/giraffe.png', 'animal', 20, None],
            ['inputs/taj_mahal.jpg', 'inputs/duomo.png', 'building', 42, None],
            ['inputs/red_velvet_cake.jpg', 'inputs/chocolate_cake.jpg', 'cake', 42, 'A photo of cake'],
        ]
        gr.Examples(examples=examples,
                    inputs=[app_image_path, struct_image_path, domain_name, random_seed, prompt],
                    outputs=[result],
                    fn=main_pipeline,
                    cache_examples=False)

demo.queue(max_size=50).launch(share=True, debug=True)