import torch
from diffusers import StableDiffusionPipeline


# Model is loaded only when needed
pipe = None


def generate_image(prompt, output_path, quality="Balanced"):

    global pipe

    # Load the model
    if pipe is None:

        print("Loading Stable Diffusion model...")
        print("This may take some time on the first generation.")

        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )

        pipe = pipe.to("cpu")

        print("Stable Diffusion model loaded!")

    # Choose number of steps based on quality
    if quality == "Fast":
        steps = 10

    elif quality == "Balanced":
        steps = 15

    else:
        steps = 20

    print("Generating image...")
    print("Quality:", quality)
    print("Inference steps:", steps)

    image = pipe(
        prompt,
        negative_prompt=(
            "blurry, low quality, distorted, deformed, "
            "pixelated, noisy, unrealistic colors, "
            "bad lighting, low resolution"
        ),
        num_inference_steps=steps,
        height=512,
        width=512
    ).images[0]

    image.save(output_path)

    print("Image saved:", output_path)

    return output_path