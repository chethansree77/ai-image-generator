import torch
from diffusers import StableDiffusionPipeline

print("Loading Stable Diffusion model...")
print("This may take some time on the first run.")

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

prompt = "a cute golden retriever puppy sitting in a beautiful green garden"

print("Generating image...")
image = pipe(
    prompt,
    num_inference_steps=20,
    height=512,
    width=512
).images[0]

image.save("test_local_image.png")

print("Image generated successfully!")
print("Saved as: test_local_image.png")