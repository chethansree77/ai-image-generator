import streamlit as st
import torch
from io import BytesIO
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Image Studio",
    page_icon="🎨",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    transform: scale(1.03);
}

.stDownloadButton > button {
    width: 100%;
    border-radius: 12px;
    transition: all 0.25s ease;
}

.stDownloadButton > button:hover {
    transform: scale(1.03);
}

.before-after-title {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎨 AI Image Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Create beautiful AI images locally using Stable Diffusion'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# STYLE PRESETS
# ============================================================

STYLE_PRESETS = {

    "None": "",

    "📷 Realistic Photography": """
photorealistic,
professional photography,
natural lighting,
realistic textures,
sharp focus,
realistic shadows,
high detail,
natural colors,
DSLR photography
""",

    "🎬 Cinematic": """
cinematic photography,
dramatic lighting,
movie scene,
cinematic composition,
depth of field,
volumetric lighting,
film look,
professional color grading,
high detail
""",

    "🌸 Anime": """
beautiful anime style,
Japanese animation,
detailed anime artwork,
expressive eyes,
clean line art,
vibrant colors,
detailed background,
high quality anime illustration
""",

    "🧊 3D Render": """
high quality 3D render,
3D character design,
realistic materials,
detailed textures,
studio lighting,
cinematic lighting,
octane render style,
sharp details
""",

    "🧚 Fantasy": """
epic fantasy artwork,
magical atmosphere,
beautiful fantasy environment,
mystical lighting,
glowing particles,
detailed clothing,
fantasy world,
highly detailed digital art
""",

    "🌿 Nature": """
beautiful natural environment,
lush greenery,
realistic plants,
natural sunlight,
detailed landscape,
atmospheric lighting,
photorealistic nature photography,
high detail
""",

    "👤 Portrait": """
professional portrait photography,
detailed face,
natural skin texture,
sharp eyes,
soft lighting,
realistic hair,
shallow depth of field,
professional camera photography
""",

    "🏙️ Cyberpunk": """
futuristic cyberpunk city,
neon lights,
glowing signs,
rainy streets,
futuristic buildings,
cinematic atmosphere,
dramatic lighting,
highly detailed
""",

    "🎨 Oil Painting": """
traditional oil painting,
visible brush strokes,
rich colors,
classic artistic composition,
detailed painting,
museum quality artwork,
beautiful lighting
""",

    "📸 Vintage": """
vintage photography,
retro aesthetic,
film grain,
classic camera look,
warm tones,
nostalgic atmosphere,
analog photography,
detailed image
"""
}

# ============================================================
# ASPECT RATIOS
# ============================================================

ASPECT_RATIOS = {

    "⬜ Square (1:1)": {
        "width": 512,
        "height": 512
    },

    "🧍 Portrait (4:5)": {
        "width": 512,
        "height": 640
    },

    "📱 Vertical (9:16)": {
        "width": 384,
        "height": 640
    },

    "🖼️ Landscape (4:3)": {
        "width": 512,
        "height": 384
    },

    "🌄 Widescreen (16:9)": {
        "width": 512,
        "height": 288
    }
}

# ============================================================
# SESSION STATE
# ============================================================

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

if "previous_image" not in st.session_state:
    st.session_state.previous_image = None

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipe = pipe.to("cpu")

    return pipe


# ============================================================
# IMAGE TO IMAGE MODEL
# ============================================================

@st.cache_resource
def create_img2img(_pipe):

    return StableDiffusionImg2ImgPipeline.from_pipe(
        _pipe,
        safety_checker=None,
        requires_safety_checker=False
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Generation Settings")

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    quality = st.selectbox(
        "Image Quality",
        [
            "Fast",
            "Balanced",
            "High",
            "4K"
        ],
        index=1
    )

    if quality == "Fast":
        steps = 15

    elif quality == "Balanced":
        steps = 25

    elif quality == "High":
        steps = 40

    else:
        steps = 25

    st.write(
        f"Generation Steps: **{steps}**"
    )

    # --------------------------------------------------------
    # ASPECT RATIO
    # --------------------------------------------------------

    st.subheader("📐 Aspect Ratio")

    aspect_ratio = st.selectbox(
        "Choose image shape",
        list(ASPECT_RATIOS.keys())
    )

    selected_size = ASPECT_RATIOS[
        aspect_ratio
    ]

    width = selected_size["width"]
    height = selected_size["height"]

    st.info(
        f"Output size: **{width} × {height}**"
    )

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=999999999,
        value=42,
        step=1
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎨 Text to Image",
        "🖼️ Image to Image",
        "🕘 History"
    ]
)

# ============================================================
# TEXT TO IMAGE
# ============================================================

with tab1:

    st.subheader("✨ Create an Image")

    prompt = st.text_area(
        "Enter your prompt",
        placeholder=(
            "Example: "
            "A golden retriever puppy sitting "
            "in a beautiful green garden"
        ),
        height=120
    )

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    style = st.selectbox(
        "🎨 Choose a Style",
        list(STYLE_PRESETS.keys())
    )

    style_text = STYLE_PRESETS[style]

    final_prompt = prompt.strip()

    if style_text:

        final_prompt = (
            final_prompt
            + ", "
            + style_text.strip().replace(
                "\n",
                " "
            )
        )

    # --------------------------------------------------------
    # PROMPT PREVIEW
    # --------------------------------------------------------

    if final_prompt:

        st.markdown(
            "### 📝 Prompt Preview"
        )

        st.info(final_prompt)

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "🚀 Generate Image",
        type="primary"
    ):

        if not prompt.strip():

            st.warning(
                "Please enter a prompt first."
            )

        else:

            with st.spinner(
                "Loading AI model..."
            ):

                pipe = load_model()

            progress_bar = st.progress(0)

            progress_text = st.empty()

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(seed)

            def callback(
                pipe,
                step_index,
                timestep,
                callback_kwargs
            ):

                progress = int(
                    ((step_index + 1) / steps) * 100
                )

                progress = min(
                    max(progress, 0),
                    100
                )

                progress_bar.progress(
                    progress
                )

                progress_text.write(
                    f"Generating image... "
                    f"{step_index + 1}/{steps}"
                )

                return callback_kwargs

            try:

                result = pipe(
                    prompt=final_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=generator,
                    callback_on_step_end=callback
                )

                image = result.images[0]

                # ------------------------------------------------
                # 4K OUTPUT
                # ------------------------------------------------

                if quality == "4K":

                    image = image.resize(
                        (3840, 2160),
                        Image.Resampling.LANCZOS
                    )

                # ------------------------------------------------
                # SAVE PREVIOUS IMAGE
                # ------------------------------------------------

                if (
                    st.session_state.generated_image
                    is not None
                ):

                    st.session_state.previous_image = (
                        st.session_state.generated_image.copy()
                    )

                # ------------------------------------------------
                # SAVE NEW IMAGE
                # ------------------------------------------------

                st.session_state.generated_image = (
                    image
                )

                # ------------------------------------------------
                # HISTORY
                # ------------------------------------------------

                history_item = {
                    "image": image.copy(),
                    "prompt": prompt,
                    "style": style,
                    "aspect_ratio": aspect_ratio,
                    "seed": seed
                }

                st.session_state.history.insert(
                    0,
                    history_item
                )

                st.session_state.history = (
                    st.session_state.history[:6]
                )

                progress_bar.progress(100)

                progress_text.write(
                    f"Generation complete! "
                    f"{steps}/{steps}"
                )

                st.success(
                    "🎉 Image generated successfully!"
                )

            except Exception as e:

                st.error(
                    "Image generation failed."
                )

                st.exception(e)

    # --------------------------------------------------------
    # CURRENT IMAGE
    # --------------------------------------------------------

    if (
        st.session_state.generated_image
        is not None
    ):

        st.markdown(
            "### 🖼️ Generated Image"
        )

        st.image(
            st.session_state.generated_image,
            use_container_width=True
        )

        image_buffer = BytesIO()

        st.session_state.generated_image.save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Image",
            data=image_buffer.getvalue(),
            file_name="ai_generated_image.png",
            mime="image/png",
            on_click="ignore"
        )


# ============================================================
# BEFORE / AFTER COMPARISON
# ============================================================

if (
    st.session_state.previous_image
    is not None
    and
    st.session_state.generated_image
    is not None
):

    st.divider()

    st.markdown(
        '<div class="before-after-title">'
        '🔄 Before & After'
        '</div>',
        unsafe_allow_html=True
    )

    before_col, after_col = st.columns(2)

    with before_col:

        st.markdown("### 🕐 Before")

        st.image(
            st.session_state.previous_image,
            use_container_width=True
        )

    with after_col:

        st.markdown("### ✨ After")

        st.image(
            st.session_state.generated_image,
            use_container_width=True
        )


# ============================================================
# IMAGE TO IMAGE
# ============================================================

with tab2:

    st.subheader(
        "🖼️ Transform an Existing Image"
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    transformation_strength = st.slider(
        "Transformation Strength",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )

    img2img_prompt = st.text_area(
        "Describe how you want to transform it",
        placeholder=(
            "Example: Turn this image into a "
            "cinematic scene"
        ),
        height=100
    )

    if uploaded_file:

        input_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            input_image,
            caption="Original Image",
            use_container_width=True
        )

        if st.button(
            "✨ Transform Image",
            type="primary"
        ):

            with st.spinner(
                "Loading image-to-image model..."
            ):

                pipe = load_model()

                img2img_pipe = create_img2img(
                    pipe
                )

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(seed)

            try:

                result = img2img_pipe(
                    prompt=img2img_prompt,
                    image=input_image,
                    strength=transformation_strength,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    generator=generator
                )

                new_image = result.images[0]

                st.session_state.previous_image = (
                    st.session_state.generated_image
                )

                st.session_state.generated_image = (
                    new_image
                )

                st.success(
                    "Image transformed successfully!"
                )

                st.image(
                    new_image,
                    caption="Transformed Image",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "Image transformation failed."
                )

                st.exception(e)


# ============================================================
# HISTORY
# ============================================================

with tab3:

    st.subheader(
        "🕘 Generation History"
    )

    if not st.session_state.history:

        st.info(
            "No generated images yet."
        )

    else:

        for index, item in enumerate(
            st.session_state.history
        ):

            st.markdown(
                f"### Image {index + 1}"
            )

            st.image(
                item["image"],
                use_container_width=True
            )

            st.write(
                f"**Prompt:** {item['prompt']}"
            )

            st.write(
                f"**Style:** {item['style']}"
            )

            st.write(
                f"**Aspect Ratio:** "
                f"{item['aspect_ratio']}"
            )

            st.write(
                f"**Seed:** {item['seed']}"
            )

            buffer = BytesIO()

            item["image"].save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "⬇️ Download",
                data=buffer.getvalue(),
                file_name=f"history_{index + 1}.png",
                mime="image/png",
                key=f"download_history_{index}"
            )

            st.divider()


# ============================================================
# IMAGE TOOLS
# ============================================================

if (
    st.session_state.generated_image
    is not None
):

    st.divider()

    st.subheader(
        "🛠️ Image Tools"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # ENHANCE
    # --------------------------------------------------------

    with col1:

        if st.button(
            "✨ Enhance Details"
        ):

            with st.spinner(
                "Enhancing image..."
            ):

                try:

                    pipe = load_model()

                    img2img_pipe = create_img2img(
                        pipe
                    )

                    current_image = (
                        st.session_state.generated_image
                    )

                    st.session_state.previous_image = (
                        current_image.copy()
                    )

                    result = img2img_pipe(
                        prompt=(
                            "high detail, sharp focus, "
                            "beautiful lighting, "
                            "high quality"
                        ),
                        image=current_image,
                        strength=0.15,
                        guidance_scale=7.0,
                        num_inference_steps=15
                    )

                    st.session_state.generated_image = (
                        result.images[0]
                    )

                    st.success(
                        "Details enhanced!"
                    )

                except Exception as e:

                    st.error(
                        "Enhancement failed."
                    )

                    st.exception(e)

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    with col2:

        if st.button(
            "🔍 2× Upscale"
        ):

            current_image = (
                st.session_state.generated_image
            )

            st.session_state.previous_image = (
                current_image.copy()
            )

            new_width = (
                current_image.width * 2
            )

            new_height = (
                current_image.height * 2
            )

            upscaled = current_image.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.LANCZOS
            )

            upscaled = upscaled.filter(
                ImageFilter.UnsharpMask(
                    radius=1,
                    percent=120,
                    threshold=3
                )
            )

            st.session_state.generated_image = (
                upscaled
            )

            st.success(
                "Image upscaled 2×!"
            )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    with col3:

        if st.button(
            "🔄 Reset"
        ):

            st.session_state.generated_image = None

            st.session_state.previous_image = None

            st.rerun()

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    buffer = BytesIO()

    st.session_state.generated_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        "⬇️ Download Current Image",
        data=buffer.getvalue(),
        file_name="ai_image.png",
        mime="image/png",
        on_click="ignore"
    )