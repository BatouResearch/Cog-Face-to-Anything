import json
import os
import shutil
import subprocess
import time
import torch
from typing import List, Optional, Tuple, Union
from weights import WeightsDownloadCache
import numpy as np
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from dataset_and_utils import TokenEmbeddingsHandler

CONTROL_CACHE = "./control-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_CACHE = "./feature-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):

    def load_lora(self, weights, pipe):
        print("Loading Unet LoRA")
        self.is_lora = True
        weights = str(weights)
        self.tuned_weights = weights
        local_weights_cache = self.weights_cache.ensure(weights)
        self.path = os.path.join(local_weights_cache, "lora.safetensors")
        pipe.load_lora_weights(self.path, adapter_name="lora")
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True

    def unload_lora(self, pipe):
        shutil.rmtree(self.path)
        pipe.unload_lora_weights()
        self.tuned_model = False


    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

        self.weights_cache = WeightsDownloadCache()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        self.depth_estimator = DPTForDepthEstimation.from_pretrained(FEATURE_CACHE).to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(FEATURE_CACHE)

        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            torch_dtype=torch.float16,
        )
               
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16
        )

        print("Loading SDXL Controlnet pipeline...")
        self.control_text2img_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            image_encoder=self.image_encoder,
            use_safetensors=True,
            variant="fp16",
        )
        self.control_text2img_pipe.to("cuda")

        self.control_img2img_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            controlnet=controlnet,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.control_text2img_pipe.to("cuda")
        
        self.control_img2img_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
        self.control_text2img_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")

        self.is_lora = False
        if weights or os.path.exists("./trained-model"):
            self.load_lora(weights, self.control_text2img_pipe)

        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    
    def resize_image(self, image):
        image_width, image_height = image.size
        print("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        print("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        return image, new_width, new_height
    
    def resize_to_allowed_dimensions(self, width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        height, width = image.shape[2], image.shape[3]

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        img2img: bool = Input(
            description="Use img2img pipeline, it will use the image input both as the control image and the base image.",
            default=None
        ),
        condition_scale: float = Input(
            description="The bigger this number is, the more ControlNet interferes",
            default=0.5,
            ge=0.0,
            le=2.0,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        ip_scale: float = Input(
            description="IP Adapter strength.",
            ge=0.0,
            le=1.0,
            default=0.2,
        ),
        strength: float = Input(
            description="When img2img is active, the denoising strength. 1 means total destruction of the input image.",
            default=0.7,
            ge=0.0,
            le=1.0,
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if lora_weights:
            self.load_lora(lora_weights, self.control_text2img_pipe)

        # OOMs can leave vae in bad state
        if self.control_text2img_pipe.vae.dtype == torch.float32:
            self.control_text2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")
        image = self.load_image(image)
        resized_image, width, height = self.resize_image(image)

        if (img2img) :
            print("img2img mode")
            sdxl_kwargs["image"] = resized_image
            sdxl_kwargs["control_image"] = self.get_depth_map(image)
            sdxl_kwargs["ip_adapter_image"] = image
            sdxl_kwargs["strength"] = strength
            sdxl_kwargs["controlnet_conditioning_scale"] = condition_scale
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.control_text2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["image"] = self.get_depth_map(image)
            sdxl_kwargs["ip_adapter_image"] = image
            sdxl_kwargs["controlnet_conditioning_scale"] = condition_scale
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.control_text2img_pipe

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        #pipe.set_ip_adapter_scale(ip_scale)

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output = pipe(**common_args, **sdxl_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache

        output_paths = []
        for i, nsfw in enumerate(output):
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
