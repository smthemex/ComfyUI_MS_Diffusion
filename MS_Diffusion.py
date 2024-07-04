# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import random
import sys
import re
from PIL import ImageFont
import yaml
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import (StableDiffusionXLPipeline, DiffusionPipeline, DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler,
                       AutoPipelineForInpainting, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, UNet2DConditionModel,
                       AutoPipelineForText2Image, StableDiffusionXLControlNetImg2ImgPipeline, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, DDPMScheduler, LCMScheduler)
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
from .msdiffusion.models.projection import Resampler
from .msdiffusion.models.model import MSAdapter
from .msdiffusion.utils import get_phrase_idx, get_eot_idx
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import folder_paths
from comfy.utils import common_upscale
from nodes import LoadImage
from .utils.utils import get_comic
from .utils.style_template import styles
STYLE_NAMES = list(styles.keys())
import diffusers
LoadImage=LoadImage()

dif_version = str(diffusers.__version__)
dif_version_int = int(dif_version.split(".")[1])
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

fonts_path = os.path.join(dir_path, "fonts")
fonts_lists = os.listdir(fonts_path)

controlnet_list=["controlnet-openpose-sdxl-1.0","controlnet-zoe-depth-sdxl-1.0","controlnet-scribble-sdxl-1.0","controlnet-tile-sdxl-1.0","controlnet-depth-sdxl-1.0","controlnet-canny-sdxl-1.0","MistoLine","sdxl-controlnet-seg"]


control_paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "config.json" in files:
                control_paths.append(os.path.relpath(root, start=search_path))
                control_paths = [z for z in control_paths if z.split("\\")[-1] in controlnet_list or z in controlnet_list]

if control_paths:
    control_paths = ["none"] + [x for x in control_paths if x]
else:
    control_paths = ["none",]

diff_paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                diff_paths.append(os.path.relpath(root, start=search_path))
if diff_paths:
    diff_paths = ["none"] + [x for x in diff_paths if x]
else:
    diff_paths = ["none",]

clip_paths = []
for search_path in folder_paths.get_folder_paths("clip_vision"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "config.json" in files:
                clip_paths.append(os.path.relpath(root, start=search_path))
if diff_paths:
    clip_paths = ["none"] + [x for x in clip_paths if x]
else:
    clip_paths = ["none",]

def pil2tensor(img):
    img_convert_to_numpy = np.array(img)  # (32, 32, 3)
    tensor = torch.tensor(img_convert_to_numpy.transpose(2, 0, 1) / 255)  # torch.Size([3, 32, 32])
    return tensor

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[style_name])
    #print(p, "test0", n)
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[style_name])
    #print(p,"test1",n)
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

loras_path = get_instance_path(os.path.join(dir_path,"config","lora.yaml"))
def get_lora_dict():
    # 打开并读取YAML文件
    with open(loras_path, 'r', encoding="UTF-8") as stream:
        try:
            # 解析YAML文件内容
            data = yaml.safe_load(stream)

            # 此时 'data' 是一个Python字典，里面包含了YAML文件的所有数据
            # print(data)
            return data

        except yaml.YAMLError as exc:
            # 如果在解析过程中发生了错误，打印异常信息
            print(exc)

datas = get_lora_dict()
lora_lightning_list = datas["lightning_xl_lora"]

scheduler_list = [
    "Euler", "Euler a", "DDIM", "DDPM", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras",
    "DPM++ SDE", "DPM++ SDE Karras", "DPM2",
    "DPM2 Karras", "DPM2 a", "DPM2 a Karras", "Heun", "LCM", "LMS", "LMS Karras", "UniPC"
]


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img


def add_pil(list, list_add, num):
    new_list = list[:num] + list_add + list[num:]
    return new_list

def tensor_to_image(tensor):
    # tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

# get fonts list
def has_parentheses(s):
    return bool(re.search(r'\(.*?\)', s))

def contains_brackets(s):
    return '[' in s or ']' in s

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = phi2narry(value)
        list_in[i] = modified_value
    return list_in

def remove_punctuation_from_strings(lst):
    pattern = r"[\W]+$"  # 匹配字符串末尾的所有非单词字符
    return [re.sub(pattern, '', s) for s in lst]

def format_punctuation_from_strings(lst):
    pattern = r"[\W]+$"  # 匹配字符串末尾的所有非单词字符
    return [re.sub(pattern, ';', s) for s in lst]

def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in

def narry_list_pil(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = tensor_to_image(value)
        list_in[i] = modified_value
    return list_in

def get_local_path(file_path, model_path,file):
    path = os.path.join(file_path, "models", file, model_path)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path


def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list

def get_scheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor_to_image(samples)
    return img

def input_size_adaptation_output(img_tensor,base_in, width, height):
    #basein=512
    if width == height:
        img_pil = nomarl_upscale(img_tensor, base_in, base_in)  # 2pil
    else:
        if min(1,width/ height)==1: #高
            r=height/base_in
            img_pil = nomarl_upscale(img_tensor, round(width/r), base_in)  # 2pil
        else: #宽
            r=width/base_in
            img_pil = nomarl_upscale(img_tensor, base_in, round(height/r))  # 2pil
        img_pil.resize((width,height))
    return img_pil

def instance_path(path, repo,file):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(file_path, path,file)
            repo = get_instance_path(model_path)
    return repo

class MSdiffusion_Model_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["none"]+folder_paths.get_filename_list("checkpoints"),),
                "diffuser_model":  (diff_paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "clip_vision_local":(clip_paths,),
                "clip_vision_repo": ("STRING", {"default": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"}),
                "controlnet_model_path": (control_paths,),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_words": ("STRING", {"default": "best quality"}),
                "scheduler": (scheduler_list,),}
        }

    RETURN_TYPES = ("MODEL","MODEL","MODEL","MODEL", "STRING",)
    RETURN_NAMES = ("pipe","ms_model","image_encoder", "image_processor","info",)
    FUNCTION = "ms_model_loader"
    CATEGORY = "MSdiffusion"

    def ms_model_loader(self, ckpt_name,diffuser_model,repo_id,clip_vision_local, clip_vision_repo, controlnet_model_path,lora, lora_scale, trigger_words,scheduler,
                           ):
        repo_id = instance_path(diffuser_model, repo_id,"diffusers")
        clip_vision=instance_path(clip_vision_local, clip_vision_repo,"clip_vision")
        if clip_vision=="none":
            raise "need clip_vision"
        scheduler_choice = get_scheduler(scheduler)

        if lora != "none":
            lora_path = folder_paths.get_full_path("loras", lora)
            lora_path = get_instance_path(lora_path)
        else:
            lora_path = ""
        if "/" in lora:
            lora = lora.split("/")[-1]
        if "\\" in lora:
            lora = lora.split("\\")[-1]

        # load SDXL pipeline
        original_config_file = os.path.join(dir_path, 'config', 'sd_xl_base.yaml')
        original_config_file = get_instance_path(original_config_file)

        if controlnet_model_path != "none": #启用control
            controlnet_model_path = get_instance_path(get_local_path(file_path, controlnet_model_path,"diffusers"))
            controlnet = ControlNetModel.from_pretrained(controlnet_model_path, variant="fp16",
                                                         use_safetensors=True,
                                                         torch_dtype=torch.float16).to(device)
            if repo_id == "none":
                if ckpt_name == "none":
                    raise "need choice checkpoints or fill repo or choice diffuser model"
                ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                repo_id = get_instance_path(ckpt_path)
                if dif_version_int >= 28:
                    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                        repo_id, original_config=original_config_file, controlnet=controlnet,
                        torch_dtype=torch.float16,
                        add_watermarker=False, ).to(device)
                else:
                    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                        repo_id, original_config_file=original_config_file, controlnet=controlnet,
                        torch_dtype=torch.float16,
                        add_watermarker=False,
                    ).to(device)
            else:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    repo_id, torch_dtype=torch.float16, add_watermarker=False, controlnet=controlnet,
                ).to(device)
        else: # 非controlnet模式
            if repo_id == "none":
                if ckpt_name == "none":
                    raise "need choice checkpoints or fill repo or choice diffuser model"
                ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                repo_id = get_instance_path(ckpt_path)
                if dif_version_int >= 28:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        repo_id, original_config=original_config_file, torch_dtype=torch.float16,
                        add_watermarker=False, ).to(device)
                else:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        repo_id, original_config_file=original_config_file, torch_dtype=torch.float16,
                        add_watermarker=False,
                    ).to(device)
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    repo_id, torch_dtype=torch.float16, add_watermarker=False,
                ).to(device)

        if lora != "none":
            if lora in lora_lightning_list:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
            else:
                pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                pipe.fuse_lora(adapter_names=[trigger_words, ], lora_scale=lora_scale)

        pipe.scheduler = scheduler_choice.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.enable_vae_slicing()
        if device != "mps":
            pipe.enable_model_cpu_offload()
        ms_dir = os.path.join(dir_path, "weights")
        photomaker_local_path = os.path.join(ms_dir, "ms_adapter.bin")
        if not os.path.exists(photomaker_local_path):
            ms_path = hf_hub_download(
                repo_id="doge1516/MS-Diffusion",
                filename="ms_adapter.bin",
                repo_type="model",
                local_dir=ms_dir,
            )
        else:
            ms_path = photomaker_local_path
        ms_ckpt = get_instance_path(ms_path)
        image_processor = CLIPImageProcessor()
        #print(type(image_processor))
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_vision).to(device, dtype=torch.float16)
        # print(type(image_encoder))
        image_encoder_projection_dim = image_encoder.config.projection_dim
        num_tokens = 16
        image_encoder_type = "clip"
        image_proj_type = "resampler"
        latent_init_mode = "grounding"
        # latent_init_mode = "random"
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=pipe.unet.config.cross_attention_dim,
            ff_mult=4,
            latent_init_mode=latent_init_mode,
            phrase_embeddings_dim=pipe.text_encoder.config.projection_dim,
        ).to(device, dtype=torch.float16)
        ms_model = MSAdapter(pipe.unet, image_proj_model, ckpt_path=ms_ckpt, device=device, num_tokens=num_tokens)
        ms_model.to(device, dtype=torch.float16)
        torch.cuda.empty_cache()
        info = str(";".join(
            [lora, trigger_words,controlnet_model_path,image_encoder_type,image_proj_type]))
        return (pipe,ms_model,image_encoder,image_processor, info,)
    

class MSdiffusion_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "pipe": ("MODEL",),
                "ms_model": ("MODEL",),
                "image_encoder": ("MODEL",),
                "image_processor": ("MODEL",),
                "character_prompt": ("STRING", {"multiline": True,
                                                "default": "[woman] wearing a white T-shirt, blue loose hair.\n"
                                                           "[man] wearing a suit,black hair."}),
                "scene_prompts": ("STRING", {"multiline": True,
                                             "default": "[woman] and [man] have breakfast,medium shot;\n[woman] go to company;\n[man] have lunch,medium shot;\n[NC] a living room."}),
                "split_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "bad anatomy, bad hands, missing fingers, extra fingers, "
                                                          "three hands, three legs, bad arms, missing legs, "
                                                          "missing arms, poorly drawn face, bad face, fused face, "
                                                          "cloned face, three crus, fused feet, fused thigh, "
                                                          "extra crus, ugly fingers, horn,"
                                                          "amputation, disconnected limbs"}),
                "img_style": (
                    ["No_style", "Realistic", "Japanese_Anime", "Digital_Oil_Painting", "Pixar_Disney_Character",
                     "Photographic", "Comic_book",
                     "Line_art", "Black_and_White_Film_Noir", "Isometric_Rooms"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "role_scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "mask_threshold": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "start_step": ("INT", {"default": 5, "min": 1, "max": 1024}),
                "controlnet_scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "display": "number"}),
                "layout_guidance":("BOOLEAN", {"default": False},),
                "guidance_list": ("STRING", {"multiline": True,"default": "0., 0.25, 0.4, 0.75;0.6, 0.25, 1., 0.75"}),
            },
            "optional": {"control_image": ("IMAGE",),}
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "prompt_array",)
    FUNCTION = "ms_sampler"
    CATEGORY = "MSdiffusion"

    def main_normal(self,prompts_dual,pipe,phrases,ms_model,input_images,num_samples,steps,seed,negative_prompt,role_scale,image_encoder,cfg,image_processor,
                    boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,height,width,img_style):
        image_ouput = []
        for i, prompt in enumerate(prompts_dual):
            prompt = apply_style_positive(img_style, prompt)
            #print(prompt)
            # used to get the attention map, return zero if the phrase is not in the prompt
            phrase_idxes = [self.get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
            eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
            # print(phrase_idxes, eot_idxes)
            images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples,
                                       num_inference_steps=steps,
                                       seed=seed,
                                       prompt=[prompt], negative_prompt=negative_prompt, scale=role_scale,
                                       image_encoder=image_encoder, guidance_scale=cfg,
                                       image_processor=image_processor, boxes=boxes,
                                       mask_threshold=mask_threshold,
                                       start_step=start_step,
                                       image_proj_type=image_proj_type,
                                       image_encoder_type=image_encoder_type,
                                       phrases=phrases,
                                       drop_grounding_tokens=drop_grounding_tokens,
                                       phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=height,
                                       width=width)
            image_ouput += images
        return image_ouput

    def main_control(self,prompts_dual,control_img_list,width,height,pipe,phrases,ms_model,input_images,num_samples,steps,seed,negative_prompt,role_scale,image_encoder,cfg,
                     image_processor,boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,controlnet_scale,img_style):
        j = 0
        image_ouput=[]
        for i, prompt in enumerate(prompts_dual):
            control_image = control_img_list[j]
            j += 1
            prompt=apply_style_positive(img_style,prompt)
            #print(prompt)
            # used to get the attention map, return zero if the phrase is not in the prompt
            phrase_idxes = [self.get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
            eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
            # print(phrase_idxes, eot_idxes)
            images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples,
                                       num_inference_steps=steps,
                                       seed=seed,
                                       prompt=[prompt], negative_prompt=negative_prompt, scale=role_scale,
                                       image_encoder=image_encoder, guidance_scale=cfg,
                                       image_processor=image_processor, boxes=boxes,
                                       mask_threshold=mask_threshold,
                                       start_step=start_step,
                                       image_proj_type=image_proj_type,
                                       image_encoder_type=image_encoder_type,
                                       phrases=phrases,
                                       drop_grounding_tokens=drop_grounding_tokens,
                                       phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=height,
                                       width=width,
                                       image=control_image, ontrolnet_conditioning_scale=controlnet_scale)

            image_ouput += images
        return image_ouput

    def get_phrases_idx(self,tokenizer, phrases, prompt):
        res = []
        phrase_cnt = {}
        for phrase in phrases:
            if phrase in phrase_cnt:
                cur_cnt = phrase_cnt[phrase]
                phrase_cnt[phrase] += 1
            else:
                cur_cnt = 0
                phrase_cnt[phrase] = 1
            res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
        return res

    def add_lora(self,trigger_words,lora,prompts_list):
        # 添加Lora trigger
        add_trigger_words = "," + trigger_words + " " + "style" + ";"
        prompts_list = remove_punctuation_from_strings(prompts_list)
        if lora not in lora_lightning_list:  # 加速lora不需要trigger
            prompts_list = [item + add_trigger_words for item in prompts_list]
        return prompts_list
    
   
    def ms_sampler(self,image,info,pipe,ms_model,image_encoder,image_processor, character_prompt, scene_prompts,split_prompt,negative_prompt,img_style,seed, steps,
                  cfg,role_scale, mask_threshold, start_step,controlnet_scale,width,height,layout_guidance,guidance_list,**kwargs):
        lora, trigger_words,controlnet_model_path,image_encoder_type,image_proj_type  = info.split(";")
    
        guidance_list=guidance_list.strip().split(";")
        box_a = guidance_list[0].split(",")
        box_b = guidance_list[1].split(",")
        box_a_float=[float(x) for x in box_a]
        box_b_float = [float(x) for x in box_b]
        box_add=[box_a_float,box_b_float]
        #print(box_a_float, box_b_float,box_add)

        #处理prompt
        if split_prompt:
            scene_prompts.replace("\n", "").replace(split_prompt, ";\n").strip()
            character_prompt.replace("\n", "").replace(split_prompt, ";\n").strip()
        else:
            scene_prompts.strip()
            character_prompt.strip()
            if "\n" not in scene_prompts:
                scene_prompts.replace(";", ";\n").strip()
            if "\n" in character_prompt:
                if character_prompt.count("\n") > 1:
                    character_prompt.replace("\n", "").replace("[", "\n[").strip()
                    if character_prompt.count("\n") > 1:
                        character_prompt.replace("\n", "").replace("[", "\n[", 2).strip()  # 多行角色在这里强行转为双角色

        # 从角色列表获取角色方括号信息
        char_origin_f = character_prompt.splitlines()
        char_describe = [char.replace("]", " ").replace("[", " ") for char in char_origin_f] #del character's []
        char_origin = [char.split("]")[0] + "]" for char in char_origin_f] # [[role1],[role2]]
        role_str=[x.replace("]", "").replace("[", "") for x in char_origin] # [role1,role2]

        prompts_origin = scene_prompts.splitlines()
        control_check=prompts_origin
        prompts_origin_unuse, negative_prompt = apply_style(img_style, prompts_origin, negative_prompt) #获取  style p prompt

        num_samples = 1 #批次
        #image = kwargs["image"]
        d1, _, _, _ = image.size()

        if d1 == 1:
            prompt_single_NC = [prompt for prompt in prompts_origin if "[NC]" in prompt]  # NC场景列表,单人,
            prompt_single_NC=[prompt.replace("[NC]","") for prompt in prompt_single_NC ]  # 去除符号
            index_single_NC = [index for index, prompt in enumerate(prompts_origin) if "[NC]" in prompt]  # NC场景index
            prompts_origin = [prompt for prompt in prompts_origin if role_str[0] in prompt]  # 排除 nc
            input_images = [nomarl_upscale(image, width, height)]
            torch.cuda.empty_cache()
            # 加入角色描述
            prompt_c = char_origin_f[0].replace("]", " ").replace("[", " ")
            role = char_origin[0].replace("]", "").replace("[", "")
            prompts_dual = [x.replace(char_origin[0],prompt_c) for x in prompts_origin]
            prompts_dual = [x.replace("]", "").replace("[", "") for x in prompts_dual]

            if layout_guidance:
                 boxes=random.choice(box_add)
                 boxes=[[boxes]] #随机选1个自定义的
                #boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # 1 role
            else:
                boxes = [[[0., 0., 0., 0.]]]  # 1 role
            phrases = [[role]]
            drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens

            if len(char_origin_f) != 1:  # 单人
                raise "need one role  prompts "
            if controlnet_model_path != "none":
                control_image = kwargs["control_image"]
                d1, _, _, _ = control_image.size()
                if d1!=len(control_check):
                    raise "The number of scene prompts that must match the control_image"
                control_img = list(torch.chunk(control_image, chunks=d1)) #tensor
                control_img = [input_size_adaptation_output(control_image, 768, width, height) for control_image in control_img ] #pil
                control_img_list=[img for index, img in enumerate(control_img) if  index not in index_single_NC ] #del NC
                if lora != "none":
                    prompts_dual=self.add_lora(trigger_words, lora, prompts_dual)
                image_main=self.main_control(prompts_dual,control_img_list,width,height,pipe,phrases,ms_model,input_images,num_samples,steps,seed,negative_prompt,role_scale,image_encoder,cfg,
                     image_processor,boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,controlnet_scale,img_style)

                if prompt_single_NC:
                    if lora != "none":
                        prompt_single_NC = self.add_lora(trigger_words, lora, prompt_single_NC)
                    control_img_list_NC= [img for index, img in enumerate(control_img) if  index in index_single_NC ] # NC
                    input_images= [Image.new('RGB', (width, height), (255, 255, 255))]
                    image_nc = self.main_control(prompt_single_NC, control_img_list_NC, width, height, pipe, phrases,
                                                   ms_model, input_images, num_samples, steps, seed, negative_prompt,
                                                   role_scale, image_encoder, cfg,
                                                   image_processor, boxes, mask_threshold, start_step, image_proj_type,
                                                   image_encoder_type, drop_grounding_tokens, controlnet_scale,img_style)
                    jj = 0
                    for i in index_single_NC:  # 重新将NC场景插入原序列
                        img = image_nc[jj]
                        image_main.insert(int(i), img)
                        jj += 1
                    image_list = narry_list(image_main)
                else:
                    image_list = narry_list(image_main)

            else: #no controlnet
                if lora != "none":
                    prompts_dual=self.add_lora(trigger_words, lora, prompts_dual)
                image_main=self.main_normal( prompts_dual, pipe, phrases, ms_model, input_images, num_samples, steps, seed,
                            negative_prompt, role_scale, image_encoder, cfg, image_processor,
                            boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                            drop_grounding_tokens, height, width,img_style)

                if prompt_single_NC:
                    if lora != "none":
                        prompt_single_NC = self.add_lora(trigger_words, lora, prompt_single_NC)
                    input_images= [Image.new('RGB', (width, height), (255, 255, 255))]
                    image_nc = self.main_normal( prompt_single_NC, pipe, phrases, ms_model, input_images, num_samples, steps, seed,
                            negative_prompt, role_scale, image_encoder, cfg, image_processor,
                            boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                            drop_grounding_tokens, height, width,img_style)
                    jj = 0
                    for i in index_single_NC:  # 重新将NC场景插入原序列
                        img = image_nc[jj]
                        image_main.insert(int(i), img)
                        jj += 1

                image_list = narry_list(image_main)
        else:
            prompt_dual_NC = [prompt for prompt in prompts_origin if
                              "[NC]" in prompt]  # NC场景列表，双人,
            prompt_dual_NC=[prompt.replace("[NC]","") for prompt in prompt_dual_NC]  # 去除符号
            index_dual_NC = [index for index, prompt in enumerate(prompts_origin) if
                             "[NC]" in prompt]  # NC场景index
            prompts_origin_role_a = [prompt for prompt in  prompts_origin if
                                     char_origin[0] in prompt and  char_origin[1] not in prompt]  # 排除 nc,仅单人A
            prompts_origin_role_a_index = [index for index, prompt in enumerate( prompts_origin) if
                                           char_origin[0] in prompt and  char_origin[1] not in prompt]
            prompts_origin_role_b = [prompt for prompt in  prompts_origin if
                                     char_origin[1] in prompt and char_origin[0] not in prompt]  # 排除 nc,仅单人B
            prompts_origin_role_b_index = [index for index, prompt in enumerate(prompts_origin) if
                                           char_origin[1] in prompt and  char_origin[0] not in prompt]
            prompts_origin = [prompt for prompt in prompts_origin if
                              (char_origin[0] in prompt and char_origin[1] in prompt)]  # 排除单人，nc,仅双人
            #print(prompt_dual_NC, "prompt_dual_NC",index_dual_NC, "index_dual_NC",prompts_origin_role_a ,"prompts_origin_role_a ",prompts_origin_role_a_index,"prompts_origin_role_a_index",prompts_origin_role_b,"prompts_origin_role_b",prompts_origin,"prompts_origin")
            img_list = list(torch.chunk(image, chunks=d1))
            input_images = [nomarl_upscale(img, width, height) for img in img_list]

            input_images_a =[input_images[0]]
            input_images_b = [input_images[1]]

            # get role name and describe
            role_a = role_str[0]
            char_describe_a= char_describe[0]
            role_b =role_str[1]
            char_describe_b = char_describe[1]

            # 双角色场景列表 prompt加描述
            prompts_dual = [item.replace(char_origin[0],char_describe_a).replace(char_origin[1],char_describe_b) for item in prompts_origin]

            torch.cuda.empty_cache()
            if layout_guidance:
                boxes=[box_add]
                #boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # man+women
            else:
                boxes = [[[0., 0., 0., 0.], [0., 0., 0., 0.]]]  # used if you want no layout guidance
            phrases = [[role_a, role_b]]
            drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens
            image_new=Image.new('RGB', (width, height), (255, 255, 255))
            if controlnet_model_path != "none":
                control_image = kwargs["control_image"]
                d1, _, _, _ = control_image.size()
                if d1 != len(control_check):
                    raise "The number of scene prompts that must match the control_image"
                control_img = list(torch.chunk(control_image, chunks=d1))  # tensor
                control_img = [input_size_adaptation_output(control_image, 768, width, height) for control_image in
                               control_img]  # pil
                control_img_list = [img for index, img in enumerate(control_img) if
                                     index  not in index_dual_NC]  # del NC
                if lora != "none":
                    prompts_dual=self.add_lora(trigger_words, lora, prompts_dual)

                image_main=self.main_control(prompts_dual,control_img_list,width,height,pipe,phrases,ms_model,input_images,num_samples,steps,seed,negative_prompt,role_scale,image_encoder,cfg,
                     image_processor,boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,controlnet_scale,img_style)
                if prompt_dual_NC:
                    if lora != "none":
                        prompt_dual_NC = self.add_lora(trigger_words, lora, prompt_dual_NC)
                    control_img_list_NC = [img for index, img in enumerate(control_img) if
                                           index in index_dual_NC]  # NC
                    input_images = [image_new,image_new]
                    image_nc = self.main_control(prompt_dual_NC, control_img_list_NC, width, height, pipe, phrases,
                                                 ms_model, input_images, num_samples, steps, seed, negative_prompt,
                                                 role_scale, image_encoder, cfg,
                                                 image_processor, boxes, mask_threshold, start_step, image_proj_type,
                                                 image_encoder_type, drop_grounding_tokens, controlnet_scale,img_style)
                    jj = 0
                    for i in index_dual_NC:  # 重新将NC场景插入原序列
                        img = image_nc[jj]
                        image_main.insert(int(i), img)
                        jj += 1

                if prompts_origin_role_a:
                    if lora != "none":
                        prompts_origin_role_a = self.add_lora(trigger_words, lora, prompts_origin_role_a)
                    prompts_origin_role_a =[x.replace(char_origin[0],char_describe_a) for x in prompts_origin_role_a] # 加入角色描述
                    control_img_list = [img for index, img in enumerate(control_img) if
                                        index in prompts_origin_role_a_index]  # role a
                    if layout_guidance:
                        boxes = random.choice(box_add)
                        boxes = [[boxes]]  # 随机选1个自定义的
                        #boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # 1 role
                    else:
                        boxes = [[[0., 0., 0., 0.]]]  # 1 role
                    phrases = [[role_a]]
                    input_images = input_images_a
                    image_dual_a = self.main_control(prompts_origin_role_a, control_img_list, width, height, pipe, phrases,
                                                   ms_model, input_images, num_samples, steps, seed, negative_prompt,
                                                   role_scale, image_encoder, cfg,
                                                   image_processor, boxes, mask_threshold, start_step, image_proj_type,
                                                   image_encoder_type, drop_grounding_tokens, controlnet_scale,img_style)
                    a = 0
                    for i in prompts_origin_role_a_index:  # 重新将NC场景插入原序列
                        img = image_dual_a[a]
                        image_main.insert(int(i), img)
                        a += 1

                if prompts_origin_role_b:
                    if lora != "none":
                        prompts_origin_role_b = self.add_lora(trigger_words, lora, prompts_origin_role_b)
                    prompts_origin_role_b = [x.replace(char_origin[1], char_describe_b) for x in
                                             prompts_origin_role_b]  # 加入角色描述
                    control_img_list = [img for index, img in enumerate(control_img) if
                                        index in prompts_origin_role_b_index]  # role b
                    if layout_guidance:
                        boxes = random.choice(box_add)
                        boxes = [[boxes]]  # 随机选1个自定义的
                        #boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # 1 role
                    else:
                        boxes = [[[0., 0., 0., 0.]]]  # 1 role
                    phrases = [[role_b]]
                    input_images = input_images_b
                    image_dual_b = self.main_control(prompts_origin_role_b, control_img_list, width, height, pipe, phrases,
                                                   ms_model, input_images, num_samples, steps, seed, negative_prompt,
                                                   role_scale, image_encoder, cfg,
                                                   image_processor, boxes, mask_threshold, start_step, image_proj_type,
                                                   image_encoder_type, drop_grounding_tokens, controlnet_scale,img_style)
                    b = 0
                    for i in prompts_origin_role_b_index:  # 重新将NC场景插入原序列
                        img = image_dual_b[b]
                        image_main.insert(int(i), img)
                        b += 1

                image_list = narry_list(image_main)

            else:
                if lora != "none":
                    prompts_dual = self.add_lora(trigger_words, lora, prompts_dual)
                image_main = self.main_normal(prompts_dual, pipe, phrases, ms_model, input_images, num_samples, steps,
                                              seed,
                                              negative_prompt, role_scale, image_encoder, cfg, image_processor,
                                              boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                                              drop_grounding_tokens, height, width,img_style)
                if prompt_dual_NC:
                    if lora != "none":
                        prompt_dual_NC = self.add_lora(trigger_words, lora, prompt_dual_NC)
                    input_images = [image_new,image_new]
                    image_nc = self.main_normal( prompt_dual_NC, pipe, phrases, ms_model, input_images, num_samples,
                                                steps, seed,
                                                negative_prompt, role_scale, image_encoder, cfg, image_processor,
                                                boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                                                drop_grounding_tokens, height, width,img_style)
                    jj = 0
                    for i in index_dual_NC:  # 重新将NC场景插入原序列
                        img = image_nc[jj]
                        image_main.insert(int(i), img)
                        jj += 1

                if prompts_origin_role_a:
                    if lora != "none":
                        prompts_origin_role_a = self.add_lora(trigger_words, lora, prompts_origin_role_a)
                    prompts_origin_role_a = [x.replace(char_origin[0], char_describe_a) for x in
                                             prompts_origin_role_a]  # 加入角色描述
                    if layout_guidance:
                        boxes = random.choice(box_add)
                        boxes = [[boxes]]  # 随机选1个自定义的
                        #boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # 1 role
                    else:
                        boxes = [[[0., 0., 0., 0.]]]  # 1 role
                    phrases = [[role_a]]
                    input_images = input_images_a
                    image_role_a = self.main_normal(prompts_origin_role_a, pipe, phrases, ms_model, input_images, num_samples,
                                                steps, seed,
                                                negative_prompt, role_scale, image_encoder, cfg, image_processor,
                                                boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                                                drop_grounding_tokens, height, width,img_style)
                    jj = 0
                    for i in prompts_origin_role_a_index:  # 重新将NC场景插入原序列
                        img = image_role_a[jj]
                        image_main.insert(int(i), img)
                        jj += 1

                if prompts_origin_role_b:
                    if lora != "none":
                        prompts_origin_role_b = self.add_lora(trigger_words, lora, prompts_origin_role_b)
                    prompts_origin_role_b = [x.replace(char_origin[1], char_describe_b) for x in
                                             prompts_origin_role_b]  # 加入角色描述
                    if layout_guidance:
                        boxes = random.choice(box_add)
                        boxes = [[boxes]]  # 随机选1个自定义的
                        #boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # 1 role
                    else:
                        boxes = [[[0., 0., 0., 0.]]]  # 1 role
                    phrases = [[role_b]]
                    input_images = input_images_b
                    image_role_b = self.main_normal(prompts_origin_role_b, pipe, phrases, ms_model, input_images, num_samples,
                                                steps, seed,
                                                negative_prompt, role_scale, image_encoder, cfg, image_processor,
                                                boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                                                drop_grounding_tokens, height, width,img_style)
                    jj = 0
                    for i in prompts_origin_role_b_index:  # 重新将NC场景插入原序列
                        img = image_role_b[jj]
                        image_main.insert(int(i), img)
                        jj += 1
                image_list = narry_list(image_main)

        image = torch.from_numpy(np.fromiter(image_list, np.dtype((np.float32, (height, width, 3)))))
        torch.cuda.empty_cache()
        return (image, scene_prompts,)

class MS_Comic_Type:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "scene_prompts": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                             "fonts_list": (fonts_lists,),
                             "text_size": ("INT", {"default": 40, "min": 1, "max": 100}),
                             "comic_type": (["Four_Pannel", "Classic_Comic_Style"],),
                             "split_lines": ("STRING", {"default": "；"}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    ETURN_NAMES = ("image",)
    FUNCTION = "comic_gen"
    CATEGORY = "MSdiffusion"

    def comic_gen(self, image, scene_prompts, fonts_list, text_size, comic_type, split_lines):
        result = [item for index, item in enumerate(image)]
        total_results = narry_list_pil(result)
        font_choice = os.path.join(dir_path, "fonts", fonts_list)
        captions = scene_prompts.splitlines()
        if len(captions) > 1:
            # captions = [caption.replace("(", "").replace(")", "") if "(" or ")" in caption else caption
            #             for caption in captions]  # del ()
            captions = [caption.replace("[NC]", "") for caption in captions]
            captions = [caption.replace("]", "").replace("[", "") for caption in captions]
            # captions = [
            #     caption.split("#")[-1] if "#" in caption else caption
            #     for caption in captions
            # ]
        else:#trans
            prompt_array = scene_prompts.replace(split_lines, "\n")
            captions = prompt_array.splitlines()
        font = ImageFont.truetype(font_choice, text_size)
        images = (
                get_comic(total_results, comic_type, captions=captions, font=font)
                + total_results
        )
        images = phi2narry(images[0])
        return (images,)

NODE_CLASS_MAPPINGS = {
    "MSdiffusion_Model_Loader": MSdiffusion_Model_Loader,
    "MSdiffusion_Sampler": MSdiffusion_Sampler,
    "MS_Comic_Type":MS_Comic_Type

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSdiffusion_Model_Loader": "MSdiffusion_Model_Loader",
    "MSdiffusion_Sampler": "MSdiffusion_Sampler",
    "MS_Comic_Type":"MS_Comic_Type"
}
