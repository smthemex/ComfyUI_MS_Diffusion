# MS-Diffusion
本节点主要方法来源于MS-Diffusion，部分内容也来源于StoryDiffusion，感谢他们的开源！

MS-Diffusion的地址: [link](https://github.com/MS-Diffusion/MS-Diffusion)
----
我的其他comfyUI插件：
-----

1、ParlerTTS node （ParlerTTS英文的音频节点）:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node（羊驼3的节点，也兼容了其他基于羊驼3的模型）:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node（高清放大节点）：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node（零样本单图制作视频）： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node（故事绘本节点）：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node（材质、融合类节点，基于pops方法）：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node（SD官方的音频节点的简单实现） ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node（基于智普AI的api节点，涵盖智普的本地大模型）：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node（基于腾讯的CustomNet做的角度控制节点）：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node（方便玩家调用镜像抱脸下载） :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node（基于模型的图像识别） :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node（生成式PBR贴图，即将上线）:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)   
13、ComfyUI_Streamv2v_Plus node（视频转绘，能用，未打磨）:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node（基于MS-diffusion做的故事话本）:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   

1.安装
-----
  在/ComfyUI /custom_node的目录下：   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_MS_Diffusion.git
  
  ```
或者用manage 安装。。   
 
2.需求文件   
----
```
pip install -r requirements.txt
```
如果缺失模块，请单独pip install    
 
3 Need  model 
----
3.1 在线模式  
基础模型：   
模型加载菜单的前三项都为了加载基础模型，你可以选择一个方便你的。第一项是使用社区的单体SDXL模型，首次加载会下载config文件，第二项是comfyUI的diffuser菜单下的diffuser模型，第三项是标准的repo，默认是stabilityai/stable-diffusion-xl-base-1.0。   

如果你要使用其中的某一项，需要把其余二项设置为none，或者空，优先走的repo方式，所以该行注意留空。  

必须的模型： 
需要下载 "ms_adapter.bin" : [下载](https://huggingface.co/doge1516/MS-Diffusion/tree/main)    
需要下载 "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k":[下载地址](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)     
文件存放的结构如下  ：  
```
├──ComfyUI/custom_nodes/ComfyUI_MS_Diffusion/
|      ├──weights/
|             ├── photomaker-v1.bin
|             ├── ms_adapter.bin

```

3.2 离线模式 
如果有预下载的默认的扩散模型，可以不填，如果地址不在默认的C盘一类，可以在“path”一栏：填写扩散模型的绝对地址，须是“/” .  
在“laion/CLIP-ViT-bigG-14-laion2B-39B-b160k” 一栏里填写你的本地clip模型的绝对路径，使用“/”，需求的文件看下面的文件结构演示。      
以下是双角色功能，离线版的模型文件结构：   
```
├── ComfyUI/custom_nodes/ComfyUI_MS_Diffusion/
|      ├──weights/
|             ├── ms_adapter.bin
├── local_path/
|     ├──CLIP ViT bigG-14-laion2B-39B-b160k/
|             ├── config.json
|             ├── preprocessor_config.json
|             ├──pytorch_model.bin.index.json
|             ├──pytorch_model-00001-of-00002.bin
|             ├──pytorch_model-00002-of-00002.bin
|             ├──special_tokens_map.json
|             ├──tokenizer.json
|             ├──tokenizer_config.json
|             ├──vocab.json
```

3.3 双角色controlnet的模型文件示例如下，仅支持SDXL controlnet   
```
├── ComfyUI/models/diffusers/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──/controlnet-zoe-depth-sdxl-1.0  
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──TheMistoAI/MistoLine 
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──xinsir/controlnet-tile-sdxl-1.0
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
   
```
control_img图片的预处理，请使用其他节点   

4 使用说明  
---
---默认的场景prompt已经涵盖了使用的三种模式，双人（2个[ ]），单人(1个[ ])，无人([NC])，请根据你的需要来更改。  
---角色prompt的标准格式是[物体]，物体可以是man，dog，不要用姓名。  
---controlnet的图片数量必须跟场景prompt的行数一致。  
---其他功能类似storydiffusion   
--- 采样节点最下面的是开始layout_guidance（自定义角色）后才能使用的功能，必须是小于等于1.0的浮点数。


5 Example
----

img2img lora      
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/img2img.png)

img2img mode, add controlnet    
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/controlnet.png)



itation
------

MS-Diffusion
```
@misc{wang2024msdiffusion,
  title={MS-Diffusion: Multi-subject Zero-shot Image Personalization with Layout Guidance}, 
  author={X. Wang and Siming Fu and Qihan Huang and Wanggui He and Hao Jiang},
  year={2024},
  eprint={2406.07209},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
IP-Adapter
```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```

