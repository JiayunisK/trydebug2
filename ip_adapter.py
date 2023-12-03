#导入相关的包
import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio

from .utils import is_torch2_available
if is_torch2_available():
    from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):#建立一个类，继承了pytorch基础模块类torch.nn.module，返回图像嵌入经过线性变换和归一化处理后的结果
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):   #定义类的初始化函数，创建实例时调用三个参数，并提供了默认值
        super().__init__()  #调用父类,这是创建自定义 PyTorch 模块时的标准做法。
        
        self.cross_attention_dim = cross_attention_dim #将传入的参数值存储为类的实例变量
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim) #创建一个线性层并且将其储存为类的实例变量self.proj,这个线性层的输入维度是 clip_embeddings_dim，输出维度是 clip_extra_context_tokens 乘以 cross_attention_dim。
        self.norm = torch.nn.LayerNorm(cross_attention_dim) #创建一个层归一化 torch.nn.LayerNorm 并将其存储为类的实例变量 self.norm。这个层归一化的特性尺寸是 cross_attention_dim。
        
    def forward(self, image_embeds):    #定义模型的前向传播函数。这个函数接受一个参数 image_embeds
        embeds = image_embeds     #将输入 image_embeds 赋值给局部变量 embeds。
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)    #-1可以自动补全缺失的值
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens) #归一化
        return clip_extra_context_tokens


class IPAdapter:   #定义类
    
    def __init__(self, sd_pipe, image_encoder_path,pretrained_audio_model_path, ip_ckpt, device, num_tokens=4):   #定义类的初始化函数，并声明参数
        #将储存的参数值传为类的实例变量
        self.device = device           
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        
        self.pipe = sd_pipe.to(self.device) #将模型传到设备上
        print("Model moved to device:", self.device)
        self.set_ip_adapter()  #调用下文定义的 set_ip_adapter 方法来设置注意力处理器
        
        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16) #从预训练模型加载图像编码器并转移到设备上
        self.clip_image_processor = CLIPImageProcessor()     #初始化CLIP图像处理模型
        # image proj model
        self.image_proj_model = self.init_proj() #初始化图像投影模型
        
        self.load_ip_adapter()   #调用下文定义的load...方法加载模型

        self.pretrained_audio_model_path= pretrained_audio_model_path
        self.processor = Wav2Vec2Processor.from_pretrained(self.pretrained_audio_model_path)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(self.pretrained_audio_model_path)



    def init_proj(self):     #定义在类内部的方法
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)  #传参并把模型移动到设备上
        return image_proj_model  
        
    def set_ip_adapter(self):    #设置IP-Adapter
        unet = self.pipe.unet    #从self.pipe中获取属性unet
        attn_procs = {}          #初始化一个空的字典
        for name in unet.attn_processors.keys():   #对这个属性值进行迭代
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim  #基于 name 是否以 "attn1.processor" 结尾来决定 cross_attention_dim 的值。如果是，则设置为 None；否则，从 unet.config 中获取 cross_attention_dim。
            if name.startswith("mid_block"):  #如果 name 以 "mid_block" 开头，则从 unet.config.block_out_channels 中获取最后一个值。
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"): #如果 name 以 "up_blocks" 开头，则从字符串 "up_blocks." 中提取块ID，并使用此ID从反向的 block_out_channels 列表中获取相应的值。
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):   #如果 name 以 "down_blocks" 开头，与 "up_blocks" 的处理方式相同，但是直接从 block_out_channels 列表中获取值。
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:   #根据 cross_attention_dim 是否为 None 来决定如何初始化注意力处理器：
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                scale=1.0,num_tokens= self.num_tokens).to(self.device, dtype=torch.float16)    #否则，创建一个 IPAttnProcessor 的实例，并传入 hidden_size、cross_attention_dim、scale 和 num_tokens 作为参数，然后将其移至指定的设备并设置数据类型为 torch.float16。
        unet.set_attn_processor(attn_procs)   #使用上面创建的 attn_procs 字典设置 unet 的注意力处理器。
        if hasattr(self.pipe, "controlnet"):  #检查 self.pipe 是否有一个名为 controlnet 的属性：
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else: #如果有，并且它是 MultiControlNetModel 的一个实例，那么对其中的每个 controlnet，都设置一个 CNAttnProcessor 的实例。如果不是 MultiControlNetModel 的实例，只设置一个 CNAttnProcessor 的实例
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
        
    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu") #使用torch.load加载模型的检查点，权重会被加载到cpu上
        self.image_proj_model.load_state_dict(state_dict["image_proj"])  #加载权重到模型，获取image_proj子字典，这个子字典包含 image_proj_model 的权重。
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values()) #这行代码创建一个新的 ModuleList，它是一个存储模块的列表，这里我们存储的是 self.pipe.unet.attn_processors 的值。简而言之，它获取了所有的注意力处理器并将它们存储在一个 ModuleList 中。
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        
    @torch.inference_mode()  #这是一个装饰器，它告诉PyTorch在接下来的函数中运行时应该使用推断模式。推断模式意味着不会进行梯度计算，这样可以提高执行速度并减少所需的内存。
    def get_image_embeds(self, pil_image):  #定义一个名为 get_image_embeds 的方法，它接收一个参数 pil_image，这是一个PIL图像对象。
        if isinstance(pil_image, Image.Image):  #检查传入的 pil_image 是否是一个PIL图像实例。
            pil_image = [pil_image]    # 如果 pil_image 是一个PIL图像实例，则将其转换为一个包含单个图像的列表。
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values    #使用 clip_image_processor（这是该类的一个属性，可能是用于处理图像的工具）处理 pil_image 并获取其像素值。这里，处理后的图像将返回PyTorch张量。
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds   #首先，将处理后的 clip_image 转移到指定的设备上，并确保其数据类型为 torch.float16。然后，通过 self.image_encoder 进行编码以获得图像嵌入。
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)  #使用 image_proj_model 对 clip_image_embeds 进行进一步处理以获取 image_prompt_embeds。
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))  # 这里，我们使用与上述相同的模型，但输入是与 clip_image_embeds 有相同形状的零张量。这可能用于获取一个无条件的图像提示嵌入。
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():  #遍历 self.pipe.unet.attn_processors 字典的所有值。这些值可能是注意力处理器的实例。
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
#这两个方法主要与图像处理和模型的配置有关。第一个方法 get_image_embeds 获取给定PIL图像的嵌入，而第二个方法 set_scale 调整注意力处理器的比例。    
    
    
    
    def generate(
        self,
        pil_image,
        audio_path,
        
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        
        audio_features = self.get_audio_embeds(audio_path)
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            audio_features=self.pipe.audio_encode( 
                audio_path,
                device=self.device,
                num_images_per_prompt=num_samples,
            )
            audio_features = torch.cat([audio_features, image_prompt_embeds], dim=1)


        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            audio_features= audio_features,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images
