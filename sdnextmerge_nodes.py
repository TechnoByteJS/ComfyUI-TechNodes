import torch
import folder_paths
from typing import Dict, Tuple, List
from collections import OrderedDict
import ast

import comfy.sd
import comfy.utils
import comfy.model_detection

from .merge import *

mbw_presets = ([
				"none",
				"GRAD_V",
				"GRAD_A",
				"FLAT_25",
				"FLAT_75",
				"WRAP08",
				"WRAP12",
				"WRAP14",
				"WRAP16",
				"MID12_50",
				"OUT07",
				"OUT12",
				"OUT12_5",
				"RING08_SOFT",
				"RING08_5",
				"RING10_5",
				"RING10_3",
				"SMOOTHSTEP",
				"REVERSE_SMOOTHSTEP",
				"2SMOOTHSTEP",
				"2R_SMOOTHSTEP",
				"3SMOOTHSTEP",
				"3R_SMOOTHSTEP",
				"4SMOOTHSTEP",
				"4R_SMOOTHSTEP",
				"HALF_SMOOTHSTEP",
				"HALF_R_SMOOTHSTEP",
				"ONE_THIRD_SMOOTHSTEP",
				"ONE_THIRD_R_SMOOTHSTEP",
				"ONE_FOURTH_SMOOTHSTEP",
				"ONE_FOURTH_R_SMOOTHSTEP",
				"COSINE",
				"REVERSE_COSINE",
				"CUBIC_HERMITE",
				"REVERSE_CUBIC_HERMITE",
				"FAKE_REVERSE_CUBIC_HERMITE",
				"LOW_OFFSET_CUBIC_HERMITE",
				"ALL_A",
				"ALL_B",
			], {"default": "none"})

class SDNextMerge:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"optional": {
				"optional_model_a": ["MODEL"],
				"optional_clip_a": ["CLIP"],
	
				"optional_model_b": ["MODEL"],
				"optional_clip_b": ["CLIP"],
	
				"optional_model_c": ["MODEL"],
				"optional_clip_c": ["CLIP"],
	
				"optional_mbw_layers_alpha": ["MBW_LAYERS"],
			},
			"required": {
				"model_a": (["none"] + folder_paths.get_filename_list("checkpoints"), {"multiline": False}),
				"model_b": (["none"] + folder_paths.get_filename_list("checkpoints"), {"multiline": False}),
				"model_c": (["none"] + folder_paths.get_filename_list("checkpoints"), {"multiline": False}),
				"merge_mode": ([
					"weighted_sum",
					"weighted_subtraction",
					"tensor_sum",
					"add_difference",
					"sum_twice",
					"triple_sum",
					"euclidean_add_difference",
					"multiply_difference",
					"top_k_tensor_sum",
					"similarity_add_difference",
					"distribution_crossover",
					"ties_add_difference",
				],),
				"precision": (["fp16", "original"],),
				"weights_clip": ("BOOLEAN", {"default": True}),
				"mem_device": (["cuda", "cpu"],),
				"work_device": (["cuda", "cpu"],),
				"threads": ("INT", {"default": 4, "min": 1, "max": 24}),
				"mbw_preset_alpha": mbw_presets,
				"alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"beta": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
				"re_basin": ("BOOLEAN", {"default": False}),
				"re_basin_iterations": ("INT", {"default": 5, "min": 1, "max": 25})
			}
		}

	RETURN_TYPES = ["MODEL", "CLIP"]
	FUNCTION = "merge"

	CATEGORY = "TechNodes/merging"

	# The main merge function
	def merge(self, model_a, model_b, model_c, merge_mode, precision, weights_clip, mem_device, work_device, threads, mbw_preset_alpha, alpha, beta, re_basin, re_basin_iterations, optional_model_a = None, optional_clip_a = None, optional_model_b = None, optional_clip_b = None, optional_model_c = None, optional_clip_c = None, optional_mbw_layers_alpha = None):
	 
		if model_a == "none" and optional_model_a is None:
			raise ValueError("Need either model_a or optional_model_a!")

		if model_b == "none" and optional_model_b is None:
			raise ValueError("Need either model_b or optional_model_b!")

		if model_a == "none" and optional_clip_a is None:
			raise ValueError("Need either model_a or optional_clip_a!")

		if model_b == "none" and optional_clip_b is None:
			raise ValueError("Need either model_b or optional_clip_b!")

		models = { }
  
		if model_a != "none":
			if optional_model_a is None or optional_clip_a is None:
				models['model_a'] = folder_paths.get_full_path("checkpoints", model_a)
   
		if model_b != "none":
			if optional_model_b is None or optional_clip_b is None:
				models['model_b'] = folder_paths.get_full_path("checkpoints", model_b)

		# Add model C if the merge method needs it
		if merge_mode in ["add_difference", "sum_twice", "triple_sum", "euclidean_add_difference", "multiply_difference", "similarity_add_difference", "distribution_crossover", "ties_add_difference"]:
			if model_c == "none" and optional_model_c is None:
				raise ValueError("Need either model_c or optional_model_c!")

			if model_c == "none" and optional_clip_c is None:
				raise ValueError("Need either model_c or optional_clip_c!")

			if model_c != "none":
				if optional_model_c is None or optional_clip_c is None:
					models['model_c'] = folder_paths.get_full_path("checkpoints", model_c)
			
		# Devices
		device = torch.device(mem_device)
		work_device = torch.device(work_device)

		# Merge Arguments
		kwargs = {
			'alpha': alpha,
			'beta': beta,
			're_basin': re_basin,
			're_basin_iterations': re_basin_iterations
		}

		# If a MBW alpha preset is selected replace the alpha with the preset
		if mbw_preset_alpha != "none":
			kwargs["alpha"] = [ mbw_preset_alpha ]

		# If a MBW alpha preset is selected replace the alpha with the preset
		if optional_mbw_layers_alpha is not None:
			kwargs["alpha"] = [ optional_mbw_layers_alpha ]
		
		# Merge the model
		merged_model = merge_models(models, merge_mode, precision, weights_clip, device, work_device, True, threads, optional_model_a, optional_clip_a, optional_model_b, optional_clip_b, optional_model_c, optional_clip_c, **kwargs)

		# Get the config and components from the merged model
		model_config = comfy.model_detection.model_config_from_unet(merged_model, "model.diffusion_model.")
		
		# Create UNet
		unet = model_config.get_model(merged_model, "model.diffusion_model.", device=device)
		unet.load_model_weights(merged_model, "model.diffusion_model.")
   
		# Create ModelPatcher
		model_patcher = comfy.model_patcher.ModelPatcher(
			unet,
			load_device=comfy.model_management.get_torch_device(),
			offload_device=comfy.model_management.unet_offload_device()
		)

		# Create CLIP
		clip_sd = model_config.process_clip_state_dict(merged_model)
		clip = comfy.sd.CLIP(model_config.clip_target(), embedding_directory=None)
		clip.load_sd(clip_sd, full_model=True)

		return (model_patcher, clip)

class SD1_MBWLayers:
	@classmethod
	def INPUT_TYPES(cls) -> Dict[str, tuple]:
		arg_dict = { }
  
		argument = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
  
		for i in range(12):
			arg_dict[f"input_blocks.{i}"] = argument
   
		arg_dict[f"middle_blocks"] = argument
  
		for i in range(12):
			arg_dict[f"output_blocks.{i}"] = argument
   
		return {"required": arg_dict}

	RETURN_TYPES = ["MBW_LAYERS"]
	FUNCTION = "return_layers"
	CATEGORY = "TechNodes/merging"
 
	def return_layers(self, **inputs) -> Dict[str, float]:
		return [ list(inputs.values()) ]

class SD1_MBWLayers_Binary:
	@classmethod
	def INPUT_TYPES(cls) -> Dict[str, tuple]:
		arg_dict = { }
  
		argument = ("BOOLEAN", {"default": False})
  
		for i in range(12):
			arg_dict[f"input_blocks.{i}"] = argument
   
		arg_dict[f"middle_blocks"] = argument
  
		for i in range(12):
			arg_dict[f"output_blocks.{i}"] = argument
   
		return {"required": arg_dict}

	RETURN_TYPES = ["MBW_LAYERS"]
	FUNCTION = "return_layers"
	CATEGORY = "TechNodes/merging"
 
	def return_layers(self, **inputs) -> Dict[str, List[int]]:
		return [list(int(value) for value in inputs.values())]

class SDXL_MBWLayers:
	@classmethod
	def INPUT_TYPES(cls) -> Dict[str, tuple]:
		arg_dict = { }
  
		argument = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
  
		for i in range(9):
			arg_dict[f"input_blocks.{i}"] = argument
   
		arg_dict[f"middle_blocks"] = argument
  
		for i in range(9):
			arg_dict[f"output_blocks.{i}"] = argument
   
		return {"required": arg_dict}

	RETURN_TYPES = ["MBW_LAYERS"]
	FUNCTION = "return_layers"
	CATEGORY = "TechNodes/merging"
 
	def return_layers(self, **inputs) -> Dict[str, float]:
		return [ list(inputs.values()) ]

class SDXL_MBWLayers_Binary:
	@classmethod
	def INPUT_TYPES(cls) -> Dict[str, tuple]:
		arg_dict = { }
  
		argument = ("BOOLEAN", {"default": False})
  
		for i in range(9):
			arg_dict[f"input_blocks.{i}"] = argument
   
		arg_dict[f"middle_blocks"] = argument
  
		for i in range(9):
			arg_dict[f"output_blocks.{i}"] = argument
   
		return {"required": arg_dict}

	RETURN_TYPES = ["MBW_LAYERS"]
	FUNCTION = "return_layers"
	CATEGORY = "TechNodes/merging"
 
	def return_layers(self, **inputs) -> Dict[str, List[int]]:
		return [list(int(value) for value in inputs.values())]

class MBWLayers_String:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mbw_layers": ("STRING", {"multiline": True, "default": "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]"} )
			}
		}

	RETURN_TYPES = ["MBW_LAYERS"]
	FUNCTION = "return_layers"
	CATEGORY = "TechNodes/merging"
 
	def return_layers(self, mbw_layers):
		return [ ast.literal_eval(mbw_layers) ]

class VAERepeat:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ["IMAGE"],
				"vae": ["VAE"],
				"count": ["INT", {"default": 4, "min": 1, "max": 1000000}],
			}
		}
	RETURN_TYPES = ["IMAGE"]
	FUNCTION = "recode"

	CATEGORY = "TechNodes/latent"

	def recode(self, vae, images, count):
		for x in range(count):
			latent = { "samples": vae.encode(images[:,:,:,:3]) } 
			images = vae.decode(latent["samples"])
		return [images]