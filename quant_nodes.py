import torch
import copy

import folder_paths

import comfy_extras.nodes_model_merging

def quantize_tensor(tensor, num_bits=8, dtype=torch.float16, dequant=True):
	"""
	Quantizes a tensor to a specified number of bits.

	Args:
		tensor (torch.Tensor): The input tensor to be quantized.
		num_bits (int): The number of bits to use for quantization (default: 8).
		dtype(torch.dtype): The datatype to use for the output (default: torch.float16).
		dequant (bool): Whether to dequantize or not (default: true).

	Returns:
		torch.Tensor: The quantized tensor.
	"""
	# Determine the minimum and maximum values of the tensor
	min_val = tensor.min()
	max_val = tensor.max()

	# Calculate the scale factor and zero point
	qmin = 0
	qmax = 2 ** num_bits - 1
	scale = (max_val - min_val) / (qmax - qmin)
	zero_point = qmin - torch.round(min_val / scale)

	# Quantize the tensor
	quantized_tensor = torch.round(tensor / scale + zero_point)
	quantized_tensor = torch.clamp(quantized_tensor, qmin, qmax)

	# Convert the quantized tensor to the datatype
	dequantized_tensor = quantized_tensor.to(dtype)

	if dequant:
		# De-quantize the tensor
		dequantized_tensor = (dequantized_tensor - zero_point) * scale
	
	return dequantized_tensor

def quantize_model(model, in_bits, mid_bits, out_bits, dtype=torch.float16, dequant=True):
	# Clone the base model to create a new one
	quantized_model = model.clone()
	
	# Get the key patches from the model with the prefix "diffusion_model."
	key_patches = quantized_model.get_key_patches("diffusion_model.")
	
	# Iterate over each key patch in the patches
	for key in key_patches:
		if ".input_" in key:
			num_bits = in_bits
		elif ".middle_" in key:
			num_bits = mid_bits
		elif ".output_" in key:
			num_bits = out_bits
		else:
			num_bits = 8

		quantized_tensor = quantize_tensor(key_patches[key][0], num_bits, dtype, dequant)
		quantized_model.add_patches({key: (quantized_tensor,)}, 1, 0)
	
	# Return the quantized model
	return quantized_model

def quantize_clip(clip, bits, dtype=torch.float16, dequant=True):
	# Clone the base model to create a new one
	quantized_clip = clip.clone()
	
	# Get the key patches from the model with the prefix "diffusion_model."
	key_patches = quantized_clip.get_key_patches()
	
	# Iterate over each key patch in the patches
	for key in key_patches:	
		quantized_tensor = quantize_tensor(key_patches[key][0], bits, dtype, dequant)
		quantized_clip.add_patches({key: (quantized_tensor,)}, 1, 0)
	
	# Return the quantized model
	return quantized_clip

def quantize_vae(vae, bits, dtype=torch.float16, dequant=True):
	# Create a clone of the VAE model
	quantized_vae = copy.deepcopy(vae)
	
	# Get the state dictionary from the clone
	state_dict = quantized_vae.first_stage_model.state_dict()
	
	# Iterate over each key-value pair in the state dictionary
	for key, value in state_dict.items():
		state_dict[key] = quantize_tensor(value, bits, dtype, dequant)
	
	# Load the quantized state dictionary back into the clone
	quantized_vae.first_stage_model.load_state_dict(state_dict)
	
	# Return the quantized clone
	return quantized_vae

class ModelQuant:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model": ["MODEL"],
				"in_bits": ("INT", {"default": 8, "min": 1, "max": 8}),
				"mid_bits": ("INT", {"default": 8, "min": 1, "max": 8}),
				"out_bits": ("INT", {"default": 8, "min": 1, "max": 8}),
			}
		}

	RETURN_TYPES = ["MODEL"]
	FUNCTION = "quant_model"

	CATEGORY = "TechNodes/quantization"

	def quant_model(self, model, in_bits, mid_bits, out_bits):
		return [quantize_model(model, in_bits, mid_bits, out_bits)]
		

class ClipQuant:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"clip": ["CLIP"],
				"bits": ("INT", {"default": 8, "min": 1, "max": 8}),
			}
		}

	RETURN_TYPES = ["CLIP"]
	FUNCTION = "quant_clip"

	CATEGORY = "TechNodes/quantization"

	def quant_clip(self, clip, bits):
		return [quantize_clip(clip, bits)]
		

class VAEQuant:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"vae": ["VAE"],
				"bits": ("INT", {"default": 8, "min": 1, "max": 8}),
			}
		}

	RETURN_TYPES = ["VAE"]
	FUNCTION = "quant_vae"

	CATEGORY = "TechNodes/quantization"

	def quant_vae(self, vae, bits):
		return [quantize_vae(vae, bits)]
