import os
import folder_paths

from tqdm import tqdm
import torch
import safetensors.torch

import comfy.sd
import comfy.utils

from . import merge_methods

from torch import nn

def merge_state_dict(sd_a, sd_b, sd_c, alpha, beta, weights, mode):
	def get_alpha(key):
		try:
			filtered = sorted(
				[x for x in weights.keys() if key.startswith(x)], key=len, reverse=True
			)
			if len(filtered) < 1:
				return alpha
			return weights[filtered[0]]
		except:
			return alpha

	ckpt_keys = (
		sd_a.keys() & sd_b.keys()
		if sd_c is None
		else sd_a.keys() & sd_b.keys() & sd_c.keys()
	)

	for key in tqdm(ckpt_keys):
		current_alpha = get_alpha(key) if weights is not None else alpha

		if mode == "weighted_sum":
			sd_a[key] = merge_methods.weighted_sum(a = sd_a[key], b = sd_b[key], alpha = current_alpha)
		elif mode == "weighted_subtraction":
			sd_a[key] = merge_methods.weighted_subtraction(a = sd_a[key], b = sd_b[key], alpha = current_alpha, beta=beta)
		elif mode == "tensor_sum":
			sd_a[key] = merge_methods.tensor_sum(a = sd_a[key], b = sd_b[key], alpha = current_alpha, beta=beta)
		elif mode == "add_difference":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.add_difference(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha)
		elif mode == "sum_twice":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.sum_twice(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha, beta = beta)
		elif mode == "triple_sum":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.triple_sum(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha, beta = beta)
		elif mode == "euclidean_add_difference":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.euclidean_add_difference(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha)
		elif mode == "multiply_difference":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.multiply_difference(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha, beta = beta)
		elif mode == "top_k_tensor_sum":
			sd_a[key] = merge_methods.top_k_tensor_sum(a = sd_a[key], b = sd_b[key], alpha = current_alpha, beta=beta)
		elif mode == "similarity_add_difference":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.similarity_add_difference(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha, beta = beta)
		elif mode == "distribution_crossover":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.distribution_crossover(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha, beta = beta)
		elif mode == "ties_add_difference":
			assert sd_c is not None, "vae_c is undefined"
			sd_a[key] = merge_methods.ties_add_difference(a = sd_a[key], b = sd_b[key], c = sd_c[key], alpha = current_alpha, beta = beta)

	return sd_a

class VAEMerge:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"vae_a": ("VAE",),
				"vae_b": ("VAE",),
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
				"alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
				"contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
				"use_blocks": ("BOOLEAN", {"default": False}),
				"block_conv_out": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_norm_out": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_0": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_mid": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_conv_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
				"block_quant_conv": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
			},
			"optional": {
				"vae_c": ("VAE",),
			}
		}

	RETURN_TYPES = ["VAE"]
	FUNCTION = "merge_vae"

	CATEGORY = "TechNodes/merging"

	def merge_vae(self, vae_a, vae_b, merge_mode, alpha, beta, brightness, contrast, use_blocks, block_conv_out, block_norm_out, block_0, block_1, block_2, block_3, block_mid, block_conv_in, block_quant_conv, vae_c=None):	
		vae_a_model = vae_a.first_stage_model.state_dict()
		vae_b_model = vae_b.first_stage_model.state_dict()
		vae_c_model = None
		if merge_mode in ["add_difference", "sum_twice", "triple_sum", "euclidean_add_difference", "multiply_difference", "similarity_add_difference", "distribution_crossover", "ties_add_difference"]:
			vae_c_model = vae_c.first_stage_model.state_dict()

		weights = {
			'encoder.conv_out': block_conv_out,
			'encoder.norm_out': block_norm_out,
			'encoder.down.0': block_0,
			'encoder.down.1': block_1,
			'encoder.down.2': block_2,
			'encoder.down.3': block_3,
			'encoder.mid': block_mid,
			'encoder.conv_in': block_conv_in,
			'quant_conv': block_quant_conv,
			'decoder.conv_out': block_conv_out,
			'decoder.norm_out': block_norm_out,
			'decoder.up.0': block_0,
			'decoder.up.1': block_1,
			'decoder.up.2': block_2,
			'decoder.up.3': block_3,
			'decoder.mid': block_mid,
			'decoder.conv_in': block_conv_in,
			'post_quant_conv': block_quant_conv
		}

		if(not use_blocks):
			weights = {}

		merged_vae = merge_state_dict(vae_a_model, vae_b_model, vae_c_model, alpha, beta, weights, mode=merge_mode)

		merged_vae["decoder.conv_out.bias"] = nn.Parameter(merged_vae["decoder.conv_out.bias"] + brightness)

		merged_vae["decoder.conv_out.weight"] = nn.Parameter(merged_vae["decoder.conv_out.weight"] + contrast / 40)

		comfy_vae = comfy.sd.VAE(merged_vae)

		return (comfy_vae,)