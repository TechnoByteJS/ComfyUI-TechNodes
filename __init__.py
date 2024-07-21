from .sdnextmerge_nodes import *
from .vae_merge import *
from .quant_nodes import *

NODE_CLASS_MAPPINGS = {
	"SDNext Merge": SDNextMerge,
	"VAE Merge": VAEMerge,

	"SD1 MBW Layers": SD1_MBWLayers,
	"SD1 MBW Layers Binary": SD1_MBWLayers_Binary,
	"SDXL MBW Layers": SDXL_MBWLayers,
	"SDXL MBW Layers Binary": SDXL_MBWLayers_Binary,
	"MBW Layers String": MBWLayers_String,

	"VAERepeat": VAERepeat,

	"ModelQuant": ModelQuant,
	"ClipQuant": ClipQuant,
	"VAEQuant": VAEQuant,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"SDNext Merge": "SDNext Merge",
	"VAE Merge": "VAE Merge",

 	"SD1 MBW Layers": "SD1 MBW Layers",
	"SD1 MBW Layers Binary": "SD1 MBW Layers Binary",
 	"SDXL MBW Layers": "SDXL MBW Layers",
	"SDXL MBW Layers Binary": "SDXL MBW Layers Binary",
 	"MBW Layers String": "MBW Layers String",

	"VAERepeat": "Repeat VAE",

	"ModelQuant": "ModelQuant",
	"ClipQuant": "ClipQuant",
	"VAEQuant": "VAEQuant",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
 