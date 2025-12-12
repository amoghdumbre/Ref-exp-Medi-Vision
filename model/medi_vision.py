import torch
from torch import nn
from .mplug import MplugVisualTransformer
from . import xbert
from transformers import AutoTokenizer, AutoConfig

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_query, x_key_value):
        # x_query: [B, N_q, C] (e.g. CLS from Large Branch)
        # x_key_value: [B, N_kv, C] (e.g. Patch tokens from Small Branch)
        
        B, Nq, C = x_query.shape
        _, Nkv, _ = x_key_value.shape
        
        q = self.wq(x_query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_key_value).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_key_value).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MediVisionModel(nn.Module):
    def __init__(self, model_config, train_dataset):
        super().__init__()
        self.model_config = model_config
        
        # --- Multi-Scale Vision Transformers ---
        # Large Branch (L-Branch)
        # Assuming config has 'large_branch' and 'small_branch' sections or we derive defaults
        lb_cfg = model_config.get('large_branch', {
            'input_resolution': 224, 'patch_size': 16, 'width': 768, 'layers': 12, 'heads': 12, 'output_dim': 768
        })
        self.large_branch = MplugVisualTransformer(
            input_resolution=lb_cfg['input_resolution'],
            patch_size=lb_cfg['patch_size'],
            width=lb_cfg['width'],
            layers=lb_cfg['layers'],
            heads=lb_cfg['heads'],
            output_dim=lb_cfg['output_dim']
        )
        
        # Small Branch (S-Branch) - Finer grain (smaller patch), fewer layers/width usually
        sb_cfg = model_config.get('small_branch', {
            'input_resolution': 224, 'patch_size': 8, 'width': 384, 'layers': 6, 'heads': 6, 'output_dim': 768 # output_dim must match for fusion often, or be projected
        })
        self.small_branch = MplugVisualTransformer(
            input_resolution=sb_cfg['input_resolution'],
            patch_size=sb_cfg['patch_size'],
            width=sb_cfg['width'],
            layers=sb_cfg['layers'],
            heads=sb_cfg['heads'],
            output_dim=sb_cfg['output_dim']
        )
        
        # Ensure dimensions match for cross attention or add projection
        self.dim = lb_cfg['output_dim']
        if sb_cfg['output_dim'] != self.dim:
             self.small_proj = nn.Linear(sb_cfg['output_dim'], self.dim)
        else:
             self.small_proj = nn.Identity()

        # Cross Attention Module
        self.cross_attention = CrossAttention(dim=self.dim)
        
        # Text Encoder (BERT-based as per RAMM base, simulating LLAMA2 capabilities via pre-trained weights if avail)
        # Note: Paper says LLAMA 2, but RAMM uses xbert. We stick to xbert for compatibility unless instructed to pull heavy LLAMA weights.
        self.text_encoder = xbert.BertModel.from_pretrained(model_config.get('text_tokenizer', 'bert-base-uncased'), config=model_config.get('text_encoder', None))
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config.get('text_tokenizer', 'bert-base-uncased'))
        
        self.text_feat_dim = self.text_encoder.config.hidden_size
        
        # Projections for fusion
        self.visual_fc = nn.Sequential(
            nn.Linear(self.dim, self.text_feat_dim),
            nn.LayerNorm(self.text_feat_dim, eps=1e-12),
            nn.Dropout(0.1)
        )
        
        # Classifier
        self.classifier = nn.Linear(self.text_feat_dim, train_dataset.num_answers)

    def forward(self, image, text):
        # --- Visual Encoding ---
        # Large Branch Forward
        # Returns [B, N, D]
        feat_large = self.large_branch(image, skip_last_layer=False) 
        
        # Small Branch Forward
        feat_small = self.small_branch(image, skip_last_layer=False)
        feat_small = self.small_proj(feat_small)
        
        # Extract CLS and Patches
        # MplugVisualTransformer returns sequence with [CLS, patch, patch, ...] usually or [patch, ..., CLS]
        # Based on mplug code: 
        # x = torch.cat([class_embedding, x], dim=1) -> CLS is at index 0.
        
        cls_large = feat_large[:, 0:1, :] # [B, 1, D]
        patch_small = feat_small[:, 1:, :] # [B, N_small, D]
        
        # --- Cross Attention ---
        # Query: Large CLS, Key/Value: Small Patches
        # Paper Figure 3: CLS from large branch attends to patch tokens from small branch
        attended_cls = self.cross_attention(cls_large, patch_small) # [B, 1, D]
        
        # Fusion (Paper says Sum or Concat, section 3.5 says "final embeddings from both branches are fused via sum operation")
        # We might sum the attended CLS with the original Large CLS, or sum Large CLS + Small CLS (Class Token Fusion in 3.3).
        # Section 3.3 says "Cross-Attention Fusion... combine patch tokens from one branch with CLS token from another".
        # Section 3.5 "final embeddings from both branches are fused via sum operation".
        # Let's assume we sum the attended outcome with the original large representation or mix them.
        # For simplicity and robustness:
        visual_embed = attended_cls + cls_large # Residual connection style or just sum
        
        # --- Text Encoding ---
        visual_embed = self.visual_fc(visual_embed) # Project to text dim
        visual_atts = torch.ones(visual_embed.size()[:-1], dtype=torch.long).to(image.device)
        
        text_input = self.text_tokenizer(text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")
        
        # --- Multimodal Fusion (using BERT cross attention) ---
        question_output = self.text_encoder(
            text_input["input_ids"].to(image.device), 
            attention_mask=text_input["attention_mask"].to(image.device),
            encoder_hidden_states=visual_embed,
            encoder_attention_mask=visual_atts,
            return_dict=True
        )
        
        hidden = question_output.last_hidden_state[:, 0]
        logits = self.classifier(hidden)
        
        return logits

class MedSAMPreprocessor:
    """
    Wrapper for MedSAM Segmentation (Placeholder).
    
    For full implementation and usage, please refer to the official MedSAM repository:
    https://github.com/bowang-lab/MedSAM.git
    
    You may also refer to the local file: @[/Users/amoghdumbre/Downloads/medsam.py]
    
    Usage (Conceptual):
        import numpy as np
        from segment_anything import sam_model_registry, SamPredictor
        
        detector = MedSAMPreprocessor(ckpt_path='work_dir/MedSAM/medsam_vit_b.pth')
        mask = detector.segment(image_array, box_prompt)
    """
    def __init__(self, medsam_ckpt_path=None, model_type='vit_b', device='cuda'):
        self.device = device
        # This requires 'segment_anything' installed and 'medsam' checkpoint
        # from segment_anything import sam_model_registry, SamPredictor 
        # self.sam_model = sam_model_registry[model_type](checkpoint=medsam_ckpt_path)
        # self.sam_model.to(device=self.device)
        # self.sam_model.eval()
        # self.predictor = SamPredictor(self.sam_model)
        pass

    def segment(self, image, box):
        """
        Segment the image within the bounding box.
        
        Args:
            image: (H, W, 3) numpy array
            box: [x_min, y_min, x_max, y_max]
        Returns:
            mask: (H, W) binary mask
        """
        # self.predictor.set_image(image)
        # masks, _, _ = self.predictor.predict(
        #     box=np.array(box),
        #     multimask_output=False
        # )
        # return masks[0]
        print("MedSAM segmentation placeholder. Please install segment-anything and provide checkpoint.")
        return None
