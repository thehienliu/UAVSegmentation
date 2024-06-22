import torch
from torch import nn
from functools import partial
from typing import OrderedDict, Dict
from .utils.modules import ViTDeit, Deconv2DBlock, Conv2DBlock

class SegmentationVITSAM(nn.Module):
  
  def __init__(self, 
               embed_dim,
               num_heads,
               depth,
               extract_layers,
               encoder_global_attn_indexes,
               drop_rate,
               num_classes,
               ckpt_path=None):
      super().__init__()

      self.prompt_embed_dim = 256
      self.drop_rate = drop_rate
      self.embed_dim = embed_dim

      if self.embed_dim < 512:
          self.skip_dim_11 = 256
          self.skip_dim_12 = 128
          self.bottleneck_dim = 312
      else:
          self.skip_dim_11 = 512
          self.skip_dim_12 = 256
          self.bottleneck_dim = 512

      self.encoder = ViTDeit(
            extract_layers=extract_layers,
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )
      if ckpt_path:
        self.load_checkpoint(ckpt_path)

      # version with shared skip_connections
      self.decoder0 = nn.Sequential(
        Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
        Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
      )  # skip connection after positional encoding, shape should be H, W, 64
      self.decoder1 = nn.Sequential(
        Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
        Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
        Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
      )  # skip connection 1
      self.decoder2 = nn.Sequential(
        Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
        Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
      )  # skip connection 2
      self.decoder3 = nn.Sequential(
        Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
      )  # skip connection 3

      # Upsample branch
      self.bottleneck_upsampler = nn.ConvTranspose2d(
          in_channels=self.embed_dim,
          out_channels=self.bottleneck_dim,
          kernel_size=2,
          stride=2,
          padding=0,
          output_padding=0,
      )

      self.decoder3_upsampler = nn.Sequential(
          Conv2DBlock(
              self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
          ),
          Conv2DBlock(
              self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
          ),
          Conv2DBlock(
              self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
          ),
          nn.ConvTranspose2d(
              in_channels=self.bottleneck_dim,
              out_channels=256,
              kernel_size=2,
              stride=2,
              padding=0,
              output_padding=0,
          ),
      )

      self.decoder2_upsampler = nn.Sequential(
          Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
          Conv2DBlock(256, 256, dropout=self.drop_rate),
          nn.ConvTranspose2d(
              in_channels=256,
              out_channels=128,
              kernel_size=2,
              stride=2,
              padding=0,
              output_padding=0,
          ),
      )

      self.decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
      )
      
      self.decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )


  def forward(self, x: torch.Tensor):
      """
      Args:
          x (torch.Tensor): Images in BCHW style
      """

      z = self.encoder(x)

      # Get original image and unpack feature from encoder
      z0, z1, z2, z3, z4 = x, *z

      outputs = self._forward_upsample(z0, z1, z2, z3, z4)
      return outputs

  def _forward_upsample(
      self,
      z0: torch.Tensor,
      z1: torch.Tensor,
      z2: torch.Tensor,
      z3: torch.Tensor,
      z4: torch.Tensor,
  ) -> torch.Tensor:

      """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
      """
      b4 = self.bottleneck_upsampler(z4)
      b3 = self.decoder3(z3)
      b3 = self.decoder3_upsampler(torch.cat([b3, b4], dim=1))
      b2 = self.decoder2(z2)
      b2 = self.decoder2_upsampler(torch.cat([b2, b3], dim=1))
      b1 = self.decoder1(z1)
      b1 = self.decoder1_upsampler(torch.cat([b1, b2], dim=1))
      b0 = self.decoder0(z0)
      branch_output = self.decoder0_header(torch.cat([b0, b1], dim=1))
      return branch_output

  def _strip_state_dict(self, state_dict: Dict) -> OrderedDict:
        """Strip the 'image_encoder' from the SAM state dict keys."""
        new_dict = {}
        for k, w in state_dict.items():
            if "image_encoder" in k:
                spl = ["".join(kk) for kk in k.split(".")]
                new_key = ".".join(spl[1:])
                new_dict[new_key] = w

        return new_dict

  def load_checkpoint(self, ckpt_path) -> None:
    """Load the weights from the checkpoint."""
    state_dict = torch.load(
        ckpt_path, map_location=lambda storage, loc: storage
    )
    try:
        msg = self.encoder.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        new_ckpt = self._strip_state_dict(state_dict)
        msg = self.encoder.load_state_dict(new_ckpt, strict=True)
    except BaseException as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")

    print(f"Loading pre-trained {type(self.encoder).__name__} checkpoint: {msg}")