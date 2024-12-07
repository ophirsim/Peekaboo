import timm
import torch

#testing out the vision encoder for dimensions

#model = timm.create_model("hf_hub:timm/vit_base_patch14_dinov2.lvd142m", pretrained=True)
#model = timm.create_model("hf_hub:timm/vit_small_patch14_dinov2.lvd142m", pretrained=True)
model = timm.create_model("hf_hub:timm/vit_small_patch8_224.dino", pretrained=True)

x = torch.randn(1,3,224,224)

y = model(x)

print(y.shape)

#print(timm.list_models())