import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s

print(f"--> CARGANDO MÓDULO: {__file__}")


class LSA64MViT(nn.Module):
    def __init__(self, num_classes=64, freeze_backbone=False, weights="DEFAULT"):
        super().__init__()
        print("--> INICIALIZANDO MODELO: LSA64MViT")
        # 1. Cargar MViT v2 Small pre-entrenado en Kinetics-400
        self.model = mvit_v2_s(weights=weights)

        # 2. Cirugía de la capa de clasificación
        original_head = self.model.head[1]
        in_features = original_head.in_features

        # Reemplazamos con una nueva capa lineal para nuestras 64 clases
        self.model.head[1] = nn.Linear(
            in_features=in_features, out_features=num_classes)

        # 3. Congelamiento opcional
        if freeze_backbone:
            self.freeze_backbone_weights()

    def freeze_backbone_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("Inicializando LSA64MViT...")
    model = LSA64MViT(num_classes=64)
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    print(f"Realizando forward pass con entrada: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    if output.shape == (2, 64):
        print("✔ Éxito: La forma de salida es correcta [2, 64].")
    else:
        print(f"✘ Fallo: Se esperaba [2, 64] pero se obtuvo {output.shape}.")
