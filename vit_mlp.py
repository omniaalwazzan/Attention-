
import torch
import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"





class convNext(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        self.calssifier =nn.Sequential(nn.Flatten(1, -1),
                                       nn.Dropout(p=0.1),
                                       #nn.Linear(in_features=262144, out_features=2)) # 1024*7*7 = 50176
                                       nn.Linear(in_features=1024, out_features=2))

    def forward(self, x):
        feature = self.feature(x) # this feature we can use when doing stnad.Att
        
        print(feature.shape)
        flatten_featur = feature.reshape(feature.size(0), -1) #this we need to plot tsne
        x = self.calssifier(feature)
        return flatten_featur
        #return #x

    
model =convNext().to(device=device,dtype=torch.float32)

img = torch.rand(1,3,224,224)
out = model(img)
print(out.shape)



# Define your Transformer model
class TransformerModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(TransformerModel, self).__init__()
        
        # Load a pre-trained CNN for image feature extraction
        convNext = models.convnext_base(pretrained=True)
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        
        # Define tabular feature processing
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model=1024+64, nhead=8, num_encoder_layers=2)
        
        # Output layer
        self.output_layer = nn.Linear(192, num_classes)
        
    def forward(self, tabular_data, image_data):
        
        feature = self.feature(image_data) # this feature we can use when doing stnad.Att
        
        print(feature.shape)
        flatten_featur = feature.reshape(feature.size(0), -1)
        # Extract image features
        #image_features = self.image_encoder(image_data)
        
        # Process tabular data
        tabular_features = self.tabular_encoder(tabular_data)
        
        # Concatenate or stack image and tabular features
        combined_features = torch.cat((tabular_features, flatten_featur), dim=1)
        
        # Add positional encodings if needed
        
        # Pass through the Transformer layers
        transformer_output = self.transformer(combined_features)
        
        # Final output layer
        output = self.output_layer(transformer_output)
        
        return output

# Instantiate the model
model = TransformerModel(num_tabular_features=3, num_classes=2)

# Assuming you have your tabular and image data as tensors
tabular_data = torch.randn(2, 3)
image_data = torch.randn(2, 3, 224, 224)  # Assuming RGB images of size 224x224

# Forward pass
output = model(tabular_data, image_data)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop and data loading will depend on your specific dataset.
