import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import AttentionBlock, img_to_patch

class Linear(nn.Module):
    '''Linear classifier'''
    def __init__(self, input_dim, n_classes):
        '''input_dim: input dimension
           n_classes: number of classes'''
        super().__init__()
        # Create one perceptron per class
        self.w = nn.Parameter(torch.empty(input_dim, n_classes))
        # We initialize the parameters to standard normal random numbers
        with torch.no_grad():
            self.w.normal_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape the input tensor (N, C, H, W) into (N, C * H * W)
        x = x @ self.w / x.size(-1)**0.5  # Compute the logits
        return x
    

class randomFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, act):
        super().__init__()
        self.w_in = torch.randn(input_dim, hidden_dim, requires_grad=False)
        self.w_out = nn.Parameter(torch.empty(hidden_dim, n_classes))
        self.act = act
        with torch.no_grad():
            self.w_out.normal_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x @ self.w_in / x.size(-1)**0.5
        x = self.act(x)
        x = x @ self.w_out / x.size(-1)**0.5
        return x
    

class deepLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, L, n_classes):
        '''input_dim: input dimension
           hidden_dim: hidden dimension
           L: number of hidden layers
           n_classes: number of classes'''
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(L):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.out_layer = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x) / x.size(-1)**0.5
        x = self.out_layer(x) / x.size(-1) # We use mean-field paramterization for the last layer
        return x
    

class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, L, n_classes, act):
        '''input_dim: input dimension
           hidden_dim: hidden dimension
           L: number of hidden layers
           n_classes: number of classes
           act: activation function'''
        super().__init__()
        self.act = act
        self.layers = nn.ModuleList()
        for _ in range(L):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.out_layer = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x) / x.size(-1)**0.5
            x = self.act(x)
        x = self.out_layer(x) / x.size(-1) # We use mean-field paramterization for the last layer
        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super(conv_block, self).__init__()
        kernel_size = 5
        self.conv  = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=2, bias=True)
        self.scale = (kernel_size**2 * in_channels + 1)**.5
        self.act = act
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x) / self.scale
        x = self.act(x)
        x = self.avgpool(x)
        return x
    

class cnn(nn.Module):
    def __init__(self, in_dim, in_channels, out_channels, num_classes, act):
        super(cnn, self).__init__()
        # for 32*32*(1 or 3) input
        self.act = act

        self.conv_block1 = conv_block(in_channels=in_channels,    out_channels=out_channels,    act=act) #32*32 -> 32*32 ->(pool)->16*16
        self.conv_block2 = conv_block(in_channels=out_channels,   out_channels=out_channels*4,  act=act) #16*16 -> 16*16 ->(pool)->8*8
        self.conv_block3 = conv_block(in_channels=out_channels*4, out_channels=out_channels*16, act=act) #8*8 -> 8*8 ->(pool)->4*4
              
        self.fc1 = nn.Linear(2048, 2048)   
        self.fc2 = nn.Linear(2048 , num_classes)
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.reshape(x.shape[0], -1)  #flatten
        # print(x.shape)
        x = self.fc1(x)/ x.size(-1)**0.5
        x = self.act(x)
        x = self.fc2(x)/ x.size(-1)
        return x


class ViT(nn.Module):
    '''Vision Transformer
       n_classes: number of classes'''
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
       """
       Inputs:
           embed_dim - Dimensionality of the input feature vectors to the Transformer
           hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                        within the Transformer
           num_channels - Number of channels of the input (3 for RGB)
           num_heads - Number of heads to use in the Multi-Head Attention block
           num_layers - Number of layers to use in the Transformer
           num_classes - Number of classes to predict
           patch_size - Number of pixels that the patches have per dimension
           num_patches - Maximum number of patches an image can have
           dropout - Amount of dropout to apply in the feed-forward network and
                     on the input encoding
       """
       super().__init__()
       self.patch_size = patch_size

       # Layers/Networks
       self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
       self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
       self.mlp_head = nn.Sequential(
           nn.LayerNorm(embed_dim),
           nn.Linear(embed_dim, num_classes)
       )
       self.dropout = nn.Dropout(dropout)

       # Parameters/Embeddings
       self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
       self.pos_embedding = nn.Parameter(torch.randn(1, 1+num_patches, embed_dim))
    
    def forward(self, x):
    # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

    # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
    
    # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

    # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out