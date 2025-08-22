import torch
from torch import nn

from .augmentations import GaussianSmoothing

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        if self.gaussianSmoothWidth > 0:
            self.gaussianSmoother = GaussianSmoothing(
                neural_dim, 20, self.gaussianSmoothWidth, dim=1
            )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):

        if self.gaussianSmoothWidth > 0:
            neuralInput = torch.permute(neuralInput, (0, 2, 1))
            neuralInput = self.gaussianSmoother(neuralInput)
            neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out


class MambaDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        d_model,
        d_state,
        d_conv,
        expand_factor,
        layer_dim,
        nDays,
        dropout,
        strideLen,
        kernelLen,
        gaussianSmoothWidth,
        bidirectional,
        device="cuda",
    ):
        super(MambaDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.d_model = d_model
        self.d_state = d_state 
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        if self.gaussianSmoothWidth > 0:
            self.gaussianSmoother = GaussianSmoothing(
                neural_dim, 20, self.gaussianSmoothWidth, dim=1
            )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        
        if self.bidirectional:
            d_mamba = d_model * 2
        else:
            d_mamba = d_model
                
        ModuleList = []
        for i in range(layer_dim):
            ModuleList.append(
                Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_mamba, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand_factor,    # Block expansion factor
                )
            )
            ModuleList.append(torch.nn.Dropout(p=self.dropout))
                    
        self.mamba_decoder = nn.Sequential(*ModuleList)  

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # Linear input layer
        if self.bidirectional:
            self.linear_input = nn.Linear(neural_dim*kernelLen * 2, d_model * 2)
        else:
            self.linear_input = nn.Linear(neural_dim*kernelLen, d_model)
                
        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(d_model * 2, n_classes + 1)
        else:
            self.fc_decoder_out = nn.Linear(d_model, n_classes + 1)  # +1 for CTC blank
        
    def forward(self, neuralInput, dayIdx):
        if self.gaussianSmoothWidth > 0:
            neuralInput = torch.permute(neuralInput, (0, 2, 1))
            neuralInput = self.gaussianSmoother(neuralInput)
            neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
           self.unfolder(
               torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
           ),
           (0, 2, 1),
        )

        if self.bidirectional:
           stridedFlip = torch.flip(stridedInputs, dims=(1,))
           stridedInputs = torch.cat((stridedInputs, stridedFlip), dim=-1)

        mamba_in = self.linear_input(stridedInputs)
       
        hid = self.mamba_decoder(mamba_in)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out

# class MambaDecoder(nn.Module):
#     def __init__(
#         self,
#         neural_dim,
#         n_classes,
#         d_model,
#         d_state,
#         d_conv,
#         expand_factor,
#         layer_dim,
#         nDays,
#         dropout,
#         strideLen,
#         kernelLen,
#         gaussianSmoothWidth,
#         bidirectional,
#         device="cuda",        
# #         self,
# #         neural_dim,
# #         n_classes,
# #         d_model,
# #         d_state,
# #         d_conv,
# #         expand_factor,
# #         layer_dim,
# #         nDays=24,
# #         dropout=0,
# #         device="cuda",
# #         strideLen=4,
# #         kernelLen=14,
# #         gaussianSmoothWidth=0,
# #         bidirectional=False,        
#     ):
#         super(MambaDecoder, self).__init__()

#         # Defining the number of layers and the nodes in each layer
#         self.layer_dim = layer_dim
#         self.d_model = d_model
#         self.d_state = d_state 
#         self.d_conv = d_conv
#         self.expand_factor = expand_factor
#         self.neural_dim = neural_dim
#         self.n_classes = n_classes
#         self.nDays = nDays
#         self.device = device
#         self.dropout = dropout
#         self.strideLen = strideLen
#         self.kernelLen = kernelLen
#         self.gaussianSmoothWidth = gaussianSmoothWidth
#         self.bidirectional = bidirectional
#         self.inputLayerNonlinearity = torch.nn.Softsign()
#         self.unfolder = torch.nn.Unfold(
#             (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
#         )
                
#         if self.gaussianSmoothWidth > 0:
#             self.gaussianSmoother = GaussianSmoothing(
#                 neural_dim, 20, self.gaussianSmoothWidth, dim=1
#             )
#         self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
#         self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

#         for x in range(nDays):
#             self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        
#         d_mamba = d_model
        
#         ModuleList_forward = []
#         ModuleList_backward = []
#         ModuleList_mixingLayer = []
#         for i in range(layer_dim):
#             ModuleList_forward.append(
#                 Mamba(
#                 # This module uses roughly 3 * expand * d_model^2 parameters
#                 d_model=d_mamba, # Model dimension d_model
#                 d_state=d_state,  # SSM state expansion factor
#                 d_conv=d_conv,    # Local convolution width
#                 expand=expand_factor,    # Block expansion factor
#                 ).to(self.device)
#             )
#             ModuleList_backward.append(
#                 Mamba(
#                 # This module uses roughly 3 * expand * d_model^2 parameters
#                 d_model=d_mamba, # Model dimension d_model
#                 d_state=d_state,  # SSM state expansion factor
#                 d_conv=d_conv,    # Local convolution width
#                 expand=expand_factor,    # Block expansion factor
#                 ).to(self.device)
#             )
#             ModuleList_mixingLayer.append(
#                 nn.Sequential(*[nn.Linear(d_model * 2, d_model * 2),
#                                torch.nn.Dropout(p=self.dropout)]).to(self.device)
#             )

#         self.mamba_forward = ModuleList_forward
#         self.mamba_backward = ModuleList_backward
#         self.mamba_mixing = ModuleList_mixingLayer

#         # Input layers
#         for x in range(nDays):
#             setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

#         for x in range(nDays):
#             thisLayer = getattr(self, "inpLayer" + str(x))
#             thisLayer.weight = torch.nn.Parameter(
#                 thisLayer.weight + torch.eye(neural_dim)
#             )

#         # Linear input layer
#         self.linear_input = nn.Linear(neural_dim*kernelLen, d_model)
        
#         # rnn outputs
#         self.fc_decoder_out = nn.Linear(d_model * 2, n_classes + 1)
          
    
#     def forward(self, neuralInput, dayIdx):
        
#         if self.gaussianSmoothWidth > 0:
#             neuralInput = torch.permute(neuralInput, (0, 2, 1))
#             neuralInput = self.gaussianSmoother(neuralInput)
#             neuralInput = torch.permute(neuralInput, (0, 2, 1))

#         # apply day layer
#         dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
#         transformedNeural = torch.einsum(
#             "btd,bdk->btk", neuralInput, dayWeights
#         ) + torch.index_select(self.dayBias, 0, dayIdx)
#         transformedNeural = self.inputLayerNonlinearity(transformedNeural)

#         # stride/kernel
#         stridedInputs = torch.permute(
#            self.unfolder(
#                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
#            ),
#            (0, 2, 1),
#         )

#         stridedInputs = self.linear_input(stridedInputs)
        
#         if self.bidirectional:
#            stridedFlip = torch.flip(stridedInputs, dims=(1,))
#            x = torch.cat((stridedInputs, stridedFlip), dim=-1)
         
#            for forward, backward, mixing in zip(
#                self.mamba_forward, self.mamba_backward, self.mamba_mixing):
#                 x_forward = forward(x[:,:,:self.d_model])
#                 x_backward = backward(x[:,:,self.d_model:])    
#                 x = torch.cat((x_forward, x_backward), dim=-1)
#                 x = mixing(x)
                
#         seq_out = self.fc_decoder_out(x)
#         return seq_out
