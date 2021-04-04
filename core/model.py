import math

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_classes, num_decoder_layers=6, num_decoder_heads=8, uncertainty_gate_type="entropy", uncertainty_threshold=0, weighted_prediction=False, extended_output=False, gpu_streams=True):
        """
        BigPictureNet Model

        Args:
            num_classes (int): Number of classes.
            num_decoder_layers (int, optional): Defaults to 6.
            num_decoder_heads (int, optional): Defaults to 8.
            uncertainty_gate_type (str, optional): Uncertainty gating mechanism to use. Can be one of: "entropy", "relative_softmax_distance", "learned", "learned_metric".
            uncertainty_threshold (int, optional): Used for the uncertainty gating mechanism. If the prediction uncertainty exceeds the uncertainty_threshold, context information is incorporated. Defaults to 0.
            weighted_prediction (bool, optional): If enabled, the model returns an uncertainty-weighted prediction if the uncertainty_gate prediction exceeds the uncertainty threshold.
            extended_output (bool, optional): Can be enabled to return predictions from both branches, uncertainty value and attention maps when the model is in eval mode.
            gpu_streams (bool, optional): If set to True and GPUs are available, multiple gpu streams may be used to parallelize encoding. Defaults to True.
        """        

        super(Model, self).__init__()

        self.NUM_CLASSES = num_classes

        self.context_encoder = Encoder()
        self.target_encoder = Encoder()

        self.CONTEXT_IMAGE_SIZE = self.context_encoder.IMAGE_SIZE
        self.TARGET_IMAGE_SIZE = self.target_encoder.IMAGE_SIZE
        self.NUM_ENCODER_FEATURES = self.context_encoder.NUM_FEATURES

        assert(self.context_encoder.NUM_FEATURES == self.target_encoder.NUM_FEATURES), "Context and target encoder must extract the same number of features."
        
        self.UNCERTAINTY_GATE_TYPE = uncertainty_gate_type
        self.UNCERTAINTY_THRESHOLD = uncertainty_threshold
        self.weighted_prediction = weighted_prediction
        self.uncertainty_gate = build_uncertainty_gate(self.NUM_ENCODER_FEATURES, self.NUM_CLASSES, self.UNCERTAINTY_GATE_TYPE)

        self.tokenizer = Tokenizer()

        self.NUM_CONTEXT_TOKENS = self.tokenizer.NUM_CONTEXT_TOKENS
        self.NUM_TOKEN_FEATURES = self.NUM_ENCODER_FEATURES
        self.NUM_DECODER_HEADS = num_decoder_heads
        self.NUM_DECODER_LAYERS = num_decoder_layers 
        
        assert(self.NUM_TOKEN_FEATURES % self.NUM_DECODER_HEADS == 0), "NUM_TOKEN_FEATURES must be divisible by NUM_DECODER_HEADS."

        self.positional_encoding = PositionalEncoding(self.NUM_CONTEXT_TOKENS, self.NUM_TOKEN_FEATURES)

        self.decoder_layers = TransformerDecoderLayerWithMap(self.NUM_TOKEN_FEATURES, nhead=self.NUM_DECODER_HEADS)
        self.decoder = TransformerDecoderWithMap(self.decoder_layers, self.NUM_DECODER_LAYERS)
 
        self.classifier = nn.Linear(self.NUM_TOKEN_FEATURES, self.NUM_CLASSES)

        self.initialize_weights()

        # cuda streams for parallel encoding
        self.gpu_streams = True if gpu_streams and torch.cuda.is_available() else False
        self.target_stream = torch.cuda.Stream(priority=-1) if self.gpu_streams else None # set target stream as high priority because context encoding may not be necessary due to uncertainty gating
        self.context_stream = torch.cuda.Stream() if self.gpu_streams else None

        self.extended_output = extended_output

    @classmethod
    def from_config(cls, cfg, num_classes=None, extended_output=False, gpu_streams=True):
        """
        Alternative initializer for the use with config files.
        """
        if num_classes is not None:
            cfg.num_classes = num_classes
        else:
            assert(hasattr(cfg, "num_classes")), "Number of classes needs to be specified via cfg or function argument."

        return cls(num_classes=cfg.num_classes, num_decoder_layers=cfg.num_decoder_layers, num_decoder_heads=cfg.num_decoder_heads, uncertainty_gate_type=cfg.uncertainty_gate_type,
                   uncertainty_threshold=cfg.uncertainty_threshold, weighted_prediction=cfg.weighted_prediction, extended_output=extended_output, gpu_streams=gpu_streams)

    def forward(self, context_images, target_images, target_bbox):

        # Encoding of both streams
        if self.gpu_streams:
            torch.cuda.synchronize()
            
        with torch.cuda.stream(self.target_stream):
            target_encoding = self.target_encoder(target_images)
            
            # Uncertainty gating for target
            uncertainty_gate_prediction, uncertainty = self.uncertainty_gate(target_encoding.detach()) # Predictions and associated confidence metrics. Detach because encoder is trained via main branch only.
            
            # During inference, return uncertainty_gate_prediction if uncertainty is below the specified uncertainty threshold.
            # Note: The current implementation makes the gating decision on a per-batch basis. We expect/recommend that a batch size of 1 is used for inference.
            if not self.training and not self.extended_output and torch.all(uncertainty < self.UNCERTAINTY_THRESHOLD).item():
                return uncertainty_gate_prediction
            
        with torch.cuda.stream(self.context_stream):
            context_encoding = self.context_encoder(context_images)
        
        if self.gpu_streams:
            torch.cuda.synchronize()

        # Tokenization and positional encoding
        context_encoding, target_encoding = self.tokenizer(context_encoding, target_encoding)
        context_encoding, target_encoding = self.positional_encoding(context_encoding, target_encoding, target_bbox)

        # Incorporation of context information using transformer decoder
        target_encoding, attention_map = self.decoder(target_encoding, context_encoding)

        # Classification
        main_prediction = self.classifier(target_encoding.squeeze(0))
        weighted_prediction = uncertainty * main_prediction.detach() + (1-uncertainty) * uncertainty_gate_prediction.detach() # detached from main branch and uncertainty gate classifier

        if self.weighted_prediction:
            main_prediction = uncertainty.detach() * main_prediction + (1-uncertainty.detach()) * uncertainty_gate_prediction.detach() # detached from uncertainty branch

        # Return accoring to model state
        if self.training:
            return uncertainty_gate_prediction, main_prediction, weighted_prediction, uncertainty
        elif self.extended_output:
            return uncertainty_gate_prediction, main_prediction, uncertainty, attention_map
        else:
            return main_prediction

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def unfreeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = True


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = torchvision.models.densenet169(pretrained=True).features
        
        self.IMAGE_SIZE = (224, 224)
        self.NUM_FEATURES = 1664

    def forward(self, image):
        return self.encoder(image)


class Tokenizer(nn.Module):

    # TODO: make implementation more general so it works with different encodings and number of (context) tokens can be specified

    def __init__(self):
        super(Tokenizer, self).__init__()

        self.NUM_CONTEXT_TOKENS = 49
        self.NUM_TARGET_TOKENS = 1
        
    def forward(self, context_encoding, target_encoding):
        """
        Creates tokens from the encoded context and target.
        The token shapes are (NUM_CONTEXT_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES)
        and (NUM_TARGET_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES) respectively.
        """

        # one target token
        target_encoding = F.relu(target_encoding)
        target_encoding = F.adaptive_avg_pool2d(target_encoding, (1, 1))
        target_encoding = torch.flatten(target_encoding, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        target_encoding = torch.unsqueeze(target_encoding, 0) # output dimension: (NUM_TARGET_TOKENS=1, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        # 49 context tokens
        context_encoding = F.relu(context_encoding)
        context_encoding = torch.flatten(context_encoding, 2, 3)
        context_encoding = context_encoding.permute(2, 0, 1) # output dimension: (NUM_CONTEXT_TOKENS=49, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        return context_encoding, target_encoding


class PositionalEncoding(nn.Module):
    
    def __init__(self, num_context_tokens, num_token_features):
        super(PositionalEncoding, self).__init__()

        self.NUM_CONTEXT_TOKENS = num_context_tokens
        self.tokens_per_dim = int(math.sqrt(self.NUM_CONTEXT_TOKENS))
        self.positional_encoding = nn.Parameter(torch.zeros(num_context_tokens, 1, num_token_features))
        self.initialize_weights()

    def forward(self, context_tokens, target_tokens, target_bbox):
        context_tokens = context_tokens + self.positional_encoding
        target_tokens = target_tokens + torch.index_select(self.positional_encoding, 0, self.bbox2token(target_bbox)).permute(1,0,2)

        return context_tokens, target_tokens

    def bbox2token(self, bbox):
        """
        Maps relative bbox coordinates to the corresponding token ids (e.g., 0 for the token in the top left).

        Arguments:
            bbox: Tensor of dim (batch_size, 4) where a row corresponds to relative coordinates
                  in the form [xmin, ymin, w, h] (e.g., [0.1, 0.3, 0.2, 0.2]).
        """
        token_ids = ((torch.ceil((bbox[:,0] + bbox[:,2]/2) * self.tokens_per_dim) - 1) + (torch.ceil((bbox[:,1] + bbox[:,3]/2) * self.tokens_per_dim) - 1) * self.tokens_per_dim).long()

        return token_ids
        
    @torch.no_grad()
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def build_uncertainty_gate(num_encoder_features, num_classes, gate_type="entropy"):
    if gate_type == "entropy":
        return UncertaintyGate(num_encoder_features, num_classes)
    elif gate_type == "relative_softmax_distance":
        return RelativeSoftmaxDistanceUncertaintyGate(num_encoder_features, num_classes)
    elif gate_type == "learned":
        return LearnedUncertaintyGate(num_encoder_features, num_classes)
    elif gate_type == "learned_metric":
        return LearnedMetricUncertaintyGate(num_encoder_features, num_classes)
    else:
        raise ValueError("Unsupported uncertainty gate type {}.".format(gate_type))


class UncertaintyGate(nn.Module):

    def __init__(self, num_features, num_classes):
        super(UncertaintyGate, self).__init__()
        self.target_classifier = nn.Linear(num_features, num_classes)
        self.initialize_weights()

    def forward(self, input_features):
        # TODO: could add a few layers here
        
        # flatten featuremap out
        input_features = F.relu(input_features)
        input_features = F.adaptive_avg_pool2d(input_features, (1, 1))
        input_features = torch.flatten(input_features, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        
        predictions = self.target_classifier(input_features) # predictions dimension: (Batchsize, NUM_CLASSES)
        uncertainty = self.compute_uncertainty(predictions.detach())
        
        return predictions, uncertainty

    @staticmethod
    def compute_uncertainty(predictions):
        return -1 * torch.sum(F.softmax(predictions, dim=1) * F.log_softmax(predictions, dim=1), dim=1, keepdim=True) # entropy as metric for uncertainty

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class RelativeSoftmaxDistanceUncertaintyGate(UncertaintyGate):
    
    @staticmethod
    def compute_uncertainty(predictions):
        top2, _ = torch.topk(F.softmax(predictions, dim=1), 2, dim=1)
        uncertainty =  1 - (torch.true_divide(top2[:,0] - top2[:,1], top2[:,0])).unsqueeze(dim=1) # relative distance between largest and 2nd largest value
        
        return uncertainty

class LearnedUncertaintyGate(UncertaintyGate):

    def __init__(self, num_features, num_classes):
        super(LearnedUncertaintyGate, self).__init__(num_features, num_classes)
        self.uncertainty_estimator = nn.Sequential(nn.Linear(num_features, num_features // 2),
                                                   nn.ReLU(),
                                                   nn.Linear(num_features // 2, 1),
                                                   nn.Sigmoid())
        self.initialize_weights()

    def forward(self, input_features):        
        # TODO: could add a few layers here
        
        # flatten featuremap out
        input_features = F.relu(input_features)
        input_features = F.adaptive_avg_pool2d(input_features, (1, 1))
        input_features = torch.flatten(input_features, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        
        predictions = self.target_classifier(input_features) # predictions dimension: (Batchsize, NUM_CLASSES)
        uncertainty = self.uncertainty_estimator(input_features)
        
        return predictions, uncertainty

class LearnedMetricUncertaintyGate(UncertaintyGate):

    def __init__(self, num_features, num_classes):
        super(LearnedMetricUncertaintyGate, self).__init__(num_features, num_classes)
        self.uncertainty_estimator = nn.Sequential(nn.Linear(num_classes, num_classes // 2),
                                                   nn.ReLU(),
                                                   nn.Linear(num_classes // 2, 1),
                                                   nn.Sigmoid())
        self.initialize_weights()

    def compute_uncertainty(self, predictions):
        return self.uncertainty_estimator(predictions)


class TransformerDecoderLayerWithMap(torch.nn.TransformerDecoderLayer):
    """
    Adapted version of torch.nn.TransformerDecoderLayer without the self-attention stage. In addition, the attention map is returned.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        tgt2, attention_map = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_map


class TransformerDecoderWithMap(torch.nn.TransformerDecoder):
    """
    Provides the same functionality as torch.nn.TransformerDecoder but returns the attention maps in addition.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt
        attention_maps = []

        for mod in self.layers:
            output, attention_map = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(attention_maps).permute(1,0,2,3) # attention map shape: (batchsize, num_decoder_layers, num_target_tokens, num_context_tokens)