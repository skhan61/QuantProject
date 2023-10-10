import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


class PositionalEncoding(nn.Module):
    """
    Didn't have much difference
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class LinearAttentionWithKQV(nn.Module):
    """
    Light on compute. MAX_LEN x D_MODEL;
    use this if OOM
    """

    def __init__(self, d_model, dropout=0.1):
        super(LinearAttentionWithKQV, self).__init__()
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.dim = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        k = self.linear_k(inputs)
        q = self.linear_q(inputs)
        v = self.linear_v(inputs)
        n = torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))

        scores = torch.bmm(k.transpose(1, 2), v) / n
        scores = self.dropout(scores)

        # print(scores.shape, mask.shape)
        if mask is not None:
            q = q.masked_fill(mask == 0, float("-inf"))

        q = torch.softmax(q, dim=1)

        attention_weights = torch.bmm(q, scores) / n

        return attention_weights


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class SelfAttention(nn.Module):
    """
    Heavy on compute. Vanilla attention MAXLEN x MAXLEN
    """

    def __init__(self, d_model, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.dim = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        k = self.linear_k(inputs)
        q = self.linear_q(inputs)
        v = self.linear_v(inputs)
        n = torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))

        scores = torch.bmm(q, k.transpose(1, 2)) / n
        # print(scores.shape)
        if mask is not None:
            scores = scores .masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=1)
        attention_weights = self.dropout(attention_weights)

        attention_weights = torch.bmm(attention_weights, v) / n

        return attention_weights

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        #self.attention = LinearAttentionWithKQV(d_model, dropout=0.1)  # drop-in attention
        self.attention = SelfAttention(d_model, dropout=0.15)
        self.layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_heads)]
        )
        self.fc = nn.Linear(num_heads * d_model, d_model)

    def forward(self, inputs, mask=None):
        x = inputs
        head_outputs = []
        for layer in range(self.num_heads):
            attention_weights = self.attention(x, mask)
            head_output = x * attention_weights
            head_output = self.layers[layer](head_output)
            head_output = F.relu(head_output)
            head_outputs.append(head_output)

        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.fc(concatenated)

        return output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        num_heads,
        num_layers,
        dropout_prob=0.15,
        max_len=5000,
    ):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.d_model = d_model

        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.attention = MultiHeadLinearAttention(d_model, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(FeedForwardLayer(d_model=d_model))
                for _ in range(num_layers)
            ]
        )
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.Linear(d_model, d_model)
        )

    def forward(self, inputs, mask=None):
        x = self.mapper(inputs)
        pe = self.positional_encoding(x)
        x = x + pe # works without PE as well
        for layer in range(self.num_layers):
            attention_weights = self.attention(x, mask)
            x = x + attention_weights
            x = F.layer_norm(x, x.shape[1:])
            x = F.dropout(x, p=self.dropout_prob)

            op = self.layers[layer](x)
            x = x + op
            x = F.layer_norm(x, x.shape[1:])
            x = F.dropout(x, p=self.dropout_prob)

        outputs = self.fc(x)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        num_heads,
        num_layers,
        dropout_prob=0.15,
        max_len=5000,
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.d_model = d_model

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SELU(),
            nn.Linear(d_model // 2, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, inputs, mask=None):
        emb = self.encoder(inputs, mask)
        outputs = self.fc(emb)
        return outputs


# def test_model():

#     inputs = [
#         torch.randint(0, 4, (5, FEATURE_DIM)).float(),
#         torch.randint(0, 4, (3, FEATURE_DIM)).float(),
#     ]
#     labels = [
#         torch.randint(0, 2, (5, OUTPUT_DIM)).float(),
#         torch.randint(0, 2, (3, OUTPUT_DIM)).float(),
#     ]

#     padded_inputs, masks_inputs = pad_sequence(inputs, padding_value=0, max_len=MAX_LEN)
#     padded_labels, masks_labels = pad_sequence(labels, padding_value=0, max_len=MAX_LEN)

#     transformer = Transformer(
#         input_dim=FEATURE_DIM,
#         d_model=HIDDEN_DIM,
#         output_dim=OUTPUT_DIM,
#         num_heads=NUM_HEADS,
#         num_layers=NUM_LAYERS,
#         max_len=MAX_LEN,
#     )

#     with torch.no_grad():
#         outputs = transformer(padded_inputs, masks_inputs)

#     assert torch.isnan(outputs).sum() == 0
#     assert outputs.shape[:2] == padded_inputs.shape[:2]
#     assert outputs.shape[-1] == len(target_names)

#     print("Input Shape:", padded_inputs.shape)
#     print("Output Shape:", outputs.shape)

#     del transformer
#     del inputs, labels
#     del padded_inputs, masks_inputs, padded_labels, masks_labels
#     del outputs

#     gc.collect()

# if __name__ == "__main__":
#     main()
# test_model()


import torch.nn as nn
import torch

# Hyperparameters
HIDDEN_LAYER_1 = 256
HIDDEN_LAYER_2 = 128
HIDDEN_LAYER_3 = 64
DROPOUT_PROB = 0.5

class RankPredictorNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RankPredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_LAYER_1)   # First fully connected layer
        self.fc2 = nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2) # Second fully connected layer
        self.fc3 = nn.Linear(HIDDEN_LAYER_2, HIDDEN_LAYER_3) # Third fully connected layer
        self.fc4 = nn.Linear(HIDDEN_LAYER_3, output_dim)  # Output layer
        self.dropout = nn.Dropout(DROPOUT_PROB)            # Dropout layer to prevent overfitting
        self.sigmoid = nn.Sigmoid()                        # Sigmoid activation function

    def forward(self, x, mask):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)  # Apply the sigmoid function to ensure output is between 0 and 1
        return x * mask      # This ensures that the outputs for padded positions are zero

# Uncomment this to test your model
# def test_rank_predictor_model():
#     inputs = [
#         torch.randint(0, 4, (5, FEATURE_DIM)).float(),
#         torch.randint(0, 4, (3, FEATURE_DIM)).float(),
#     ]

#     # Padding sequences to have the same length for batch processing
#     padded_inputs, masks_inputs = pad_sequence(inputs)

#     model = RankPredictorNN(FEATURE_DIM, OUTPUT_DIM)
#     outputs = model(padded_inputs, masks_inputs)

#     print("Input Shape:", padded_inputs.shape)
#     print("Output Shape:", outputs.shape)

# test_rank_predictor_model()

