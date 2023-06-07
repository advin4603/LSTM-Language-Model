import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dimensions: int, hidden_state_dimensions: int, stacks: int = 2):
        super().__init__()

        self.vocabulary_size, self.embedding_dimensions = vocabulary_size, embedding_dimensions

        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_dimensions)
        self.hidden_state_dimensions = hidden_state_dimensions
        self.lstm = nn.LSTM(self.embedding_dimensions, self.hidden_state_dimensions, num_layers=stacks,
                            batch_first=True)

        self.decoder = nn.Linear(self.hidden_state_dimensions, self.vocabulary_size)

    def forward(self, sentence: torch.Tensor):
        embeddings = self.embedding_layer(sentence)

        lstm_out, _ = self.lstm(embeddings)

        return self.decoder(lstm_out)


class LitLSTMLanguageModel(pl.LightningModule):
    def __init__(self, learning_rate: float, vocabulary_size: int, embedding_dimensions: int, hidden_state_dimensions: int,
                 stacks: int = 2):
        super().__init__()
        self.lstm_language_model = LSTMLanguageModel(vocabulary_size, embedding_dimensions, hidden_state_dimensions,
                                                     stacks)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

    def training_step(self, batch: tuple[torch.tensor, torch.tensor]):
        x, y = batch
        pred = self.lstm_language_model(x).permute(0, 2, 1)
        loss = F.cross_entropy(pred, y)
        self.log("Train Perplexity", torch.exp(loss), prog_bar=True, on_step=True)
        return loss

    def test_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        pred = self.lstm_language_model(x).permute(0, 2, 1)
        loss = F.cross_entropy(pred, y)
        perplexity = torch.exp(loss)
        self.log("Test Perplexity", perplexity, prog_bar=True, on_epoch=True, on_step=False)
        return perplexity

    def validation_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        pred = self.lstm_language_model(x).permute(0, 2, 1)
        loss = F.cross_entropy(pred, y)
        perplexity = torch.exp(loss)
        self.log("Validation Perplexity", perplexity, prog_bar=True, on_epoch=True, on_step=False)
        return perplexity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
