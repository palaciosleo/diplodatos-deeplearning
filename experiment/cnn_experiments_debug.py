import argparse
import gzip
import json
import logging
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import MeliChallengeDataset
from utils import PadSequences


logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    level=logging.INFO
)


class CNNClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 filter_count=100,
                 filters_length=[2, 3, 4],
                 hidden_layers=[256, 512],
                 batch_size=128,
                 dropout=0.3,
                 vector_size=300,
                 freeze_embedings=True):
        super().__init__()
        with gzip.open(token_to_index, "rt") as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, "rt") as fh:
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    embeddings_matrix[token_to_index[word]] =\
                        torch.FloatTensor([float(n) for n in vector.split()])
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        self.filter_count = filter_count
        self.filters_length = filters_length
        self.batch_size = batch_size
        self.dropout = dropout
        self.convs = []
        for filter_length in self.filters_length:
            self.convs.append(
                nn.Conv1d(vector_size, self.filter_count, filter_length)
            )
        self.convs = nn.ModuleList(self.convs)

        self.fc = nn.Linear(self.filter_count * len(self.filters_length), self.batch_size)

        self.hidden_layers = [
            nn.Linear(batch_size, hidden_layers[0])
        ]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size

    @staticmethod
    def conv_global_max_pool(x, conv):
        return F.relu(conv(x).transpose(1, 2).max(1)[0])

    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)
        x = [self.conv_global_max_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)

        x = torch.sigmoid(self.output(x))
        return x


if __name__ == "__main__":
    train_data = '../data/meli-challenge-2019/spanish.train.jsonl.gz'
    test_data = None
    token_to_index = '../data/meli-challenge-2019/spanish_token_to_index.json.gz'
    pretrained_embeddings = '../data/SBW-vectors-300-min5.txt.gz'
    language = 'spanish'
    validation_data = '../data/meli-challenge-2019/spanish.validation.jsonl.gz'
    embeddings_size = 300
    dropout = 0.3
    batch_size = 128
    filter_count = 5
    filters_length = [2, 3, 4]
    epochs = 3
    exp_name = 'CNN'

    pad_sequences = PadSequences(
        pad_value=0,
        max_length=None,
        min_length=max(filters_length)
    )

    logging.info("Building training dataset")
    train_dataset = MeliChallengeDataset(
        dataset_path=train_data,
        random_buffer_size=2048  # This can be a hypterparameter
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # This can be a hyperparameter
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )

    if validation_data:
        logging.info("Building validation dataset")
        validation_dataset = MeliChallengeDataset(
            dataset_path=validation_data,
            random_buffer_size=1
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        validation_dataset = None
        validation_loader = None

    if test_data:
        logging.info("Building test dataset")
        test_dataset = MeliChallengeDataset(
            dataset_path=test_data,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f"diplodatos.{language}.{exp_name}")

    with mlflow.start_run():
        logging.info("Starting experiment")
        # Log all relevent hyperparameters

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        logging.info("Building classifier")
        model = CNNClassifier(
            pretrained_embeddings_path=pretrained_embeddings,
            token_to_index=token_to_index,
            n_labels=train_dataset.n_labels,
            dropout=dropout,
            batch_size=batch_size,
            vector_size=embeddings_size,
            filter_count=filter_count,
            filters_length=filters_length,
            freeze_embedings=True  # This can be a hyperparameter
        )
        mlflow.log_params({
            "model_type": "Convolutional Neural Network",
            "embeddings": pretrained_embeddings,
            "batch_size": batch_size,
            "filter_count": filter_count,
            "filters_length":  filters_length,
            "dropout": dropout,
            "embeddings_size": embeddings_size,
            "epochs": epochs,
            "model": str(model)[:249]
        })
        logging.info(str(model))

        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,  # This can be a hyperparameter
            weight_decay=1e-5  # This can be a hyperparameter
        )

        logging.info("Training classifier")
        for epoch in trange(epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch["data"].to(device)
                target = batch["target"].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric("train_loss", sum(running_loss) / len(running_loss), epoch)

            if validation_dataset:
                logging.info("Evaluating model on validation")
                model.eval()
                running_loss = []
                targets = []
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch["data"].to(device)
                        target = batch["target"].to(device)
                        output = model(data)
                        running_loss.append(
                            loss(output, target).item()
                        )
                        targets.extend(batch["target"].numpy())
                        predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                    mlflow.log_metric("validation_loss", sum(running_loss) / len(running_loss), epoch)
                    mlflow.log_metric("validation_bacc", balanced_accuracy_score(targets, predictions), epoch)

        if test_dataset:
            logging.info("Evaluating model on test")
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)
                    output = model(data)
                    running_loss.append(
                        loss(output, target).item()
                    )
                    targets.extend(batch["target"].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric("test_loss", sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric("test_bacc", balanced_accuracy_score(targets, predictions), epoch)
