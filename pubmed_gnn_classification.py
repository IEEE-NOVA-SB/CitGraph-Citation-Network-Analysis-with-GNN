import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# Data Module
class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def prepare_data(self):
        # Perform dataset splitting
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        print("Dataset size:", dataset_size)
        print("Train size:", train_size)
        print("Validation size:", val_size)
        print("Test size:", test_size)
    
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
    
        print("Train dataset size:", len(self.train_dataset))
        print("Validation dataset size:", len(self.val_dataset))
        print("Test dataset size:", len(self.test_dataset))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Model Definition
class GNNModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GNNModel, self).__init__()
        # Initialize GCNConv layers
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)  # No activation for the last layer
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        out = self(batch)
        loss = torch.nn.functional.cross_entropy(out, y)
        self.log('train_loss', loss)  # Logging training loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        out = self(batch)
        val_loss = torch.nn.functional.cross_entropy(out, y)
        self.log('val_loss', val_loss)  # Logging validation loss

    def test_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        out = self(batch)
        test_loss = torch.nn.functional.cross_entropy(out, y)
        self.log('test_loss', test_loss)  # Logging test loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Load Dataset
dataset = Planetoid(root='/tmp/PubMed', name='PubMed')

# Hidden dimensions: increasing progressively
hidden_dims = [64, 128, 256]  # Adjusted hidden dimensions for 3 layers

# Initialize the data module
data_module = GraphDataModule(dataset)
data_module.prepare_data()  # Split the dataset

# Initialize the model with updated hidden dimensions
model = GNNModel(input_dim=dataset.num_node_features, hidden_dims=hidden_dims, output_dim=dataset.num_classes)

# Initialize the trainer
trainer = pl.Trainer(max_epochs=200,)  # Automatically use GPU if available

# Start training
trainer.fit(model, data_module.train_dataloader())

# Run testing
trainer.test(datamodule=data_module.test_dataloader())
