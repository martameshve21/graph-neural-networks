
# აქ მოთავსდება მოდელისა და სწავლების ძირითადი კოდი.
# ============================================================
# 1. ბიბლიოთეკების იმპორტი
# ============================================================
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# ============================================================
# 2. შემთხვევითობის კონტროლი
# ============================================================

SEED = 42

random.seed(SEED) 
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. მოწყობილობის არჩევა
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"გამოყენებული მოწყობილობა: {device}")


# ============================================================
# 4. CORA მონაცემთა ნაკრების ჩატვირთვა
# ============================================================

dataset = Planetoid(
    root="data/Cora",
    name="Cora"
)

data = dataset[0]

print("\nმონაცემთა ნაკრები:")
print(dataset)

print("\nგრაფის მონაცემები:")
print(data)


# ============================================================
# 5. მონაცემების ძირითადი მახასიათებლები
# ============================================================

print("\n--- CORA-ს მახასიათებლები ---")

print(f"გრაფების რაოდენობა: {len(dataset)}")
print(f"წვეროების რაოდენობა: {data.num_nodes}")
print(f"წიბოების რაოდენობა: {data.num_edges}")
print(f"მახასიათებლების რაოდენობა: {dataset.num_features}")
print(f"კლასების რაოდენობა: {dataset.num_classes}")

print(
    f"სასწავლო წვეროების რაოდენობა: "
    f"{int(data.train_mask.sum())}"
)

print(
    f"Validation წვეროების რაოდენობა: "
    f"{int(data.val_mask.sum())}"
)

print(
    f"სატესტო წვეროების რაოდენობა: "
    f"{int(data.test_mask.sum())}"
)


# ============================================================
# 6. GCN მოდელის განსაზღვრა
# ============================================================

class GCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_probability
    ):
        super().__init__()

        self.dropout_probability = dropout_probability

        self.gcn1 = GCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim
        )

        self.gcn2 = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim
        )

        self.gcn3 = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim
        )

        self.linear1 = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim
        )

        self.linear2 = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim
        )

    def forward(self, x, edge_index):

        # პირველი GCN ფენა
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout_probability,
            training=self.training
        )

        # მეორე GCN ფენა
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout_probability,
            training=self.training
        )

        # მესამე GCN ფენა
        x = self.gcn3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout_probability,
            training=self.training
        )

        # კლასიფიკატორის პირველი Linear ფენა
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout_probability,
            training=self.training
        )

        # კლასიფიკატორის გამომავალი ფენა
        x = self.linear2(x)

        # თითოეული წვეროსთვის კლასების log-probability
        return F.log_softmax(x, dim=1)


# ============================================================
# 7. ჰიპერპარამეტრები
# ============================================================

HIDDEN_DIM = 64
DROPOUT_PROBABILITY = 0.5
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200


# ============================================================
# 8. მოდელის შექმნა
# ============================================================

model = GCN(
    input_dim=dataset.num_features,
    hidden_dim=HIDDEN_DIM,
    output_dim=dataset.num_classes,
    dropout_probability=DROPOUT_PROBABILITY
).to(device)

data = data.to(device)

print("\nმოდელის არქიტექტურა:")
print(model)


# ============================================================
# 9. ოპტიმიზატორი და დანაკარგის ფუნქცია
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.NLLLoss()


# ============================================================
# 10. სიზუსტის გამოთვლის ფუნქცია
# ============================================================

def calculate_accuracy(log_probabilities, labels):

    predicted_classes = log_probabilities.argmax(dim=1)

    correct_predictions = (
        predicted_classes == labels
    ).sum().item()

    accuracy = correct_predictions / labels.size(0)

    return accuracy


# ============================================================
# 11. ერთი epoch-ის სწავლების ფუნქცია
# ============================================================

def train_one_epoch():

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    train_output = output[data.train_mask]
    train_labels = data.y[data.train_mask]

    loss = criterion(
        train_output,
        train_labels
    )

    loss.backward()

    optimizer.step()

    train_accuracy = calculate_accuracy(
        train_output,
        train_labels
    )

    return loss.item(), train_accuracy


# ============================================================
# 12. შეფასების ფუნქცია
# ============================================================

@torch.no_grad()
def evaluate(mask):

    model.eval()

    output = model(
        data.x,
        data.edge_index
    )

    selected_output = output[mask]
    selected_labels = data.y[mask]

    loss = criterion(
        selected_output,
        selected_labels
    )

    accuracy = calculate_accuracy(
        selected_output,
        selected_labels
    )

    return loss.item(), accuracy


# ============================================================
# 13. მოდელის სწავლება
# ============================================================

best_validation_loss = float("inf")
best_model_state = None

history = {
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": []
}

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch()

    validation_loss, validation_accuracy = evaluate(
        data.val_mask
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(
        validation_accuracy
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch: {epoch:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.4f}"
        )


# ============================================================
# 14. საუკეთესო მოდელის აღდგენა
# ============================================================

model.load_state_dict(best_model_state)

model = model.to(device)


# ============================================================
# 15. სატესტო სიმრავლეზე საბოლოო შეფასება
# ============================================================

test_loss, test_accuracy = evaluate(
    data.test_mask
)

print("\n--- საბოლოო შედეგი ---")

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# ============================================================
# 16. რამდენიმე წვეროს პროგნოზის ნახვა
# ============================================================

model.eval()

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )

    predicted_classes = output.argmax(dim=1)


test_node_indices = (
    data.test_mask.nonzero(as_tuple=False)
    .view(-1)
)

print("\nპირველი 10 სატესტო წვეროს პროგნოზები:")

for node_index in test_node_indices[:10]:

    node_id = node_index.item()

    predicted_class = predicted_classes[node_id].item()
    actual_class = data.y[node_id].item()

    print(
        f"წვერო {node_id:4d} | "
        f"ნამდვილი კლასი: {actual_class} | "
        f"პროგნოზირებული კლასი: {predicted_class}"
    )


results_directory = Path("results")
results_directory.mkdir(exist_ok=True)

history_path = results_directory / "history.json"

with history_path.open("w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)

print(f"სწავლის ისტორია შენახულია: {history_path}")