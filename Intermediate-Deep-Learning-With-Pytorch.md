# Deep Learning (24AM654)

Lab Assignment - 2 \
**Course :** Intermediate Deep Learning with Pytorch (Datacamp).

## Module-1

Training Robust Neural Nets:

1. Pytorch & OOP *(10-03-2025, Mon)*
2. Optimizers training & Evaluation *(11-03-2025, Tue)*

---

**Pytorhc & OOP: *(10-03-2025, Mon)***

We use pytorch to define:

- Pytorch Datasets.
- Pytorch Models.

To build a custom dataset:

- Data Loader.

        from torch.utils.data import Dataset  

        class WaterDataset(Dataset):

            def __init__(self, csv_path):
                
                # below super().__init__() helps make the class behaves like a torch dataset.
                
                super().__init__()
                df = pd.read_csv(csv_path)
                self.data = df.to_numpy()

            # we need to create a len function to return the size of the datset.

            def __len__(self):
                return self.data.shape[0]

            # we need to create a access function "__getitem__" that takes an value as an index (idx) and returns the value at that index (idx), the function returns features and lebels.

            def __getitem__(slef, idx):
                features = self.data[idx : -1]
                labels = self.data[idx, -1]
                return features, labels

Now we create an instance of a dataset class thats already created and pass the path of the dataset.

        dataset = WaterDatset(
            "__file-path__"
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size = 2,
            shuffle = True,
        )

We can use **next(iter( ))** to iterate over the features & labels.

    features, labels =next(iter(dataloader_train))

Now using OOP of python we create a class based model.

        # Net() is the base class for creating a NN in pytorch.
        and we add all the Linear() layers in the __init__ class.

        class Net(nn.module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(a, b)
                self.fc2 = nn.Linear(b, c)
                self.fc3 = nn.Linear(c, d)

        # now we create a forward( ) function that makes a forward pass across the network.

        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))

            return x
        net = Net()

---

**Optimizers training & Evaluation: *(11-03-2025, Tue)***
