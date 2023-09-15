from ..utils import *
from ..data_load.data_loader import TrainDataset, ValidDataset, TestDataset

class TrainProcessor():
    def __init__(self, args, data) -> None:
        self.args = args
        self.dataset = TrainDataset(args, data)
        self.data_loader = DataLoader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=self.args.batch_size
        )

    def process_epoch(self, model, optimizer):
        model.train()
        total_loss = 0.0
        data_len = 0
        for batch in tqdm(self.data_loader):
            data_len += len(batch)
            x_train, y_train = batch
            optimizer.zero_grad()
            loss = model.loss(x_train.to(self.args.device), y_train.to(self.args.device)).float()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / data_len

class ValidProcessor():
    def __init__(self, args, data) -> None:
        self.args = args
        self.batch_size = 4
        self.dataset = ValidDataset(args, data)
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def process_epoch(self, model):
        model.eval()
        total_accuracy = 0.0
        val_data_len = 0
        # # for example:
        # for batch in self.data_loader:
        #     x_valid, y_valid = batch
        #     y_pred = model.predict(x_valid.to(self.args.device))
        #     rights = 0
        #     for a, b in zip(y_valid.detach().cpu().numpy(), y_pred.detach().cpu().numpy() > 0.5):
        #         if a and b:
        #             rights += 1
        #         if not a and not b:
        #             rights += 1
        #     total_accuracy += rights
        #     val_data_len += len(y_pred)
        # print(total_accuracy)
        # print(val_data_len)
        acc_accuracy = total_accuracy / val_data_len
        return acc_accuracy


class TestProcessor():
    def __init__(self, args, data) -> None:
        self.args = args
        self.batch_size = 4
        self.dataset = TestDataset(args, data)
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def process_epoch(self, model):
        model.eval()
        total_accuracy = 0.0
        val_data_len = 0
        for batch in tqdm(self.data_loader):
            x_test, y_test = batch
            y_pred = model.predict(x_test.to(self.args.device))
            rights = 0
            for a, b in zip(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy() > 0.5):
                if a and b:
                    rights += 1
                if not a and not b:
                    rights += 1
            total_accuracy += rights
            val_data_len += len(y_pred)
        acc_accuracy = total_accuracy / val_data_len
        return acc_accuracy