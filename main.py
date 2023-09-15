from src.utils import *
from src.parse_args import args
from src.data_load.MyData import MyData
from src.model.MyModel import MyModel
from src.train import Trainer
from src.test import Tester

class Instructor():
    def __init__(self) -> None:
        """ 1. Set parameters, seeds, logger, paths and device"""
        """ Set parameters """
        self.args = args
        """ Set seeds """
        set_seeds(self.args.seed)
        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        now_time = get_datetime()
        self.args.log_path += now_time
        logging_file_name = f'{self.args.log_path}.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger
        """ Set paths """
        if os.path.exists(self.args.save_path):
            shutil.rmtree(self.args.save_path, True)
        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)
        """ Set device """
        torch.cuda.set_device(int(self.args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device
        print(self.args.device)
        """ 2. Define data """
        self.data = MyData(self.args)
        """ 3. Define model """
        self.model, self.optimizer = self.create_model()
        self.args.logger.info(self.args)

    def create_model(self):
        model = MyModel(self.args)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        # optimizer = Adam([
        #     {"params": model.xxx.parameters(), "lr": self.args.lr_1},
        #     {"params": model.yyy.parameters(), "lr": self.args.lr_2},
        # ])
        return model, optimizer

    def save_model(self, is_best=False):
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch
        checkpoint_path = os.path.join(
            self.args.save_path,
            f'checkpoint-{self.args.epoch}.tar'
        )
        torch.save(checkpoint_dict, checkpoint_path)
        if is_best:
            best_path = os.path.join(
                self.args.save_path,
                f"model_best.tar"
            )
            shutil.copyfile(checkpoint_path, best_path)

    def load_checkpoint(self, best_path):
        if os.path.isfile(best_path):
            logging.info(f"=> loading checkpoint {best_path}")
            checkpoint = torch.load(
                best_path, map_location=f"cuda:{self.args.gpu}"
            )
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            logging.info(f"=> No checking found at {best_path}")

    def train(self):
        print("Begin Training")
        self.best_valid = 0.0
        trainer = Trainer(self.args, self.data, self.model, self.optimizer)
        for epoch in range(int(self.args.num_epoch)):
            self.args.epoch = epoch
            total_loss, valid_res = trainer.run_epoch()
            if valid_res > self.best_valid:
                self.best_valid = valid_res
                self.save_model(is_best=True)
            else:
                self.save_model()
            if epoch % 1 == 0:
                self.args.logger.info(
                    f'Epoch: {self.args.epoch} Loss: {total_loss} Acc: {valid_res} Best Acc: {self.best_valid}'
                )

    def test(self):
        print("Begin Testing")
        best_checkpoint_path = f"model_best.tar"
        best_path = os.path.join(
            self.args.save_path, best_checkpoint_path
        )
        self.load_checkpoint(best_path)
        tester = Tester(self.args, self.data, self.model)
        res = tester.run_epoch()
        self.args.logger.info(f"Test Results: Acc: {res}")

    def run(self):
        """ 1. Train the model. """
        self.train()
        """ 2. Test the model. """
        self.test()

if __name__ == "__main__":
    ins = Instructor(args)
    ins.run()