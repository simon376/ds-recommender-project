import os
from matplotlib import pyplot as plt
import numpy as np
import yagmail
import tensorflow as tf
from tensorflow import keras
from typing import Optional
import datetime

class EmailCallback(keras.callbacks.Callback):
    message_contents = ""
    yag: yagmail.SMTP
    
    def __init__(self, to: Optional[str]=None, train_size=None, val_size=None, wait_for_train_end=False) -> None:
        super().__init__()
        print("setting up yagmail...")
        import credentials
        self.yag = yagmail.SMTP(credentials.username, credentials.app_password)
        self.to = to if to is not None else credentials.username
        if train_size is not None:
            self.message_contents += f"size of train dataset: {train_size}\n"
        if val_size is not None:
            self.message_contents += f"size of validation dataset: {val_size}\n"
        self.wait_for_train_end = wait_for_train_end
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []


    def on_train_begin(self, logs=None):
        msg = f"Start Training with Model {self.model.name}, wall clock time: {datetime.datetime.now()}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n---\n")
        print(msg)

    def on_train_end(self, logs=None):
        msg = f"Training finished, wall clock time: {datetime.datetime.now()}.\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n---\n")
        print(msg)


        N = np.arange(0, len(self.losses))

        # You can chose the style of your preference
        # print(plt.style.available) to see the available options
        plt.style.use("seaborn")

        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure()
        plt.plot(N, self.losses, label = "train_loss")
        plt.plot(N, self.acc, label = "train_acc")
        plt.plot(N, self.val_losses, label = "val_loss")
        plt.plot(N, self.val_acc, label = "val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        temp = self.get_temp_dir()
        plt.savefig(os.path.join(temp, f"{self.model.name}_histplot.png"))
        plt.close()


        if not self.wait_for_train_end:
            print("sending e-mail...")
            self.send_message()

    def on_test_begin(self, logs=None):
        msg = f"Starting testing with model {self.model.name}.\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n---\n")
        print(msg)

    def on_test_end(self, logs=None):
        msg = f"stop testing.\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n---\n")
        print(msg)
        if self.wait_for_train_end:
            print("sending e-mail..")
            self.send_message()

    def on_epoch_end(self, epoch, logs={}):
        log_keys = [f"{k}: {v:7.2f}" for k,v in logs.items() if k != 'loss']
        msg = f"The average loss for epoch {epoch} is {logs['loss']:7.2f} with the following metrics: {log_keys}."
        self.message_contents += (msg + "\n")
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

    @staticmethod
    def get_temp_dir():
        import os
        temp_dir = "./temp/"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        return temp_dir


    def send_message(self):
        import datetime
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        import os
        import shutil


        res = ""
        temp_dir = self.get_temp_dir()
        modelplot_fn = os.path.join(temp_dir, f"{self.model.name}_plot.png")
        histplot_fn = os.path.join(temp_dir, f"{self.model.name}_histplot.png")
        attachments = [histplot_fn, modelplot_fn] if os.path.exists(histplot_fn) else [modelplot_fn]
        keras.utils.plot_model(
            self.model, 
            to_file=modelplot_fn, 
            show_shapes=True, 
            rankdir="LR")
    
        res = self.yag.send(
            to=self.to, 
            subject=f"TensorFlow Training Callback {self.model.name} {date}", 
            contents=self.message_contents,
            attachments=attachments)

        shutil.rmtree(temp_dir)
        print("e-mail sent.")
        print(res)

    @staticmethod
    def get_model_summary(model):
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string


def send_hyperparameter_results_email(tuner, _to = None, ):
    import credentials
    yag = yagmail.SMTP(credentials.username, credentials.app_password)
    to = _to if _to is not None else credentials.username
    import datetime
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    import os
    import shutil

    summary = None
    model_name = "unknown model"
    try:
        # Get the top model.
        best_model= tuner.get_best_models()
        # Build the model.
        # Needed for `Sequential` without specified `input_shape`.
        best_model.build(input_shape=(None, 28, 28))
        summary = best_model.summary()
        model_name = best_model.name
    except:
        pass
    
    res = ""
    import os
    temp_dir = "./temp/"
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    contents = f"search space summary: {tuner.search_space_summary()}\n"
    contents += f"search results summary: {tuner.results_summary()}\n "
    if summary is not None:
        contents += (f"best model: {summary}\n---\n")
    try:
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        contents += f"very best hyperparameters: {best_hps}"
    except:
        pass

    res = yag.send(
        to=to, 
        subject=f"TensorFlow Training Callback {model_name} {date}", 
        contents=contents)

    shutil.rmtree(temp_dir)
    print("hyperparameter training results e-mail sent.")
    print(res)

