import yagmail
import tensorflow as tf
from tensorflow import keras


class EmailCallback(keras.callbacks.Callback):
    message_contents = ""
    yag: yagmail.SMTP
    
    def __init__(self, to: str="simon376@gmail.com", train_size=None, val_size=None, wait_for_train_end=False) -> None:
        super().__init__()
        print("setting up yagmail...")
        self.yag = yagmail.SMTP("simon376@gmail.com", oauth2_file="~/client_secret_1098025119208-cfsgjrjvin482rsin7otp0sckch691p3.apps.googleusercontent.com.json")
        self.to = to
        if train_size is not None:
            self.message_contents += f"size of train dataset: {train_size}\n"
        if val_size is not None:
            self.message_contents += f"size of validation dataset: {val_size}\n"
        self.wait_for_train_end = wait_for_train_end


    def on_train_begin(self, logs=None):
        msg = f"Start Training with Model {self.model.name}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n---\n")
        print(msg)

    def on_train_end(self, logs=None):
        msg = f"Stop training.\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n---\n")
        print(msg)
        if not self.wait_for_train_end:
            print("sendming e-mail...")
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

    def on_epoch_end(self, epoch, logs=None):
        log_keys = [f"{k}: {v:7.2f}" for k,v in logs.items() if k != 'loss']
        msg = f"The average loss for epoch {epoch} is {logs['loss']:7.2f} with the following metrics: {log_keys}."
        self.message_contents += (msg + "\n")


    def send_message(self):
        import datetime
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        import os
        import shutil


        res = ""
        temp_dir = "./temp/"
        os.mkdir(temp_dir)
        fn = os.path.join(temp_dir, "plot.png")
        keras.utils.plot_model(
            self.model, 
            to_file=fn, 
            show_shapes=True, 
            rankdir="LR")
    
        res = self.yag.send(
            to=self.to, 
            subject=f"TensorFlow Training Callback {self.model.name} {date}", 
            contents=self.message_contents,
            attachments=fn)
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
