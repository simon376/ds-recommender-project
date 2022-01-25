import yagmail
setup_called = False

def setup():
    global setup_called
    if setup_called:
        return
    yag = yagmail.SMTP("simon376@gmail.com", oauth2_file="~/client_secret_1098025119208-cfsgjrjvin482rsin7otp0sckch691p3.apps.googleusercontent.com.json")
    yag.send(subject="Great!")
    setup_called = True

import tensorflow as tf
from tensorflow import keras


class EmailCallback(keras.callbacks.Callback):
    message_contents = ""
    yag: yagmail.SMTP
    
    def __init__(self, to: str="simon376@gmail.com", train_size=None, test_size=None) -> None:
        super().__init__()
        print("setting up yagmail...")
        self.yag = yagmail.SMTP("simon376@gmail.com", oauth2_file="~/client_secret_1098025119208-cfsgjrjvin482rsin7otp0sckch691p3.apps.googleusercontent.com.json")
        self.to = to
        if train_size is not None:
            self.message_contents += f"size of train dataset: {train_size}\n"
        if test_size is not None:
            self.message_contents += f"size of test dataset: {test_size}\n"


    def on_train_begin(self, logs=None):
        msg = "Start Training with Model...\n"
        # msg += self.get_model_summary(self.model)
        keys = str(list(logs.keys()))
        msg += f"Starting training; got log keys: {keys}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n\n---\n")
        print(msg)

    def on_train_end(self, logs=None):
        keys = str(list(logs.keys()))
        msg = f"Stop training; got log keys: {(keys)}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n\n---\n")
        print(msg)

    def on_test_begin(self, logs=None):
        keys = str(list(logs.keys()))
        msg = f"Starting testing; got log keys: {keys}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n\n---\n")
        print(msg)

    def on_test_end(self, logs=None):
        keys = str(list(logs.keys()))
        msg = f"stop testing; got log keys: {keys}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n\n---\n")
        print(msg)
        print("sending e-mail..")
        self.send_message()

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
            subject=f"TensorFlow Training Callback {date}", 
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
