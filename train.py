import os
import gc
from datetime import datetime
import tensorflow as tf
from models.networks import Pnet, Rnet, Onet
from tf_keras import optimizers
import tf_keras
from argparse import ArgumentParser
from tqdm import tqdm
from utils.data_helper import parse_fn_pnet, parse_fn_rnet, parse_fn_onet, image_augmentation, img_normalize
from utils.summary_writer import SummaryWriter
from loss.ohdm import pnet_loss, rnet_loss, onet_loss

def time_stamp():
    now = datetime.now()
    daytime = now.strftime("%Y-%m-%d_%H:%M:%S")
    
    return daytime
    


def train_pnet(data_path, save_path, optimizer, epochs, batch_size, log_dir, init_weight_path):
    print("Training Pnet...")
    pnet = Pnet()
    pnet(tf.ones((1, 12, 12, 3), dtype=tf.float32))
    pnet.summary()
    if init_weight_path:
        pnet.load_weights(init_weight_path)

    stage = "pnet"

    dataloader, iters_per_epoch = load_data(data_path, batch_size, stage)

    logdir = f"{log_dir}/{stage}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir=logdir)

    savepath = f"{save_path}/{stage}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    
    train_loops(pnet, stage, save_path, dataloader, optimizer, epochs, iters_per_epoch, writer)

    now = time_stamp()
    print(f"Training of Pnet finished at {now}")



def train_rnet(data_path, save_path, optimizer, epochs, batch_size, log_dir, init_weight_path):
    print("Training Rnet...")
    rnet = Rnet()
    rnet(tf.ones((1, 24, 24, 3), dtype=tf.float32))
    rnet.summary()

    if init_weight_path:
        rnet.load_weights(init_weight_path)

    stage = "rnet"

    dataloader, iters_per_epoch = load_data(data_path, batch_size, stage)
    
    logdir = f"{log_dir}/{stage}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir=logdir)

    savepath = f"{save_path}/{stage}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    train_loops(rnet, stage, save_path, dataloader, optimizer, epochs, iters_per_epoch, writer)

    now = time_stamp()
    print(f"Training of Pnet finished at {now}")



def train_onet(data_path, save_path, optimizer, epochs, batch_size, log_dir, init_weight_path):
    print("Training Onet...")
    onet = Onet()
    onet(tf.ones((1, 48, 48, 3), dtype=tf.float32))
    onet.summary()
    
    if init_weight_path:
        onet.load_weights(init_weight_path)

    stage = "onet"

    dataloader, iters_per_epoch = load_data(data_path, batch_size, stage)
    
    logdir = f"{log_dir}/{stage}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir=logdir)

    savepath = f"{save_path}/{stage}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    train_loops(onet, stage, save_path, dataloader, optimizer, epochs, iters_per_epoch, writer)

    now = time_stamp()
    print(f"Training of Onet finished at {now}")



def load_data(tfrecord_path: str, batch_size: int, stage: str):
    parse_fns = {"pnet": parse_fn_pnet, "rnet": parse_fn_rnet, "onet": parse_fn_onet}
    parse_fn = parse_fns.get(stage)
    dataset = tf.data.TFRecordDataset(filenames=tfrecord_path)
    dataset = dataset.map(parse_fn)
    dataloader = dataset.shuffle(12800).batch(batch_size)
    
    iters_per_epoch = 0
    for _ in dataloader:
        iters_per_epoch += 1
    
    return dataloader, iters_per_epoch


def train_loops(
        model: Pnet|Rnet|Onet, 
        stage: str,  
        save_path: str,  
        data_loader: tf.data.Dataset,  
        optimizer: optimizers.Optimizer,  # Adam
        epochs: int, 
        iters_per_epoch: int, 
        writer: SummaryWriter
    ):
    loss_fns = {"pnet": pnet_loss, "rnet": rnet_loss, "onet": onet_loss}

    loss_fn = loss_fns.get(stage)

    for epoch in range(epochs):
        iters = epoch * iters_per_epoch
        i = 0
        for samples in tqdm(
            data_loader, desc=f"Training iteration {epoch + 1}", total=iters_per_epoch):
            images = samples.get("image")
            images = image_augmentation(images)
            images = img_normalize(images)
            labels = samples.get("label")
            bboxes = samples.get("bbox")
            landmarks = None
            landmarks_loc = None
            if stage == "onet":
                landmarks = samples.get("landmarks")
            try:
                with tf.GradientTape() as tape:
                    outputs = model(images)
                    face_cls = outputs[0]
                    bboxes_reg = outputs[1]
                    if stage == "onet":
                        landmarks_loc = outputs[2]
                        loss = loss_fn(face_cls, bboxes_reg, landmarks_loc, labels, bboxes, landmarks)
                    else:
                        loss = loss_fn(face_cls, bboxes_reg, labels, bboxes)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                i += 1
                
            except Exception as e:
                print(str(e))
                continue

            if i % 100 == 0:
                writer.add_scalar(f"{stage} loss", loss.numpy(), iters + i)
            
        model.save(f"{save_path}/{stage}/{epoch + 1}.keras")
        
        del images
        del labels
        del bboxes
        if landmarks is not None:
            del landmarks
        del samples
        del face_cls
        del bboxes_reg
        if landmarks_loc is not None:
            del landmarks_loc
        del outputs
        del grads
        tf.keras.backend.clear_session()
        tf_keras.backend.clear_session()
        gc.collect()



def train():
    parser = ArgumentParser(
        prog="MTCNN Trainer",
        description="Training three stages of MTCNN Face Detector"
    )

    parser.add_argument("--stage", required=True, type=str, help="Specify which stage to train")
    
    parser.add_argument("--data_path", required=True, type=str, help="Specify the direction of training data")
    
    parser.add_argument("--save_path", required=True, type=str, help="Specify the directory to save the trained model weight")
    
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-3, help="Specify the initial learning rate of optimizer")
    
    parser.add_argument("--optimizer", required=False, type=str, default="adam", help="Specify the optimizer to train networks")
    
    parser.add_argument("--epochs", required=False, type=int, default=5, help="Specify the training epochs")
    
    parser.add_argument("--batch_size",required=False, type=int, default=32, help="Specify the of training training mini-batch")
    
    parser.add_argument("--log_dir", required=False, type=str, default="logs", help="Specify log dir for tensorboard events")
    
    parser.add_argument("--seed", required=False, type=int, default="1394", help="Specify the random seed.")

    parser.add_argument("--init_weight_path", required=False, type=str, default="", help="Give the weight path to load and initial the model")


    args = parser.parse_args()
    stage = args.stage
    data_path = args.data_path
    save_path = args.save_path
    learning_rate = args.learning_rate
    optimizer_option = args.optimizer
    epochs = args.epochs
    batch_size = args.batch_size
    log_dir = args.log_dir
    seed = args.seed
    tf.random.set_seed(seed)
    init_weight_path = args.init_weight_path

    # TODO
    # Add support of other optimizers, apart from Adam Optimizer
    if optimizer_option != "adam":
        raise ValueError(f"Unrecognized value of optimizer: `{stage}`.\n For now only Adam Optimizer is supported.")
    
    optimizer = optimizers.Adam(learning_rate)


    if stage == "pnet":
        train_pnet(data_path, save_path, optimizer, epochs, batch_size, log_dir, init_weight_path)
    elif stage == "rnet":
        train_rnet(data_path, save_path, optimizer, epochs, batch_size, log_dir, init_weight_path)
    elif stage == "onet":
        train_onet(data_path, save_path, optimizer, epochs, batch_size, log_dir, init_weight_path)
    else:
        raise ValueError(f"Unrecognized value of stage: `{stage}`.\n The stage value must be `pnet`, `rnet` or `onet`.")
    

if __name__ == "__main__":
    train()