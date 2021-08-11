import os
import xgboost as xgb

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import ray
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from core.dataset import SpokenDigitDataset
from core.utils import getFeatures, currtime

def get_train_test_dmatrix(input_size=3):
    # Load dataset
    print('Current working directory:', os.getcwd())
    image_root = '/data/acp20ww/secure-robotic-delivery-system/cnn_xgboost/data/melimages_55x55_png/'
    if input_size == 1:
        image_root = '/data/acp20ww/secure-robotic-delivery-system/cnn_xgboost/data/melimages/'

    if os.path.isfile(os.path.join(image_root, 'train.buffer')):
        print('Load cached train and test data')
        dtrain = xgb.DMatrix(os.path.join(image_root, 'train.buffer'))
        dtest  = xgb.DMatrix(os.path.join(image_root, 'test.buffer'))
        return dtrain, dtest 

    meldset = SpokenDigitDataset(
        '/data/acp20ww/secure-robotic-delivery-system/cnn_xgboost/data/speech_commands_v0.01/spoken_digit.csv', 
        image_root, transforms.ToTensor())
    
    size = len(meldset)
    val_size = int(0.2 * size)
    train_size = size - val_size 
    
    # Split into train and test set
    generator=torch.Generator().manual_seed(200207124)
    train_dset, test_dset = random_split(meldset, [train_size, val_size], generator=generator)
    print(currtime(), "Number of samples in train set: ", train_size)
    print(currtime(), "Number of samples in validation set: ", val_size)

    worker_number = 1
    batch_size = 64
    trainloader = DataLoader(train_dset, batch_size=batch_size, num_workers=worker_number)
    testloader = DataLoader(test_dset, batch_size=batch_size, num_workers=worker_number)

    # converrt features using cnn_xgboost model
    print(currtime(), "Load features ...")
    best_cnn_path = '/data/acp20ww/secure-robotic-delivery-system/cnn_xgboost/model/best/cnn{}/best.ckpt'.format(input_size)
    print(currtime(), "    CNN model path:", best_cnn_path)
    result = getFeatures(best_cnn_path, input_size, trainloader, testloader)
    (train_features, train_labels, test_features, test_labels) = result

    # Build input matrices for XGBoost
    dtrain = xgb.DMatrix(train_features, train_labels)
    dtest = xgb.DMatrix(test_features, test_labels)

    # cache
    dtrain.save_binary(os.path.join(image_root, 'train.buffer'))
    dtest.save_binary(os.path.join(image_root,  'test.buffer'))

    return dtrain, dtest


def train_spoken_digit_1(config: dict, checkpoint_dir=None):
    # Build input matrices for XGBoost
    train_set, test_set = get_train_test_dmatrix(input_size=1)

    # Train the classifier, using the Tune callback
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")]
    )

def train_spoken_digit_3(config: dict, checkpoint_dir=None):
    # Build input matrices for XGBoost
    train_set, test_set = get_train_test_dmatrix(input_size=3)

    # Train the classifier, using the Tune callback
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")]
    )


def get_best_model_checkpoint(analysis):
    best_xgb = xgb.Booster()
    best_xgb.load_model(os.path.join(analysis.best_checkpoint, "model.xgb"))
    accuracy = 1. - analysis.best_result["eval-merror"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_xgb


def tune_xgboost(input_size=3):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "multi:softmax",
        'num_class': 10,
        "eval_metric": ["merror", "mlogloss"],

        "max_depth": tune.randint(3, 11),
        "min_child_weight": tune.choice([0.1, 0.3, 1, 3, 10, 30, 100]),

        "subsample": tune.uniform(0.5, 1.0),
        'colsample_bytree': tune.uniform(0.1, 1.0),

        "eta": tune.loguniform(1e-4, 1e-1)
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=100,  # 100 training iterations
        grace_period=1,
        reduction_factor=4)

    analysis = tune.run(
        train_spoken_digit_3 if input_size == 3 else train_spoken_digit_1,
        metric="eval-merror",
        mode="min",
        # You can add "gpu": 0.1 to allocate GPUs
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=search_space,
        num_samples=300,
        queue_trials=True,
        scheduler=scheduler)

    return analysis


if __name__ == "__main__":
    import argparse
    # process args
    parser = argparse.ArgumentParser(description='CNN-XGBoost-Tuning')
    parser.add_argument('--num_cpus', default=2, type=int, metavar='N', help='avaliable cpu number')
    parser.add_argument('--num_gpus', default=2, type=int, metavar='N', help='avaliable gpu number')
    parser.add_argument('--input_size', default=3, type=int, metavar='N', help='input channel size')
    args = parser.parse_args()

    #https://docs.ray.io/en/releases-0.8.5/package-ref.html#ray.init
    # bytes_per_gb = 1*1024*1024*1024
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    print(">>>>> Cluster resources:", ray.cluster_resources())
    
    analysis = tune_xgboost(input_size=args.input_size)

    # Load the best model checkpoint.    
    best_xgb = get_best_model_checkpoint(analysis)

    xgb_model_name = '/data/acp20ww/secure-robotic-delivery-system/cnn_xgboost/model/best/xgboost{}/xgb.model'.format(args.input_size)
    best_xgb.save_model(xgb_model_name)
    print('Save best XGBoost model to: ', xgb_model_name)

    # You could now do further predictions with
    # best_xgb.predict(...)







