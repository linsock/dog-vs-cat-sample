import pickle
import numpy as np
import os
import random
import shutil
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# constants for the alg
DS_SIZE = 99999
DS_SEP = "."
DS = "dogs-vs-cats"
TRAIN = "train"
OUT = "output"
BATCH_SIZE = 32
LABELS = {"cat": 0, "dog": 1}
FEATURES = "{}_{}_features.csv"
FEATURE_SEP = ","
LBLE = "label_encoder.cpickle"
TTV = ("train", "test", "val")
STORE = "store"
MODEL_NAME = "lr_tl_dvc_model.cpickle"


def path(*args: list, sep: str=os.sep):
    """
    :param args: list, variable number of args to be joined into a path
    :param sep: string [default=os.sep], separator used to join path
    :return:
    """
    return sep.join(args)


def list_objects(data_set_path: str, return_dirs: bool=False):
    """
    :param data_set_path: string, directory to list
    :param return_dirs: boolean, output directories list
    :return: [paths] OR ([paths],[dirs])
    """
    out = None
    directories = set()
    for (root, dirs, files) in os.walk(data_set_path):
        if return_dirs and len(dirs) != 0:
            for d in dirs:
                directories.add(d)
        if len(files) != 0:
            for f in files:
                yield path(root, f)
    if return_dirs:
        out = directories
    return out


def get_labels(image_paths: list):
    """
    :param image_paths: list, contains strings of images' path
    :return: generator of list of labels extracted from images' name
    """
    for i in image_paths:
        yield i.split(os.sep)[-2]


# split single training DS into train/test/val
def organize_ds(data_set_path: str=path(DS, TRAIN), train_test_val: bool=True, ratio: tuple=(6, 2, 2), name_split: str=DS_SEP):
    """
    :param data_set_path: string, directory to list
    :param train_test_val: boolean, split dataset into train/test/validate
    :param ratio: tuple, used only if train_test_val is True; Determine ratio % of split; DEFAULT = (6, 2, 2)
    :param name_split: string, file name separator; DEFAULT = DS_SEP equal_to "."
    :return: None, just organize files/folders
    """
    objects = dict()
    dir_to_make = set()
    for (root, dirs, files) in os.walk(data_set_path):
        for f in files:
            lab = f.split(name_split)
            label = lab[0]
            if objects.get(label) is None:
                objects[label] = []
            fin = path(root, f)
            # gather class-separated images in different lists
            objects[label].append(fin)
    if train_test_val:
        # make dir names
        for split in TTV:
            for k in LABELS.keys():
                p = path(DS, OUT, split, k)
                dir_to_make.add(p)
        # create directories with LABEL only images
        for d in dir_to_make:
            os.makedirs(d, exist_ok=True)
        # count splits
        ratio_tot = float(sum(ratio))
        for cl, objs in objects.items():
            train_len = round(len(objs) * (ratio[0] / ratio_tot))
            test_len = round(len(objs) * (ratio[1] / ratio_tot))
            val_len = round(len(objs) * (ratio[2] / ratio_tot))
            print("### SPLIT:\n\tTrain: %s test: %s Val: %s" % (train_len, test_len, val_len))
            # take care of possible length difference in division
            train_len += (len(objs) - round(train_len) - round(test_len) - round(val_len))
            # match the original length
            assert (train_len + test_len + val_len == len(objs))
            # immutable split list
            split_ratio = (train_len, test_len, val_len)
            # shuffle input images
            random.shuffle(objs)
            # split train_test_val
            for ub, split in zip(split_ratio, TTV):
                for idx in range(1, ub + 1):
                    img = objs.pop(0)
                    # i.e. dogs-vs-cats/output/[train,test,val]/[cat, dog]
                    p = path(DS, OUT, split, cl)
                    print("[%s] Elaborating path: \n\t from: %s\n\t to: %s" % (idx, img, p))
                    shutil.copy2(img, p)
    else:
        for k in LABELS.keys():
            p = path(DS, OUT, TRAIN, k)
            os.makedirs(p, exist_ok=True)
            if objects.get(k) is None:
                continue
            for obj in objects[k]:
                print("Elaborating path: \n\t from: %s\n\t to: %s" % (obj, p))
                shutil.copy2(obj, p)


def extract_features(data_set_path: str=path(DS, OUT)):
    print("Extracting features - VGG16(no top)")
    model = VGG16(weights="imagenet", include_top=False)
    classes = set(LABELS.keys())
    le = None
    for split in TTV:
        ip = list(list_objects(data_set_path=path(data_set_path, split)))
        print("SPLIT: %s - #IMGS %s" % (str(split), str(len(ip))))
        if le is None:
            le = LabelEncoder()
            le.fit(list(classes))
        fp = open(path(DS, OUT, FEATURES.format(split, "_".join(LABELS.keys()))), "w")
        tot = len(ip) // BATCH_SIZE
        for (batch, i) in enumerate(range(0, len(ip), BATCH_SIZE)):
            if len(ip) - (tot * BATCH_SIZE) == 0:
                tot += 1
            print("\tElaborating batch %s/%s" % (batch, tot))
            batch_paths = ip[i:i + BATCH_SIZE]
            batch_labels = le.transform(list(get_labels(ip[i:i + BATCH_SIZE])))
            batch_images = []
            for img in batch_paths:
                img = np.expand_dims(img_to_array(load_img(img, target_size=(224, 224))), axis=0)
                batch_images.append(imagenet_utils.preprocess_input(img))
            batch_images = np.vstack(batch_images)
            features = model.predict(batch_images, batch_size=BATCH_SIZE)
            features = features.reshape(features.shape[0], 7 * 7 * 512)
            for (lab, vec) in zip(batch_labels, features):
                vec = ",".join([str(v) for v in vec])
                fp.write("{}, {}\n".format(lab, vec))
        fp.close()
    os.makedirs(path(DS, OUT, STORE), exist_ok=True)
    with open(path(DS, OUT, STORE, LBLE), "wb") as fp:
        pickle.dump(le, fp)


def check_feature_exists(train: bool=True, test: bool=True, val: bool=True):
    """
    :param train: boolean [default=True], check if train features exists
    :param test: boolean [default=True], check if test features exists
    :param val: boolean [default=True], check if val features exists
    :return: generator of boolean list foreach split existence
    """
    to_check = [train * 1, test * 2, val * 3]
    for split in to_check:
        if split == 0:
            continue
        split -= 1
        yield os.path.exists(path(DS, OUT, FEATURES.format(TTV[split], "_".join(LABELS.keys()))))


def __load_features(path: str):
    """
    :param path: string, path of file containing generated features
    :return: tuple (data, labels)
    """
    data = []
    labels = []
    if not os.path.exists(path):
        raise FileNotFoundError("Input path doesn't point to a valid file")
    for row in open(path, mode="r"):
        split = row.split(FEATURE_SEP)
        labels.append(split[0])
        data.append(np.array(split[1:], dtype="float"))
    return data, labels


def train_model():
    """
    :return: LogisticRegression classifier
    """
    # load generated features and le
    print("[Training Model]")
    print("Loading features TRAINING data...")
    (X_train, Y_train) = __load_features(path(DS, OUT, FEATURES.format(TTV[0], "_".join(LABELS.keys()))))
    print("Loading features TEST data...")
    (X_test, Y_test) = __load_features(path(DS, OUT, FEATURES.format(TTV[1], "_".join(LABELS.keys()))))
    print("Loading Label Encoder...")
    le = pickle.load(open(path(DS, OUT, STORE, LBLE), "rb"))
    # fit model
    print("Creating Logistic Regression model...")
    model = LogisticRegression(solver="lbfgs", multi_class="auto")
    print("Fitting data to model...")
    model.fit(X_train, Y_train)
    # test model
    print("[Testing Model]")
    print("Generating predictions...")
    predictions = model.predict(X_test)
    print("Generating report...")
    print(classification_report(Y_test, predictions, target_names=le.classes_))
    return model


def save_model(model: LogisticRegression, name: str=MODEL_NAME, where: str=path(DS, OUT, STORE)):
    """
    :param model: model, contains the model you want to store
    :param name: string, the name you wish to call the model
    :param where: string, where do you want to save your model
    :return: boolean, result of model save
    """
    model_saved = True
    try:
        with open(path(where, name), "wb") as fp:
            pickle.dump(model, fp)
    except IOError:
        model_saved = False
    return model_saved


def load_model(name: str=MODEL_NAME, where: str=path(DS, OUT, STORE)):
    """
    :param name: string, the name used to call the model
    :param where: string, where do you want to retrieve your model
    :return: model, model object loaded or Exception occurred (to be verified at loading time)
    """
    model = None
    try:
        with open(path(where, name), "rb") as fp:
            model = pickle.load(fp)
    except IOError as ex:
        model = ex
    return model


def test_model(model: LogisticRegression):
    print("Loading Label Encoder...")
    le = pickle.load(open(path(DS, OUT, STORE, LBLE), "rb"))
    print("Loading features VAL data...")
    (X_val, Y_val) = __load_features(path(DS, OUT, FEATURES.format(TTV[2], "_".join(LABELS.keys()))))
    print("[Test2 Model]")
    print("Generating predictions...")
    predictions = model.predict(X_val)
    print("Generating report...")
    print(classification_report(Y_val, predictions, target_names=le.classes_))