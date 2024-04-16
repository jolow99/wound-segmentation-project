import cv2 
import json
import matplotlib.pyplot as plt
import numpy as np 
import os 

np.random.seed(42)

def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


def get_png_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".png" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def get_jpg_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".jpg" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def load_jpg_images(path):
    file_list = get_jpg_filename_list(path)
    temp_list = []
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    # x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return temp_list, file_list


def load_png_images(path):

    temp_list = []
    file_list = get_png_filename_list(path)
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    #temp_list = np.reshape(temp_list,(temp_list.shape[0], temp_list.shape[1], temp_list.shape[2], 3))
    return temp_list, file_list


def load_data(path):
    # path_train_images = path + "train/images/padded/"
    # path_train_labels = path + "train/labels/padded/"
    # path_test_images = path + "test/images/padded/"
    # path_test_labels = path + "test/labels/padded/"
    path_train_images = path + "train/images/"
    path_train_labels = path + "train/labels/"
    path_test_images = path + "test/images/"
    path_test_labels = path + "test/labels/"
    x_train, train_image_filenames_list = load_png_images(path_train_images)
    y_train, train_label_filenames_list = load_png_images(path_train_labels)
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    y_test, test_label_filenames_list = load_png_images(path_test_labels)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    x_test = normalize(x_test)
    y_test = normalize(y_test)
    return x_train, y_train, x_test, y_test, test_label_filenames_list


def load_test_images(path):
    path_test_images = path + "test/images/"
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    x_test = normalize(x_test)
    return x_test, test_image_filenames_list


def save_results(np_array, color_space, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        pred = np_array[i]
        # if color_space.lower() is 'hsi' or 'hsv':
        #     pred = cv2.cvtColor(pred, cv2.COLOR_HSV2RGB)
        # elif color_space.lower() is 'lab':
        #     pred = cv2.cvtColor(pred, cv2.COLOR_Lab2RGB)
        cv2.imwrite(outpath + filename, pred * 255.)
        i += 1


def save_rgb_results(np_array, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        cv2.imwrite(outpath + filename, np_array[i] * 255.)
        i += 1


def save_history(model, model_name, training_history, dataset, n_filters, epoch, learning_rate, loss,
                 color_space, path=None, temp_name=None):
    save_weight_filename = temp_name if temp_name else str(datetime.datetime.now())
    model.save('{}{}.hdf5'.format(path, save_weight_filename))
    with open('{}{}.json'.format(path, save_weight_filename), 'w') as f:
        json.dump(training_history.history, f, indent=2)

    json_list = ['{}{}.json'.format(path, save_weight_filename)]
    for json_filename in json_list:
        with open(json_filename) as f:
            # convert the loss json object to a python dict
            loss_dict = json.load(f)
        print_list = ['loss', 'val_loss', 'dice_coef', 'val_dice_coef']
        for item in print_list:
            item_list = []
            if item in loss_dict:
                item_list.extend(loss_dict.get(item))
                plt.plot(item_list)
        plt.title('model:{} lr:{} epoch:{} #filtr:{} Colorspaces:{}'.format(model_name, learning_rate,
                                                                            epoch, n_filters, color_space))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'test_loss', 'train_dice', 'test_dice'], loc='upper left')
        plt.savefig('{}{}.png'.format(path, save_weight_filename))
        plt.show()
        plt.clf()


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*):
    """

    def __init__(
        self,
        predictions,
        label_ids,
        inputs
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs