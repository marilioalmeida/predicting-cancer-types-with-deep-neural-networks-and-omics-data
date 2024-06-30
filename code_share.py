import code_share
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime as DT
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
import pandas as pd


TRAINING_SET_PATH = './program_1/output/Example_training_1228_TCGA_samples.npy'
TRAINING_LABEL_PATH = './program_1/output/Example_training_1228_TCGA_samples_label.npy'
OUT_FILE_NAME = './program_2/output/Example_training_1228_TCGA_samples'
TESTING_SET_PATH = './program_1/output/Example_validation_4908_TCGA_samples.npy'
TESTING_LABEL_PATH = './program_1/output/Example_validation_4908_TCGA_samples_label.npy'
FILENAME_LIST_PATH = './program_1/output/Example_validation_4908_TCGA_samples_title.npy'
INPUT_SAVE_MODEL_PATH = './program_2/output/Example_training_1228_TCGA_samples.weights.h5'
NUM_EPCHS = 100

# Custom metrics and helper functions
def TP(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    return K.sum(y_pos * y_pred_pos)

def TN(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_neg = 1 - K.round(K.clip(y_true, 0, 1))
    return K.sum(y_neg * (1 - y_pred_pos))

def FP(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_neg = 1 - K.round(K.clip(y_true, 0, 1))
    return K.sum(y_neg * y_pred_pos)

def FN(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    return K.sum(y_pos * (1 - y_pred_pos))

def MCC(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def save_train_history(train_history, train, validation, name):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(name + ".png", bbox_inches="tight")
    plt.close()

class Metrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        print("F1 score: ", _val_f1, " - Recall: ", _val_recall, " - Precision: ", _val_precision)
        logs['val_f1s'] = _val_f1
        logs['val_recalls'] = _val_recall
        logs['val_precisions'] = _val_precision
        return


def evaluate_model(model, name, test_sample, test_sample_title, test_label, test_label_compare):
    # Verifica e cria o diretório se não existir
    result_dir = f"./result/{name}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    time = DT.now()
    time = f"{time.year}{time.month}{time.day}{time.hour}{time.minute}"
    result_file_path = os.path.join(result_dir, f"{name}_{os.path.splitext(os.path.basename(TESTING_SET_PATH))[0]}_predicted_result_{time}")

    with open(result_file_path, "w") as out:        
        print(f"Training model from {INPUT_SAVE_MODEL_PATH}", file=out)
        print(f"Independent set from {TESTING_SET_PATH}", file=out)
      
        try:
            model.load_weights(INPUT_SAVE_MODEL_PATH)
            print("Successfully loading previous BEST training weights.", file=out)
        except:
            print("Failed to load previous data, training new model below.", file=out)
        
        adam = Adam(learning_rate=1e-4)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', MCC])
        
        # Evaluate model
        scores = model.evaluate(test_sample, test_label, verbose=1)
        print(f"Independent test:\tAccuracy\t{scores[1]:.3f}\tMCC\t{scores[2]:.3f}\n", file=out)
        
        # Predictions
        predictions = model.predict(test_sample)
        Prediction = predictions.argmax(axis=-1)
        
        num = 0
        start = 1
        print("Predict-result:", file=out)
        for result in Prediction:
            label_from_title = str(test_sample_title[num]).split("-")
            if int(label_from_title[2]) != result:
                print(f"{start}\tSample Name:\t{test_sample_title[num]}\tActual:\t{int(label_from_title[2])}\tPredict:\t{result}", file=out)
                start += 1
            num += 1
        
        # Confusion matrix
        confusion_matrix = pd.crosstab(test_label_compare, Prediction, rownames=['Actual'], colnames=['Predicted'], margins=True)
        print(f"\n\nConfusion Matrix:\n{confusion_matrix}", file=out)
        print(f"\n\nConfusion Matrix:\n{confusion_matrix}")
