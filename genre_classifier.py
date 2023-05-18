from abc import abstractmethod
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import librosa
import json
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(123)

def plot_stft_and_log_mel_spectrogram(audio_data, label, sample_rate=22050, hop_length=512 // 4, n_fft=512, n_mels=80):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 6), gridspec_kw={'hspace': 0.3})
    ax.set(title=label)
    ax.label_outer()
    log_mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=n_fft,
                                                         hop_length=hop_length, n_mels=n_mels)
    mel_spect = librosa.power_to_db(log_mel_spectrogram, ref=np.max)
    fig.colorbar(librosa.display.specshow(mel_spect, hop_length=hop_length, sr=sample_rate, x_axis='time', y_axis='mel',
                                          ax=ax, cmap='magma'), ax=ax)
    plt.show()


def plot_spectral_center(audio_data, label, sample_rate=22050):
    # spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # Normalising the spectral centroid for visualisation
    # Plotting the Spectral Centroid along the waveform
    plt.plot(t, spectral_centroids, color='r')
    plt.title(label)
    plt.show()


class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 300
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.00008


class MusicClassifier:
    """
    Logistic regression classifier for music genre classification.
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.opt_params = opt_params
        ## init Logistic regression weights and bias
        self.num_features = 33
        self.first_class_weights = torch.zeros(self.num_features, requires_grad=False)
        self.second_class_weights = torch.zeros(self.num_features, requires_grad=False)
        self.third_class_weights = torch.zeros(self.num_features, requires_grad=False)  # 3 classes

    def set_weights(self, w1, w2, w3) -> None:
        """
        this function sets the weights of the logistic regression model.
        """
        self.first_class_weights = w1
        self.second_class_weights = w2
        self.third_class_weights = w3

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        feats = torch.zeros(wavs.shape[0], self.num_features)
        for i, wav in enumerate(wavs):
            audio_np = wav.squeeze().numpy()
            mfcc = librosa.feature.mfcc(y=audio_np, n_mfcc=self.num_features - 3)
            zero_crossings = librosa.zero_crossings(audio_np, pad=False)
            zero_crossing_rate = np.mean(zero_crossings)
            mfcc_mean = np.mean(mfcc, axis=1)
            cur_feats = np.append(mfcc_mean, zero_crossing_rate)
            rms_mean = np.mean(librosa.feature.rms(y=audio_np))
            cur_feats = np.append(cur_feats, rms_mean)
            cur_feats = np.append(cur_feats, 1)  # bias
            # put the features in the feats tensor
            feats[i] = torch.tensor(cur_feats)
        return feats

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass through logistic regression model, outputting scores for every class of.
        the three classes
        feats: batch of extracted features
        """
        scores = torch.sigmoid(torch.matmul(feats, self.first_class_weights))
        scores = torch.cat((scores.unsqueeze(1),
                            torch.sigmoid(
                                torch.matmul(feats, self.second_class_weights)).unsqueeze(1),
                            torch.sigmoid(
                                torch.matmul(feats, self.third_class_weights)).unsqueeze(1)),
                           dim=1)
        return scores

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence 
        OptimizationParameters are passed to the initialization function
        """
        first_class_labels = torch.zeros_like(labels)
        second_class_labels = torch.zeros_like(labels)
        third_class_labels = torch.zeros_like(labels)
        first_class_labels[labels == float(Genre.CLASSICAL.value)] = 1
        second_class_labels[labels == float(Genre.HEAVY_ROCK.value)] = 1
        third_class_labels[labels == float(Genre.REGGAE.value)] = 1
        self.first_class_weights -= self.opt_params.learning_rate * (1 / feats.shape[0]) * (
            torch.matmul(output_scores[:, 0].squeeze(dim=-1) - first_class_labels.squeeze(dim=-1), feats))
        self.second_class_weights -= self.opt_params.learning_rate * (1 / feats.shape[0]) * torch.matmul(
            output_scores[:, 1].squeeze(dim=-1) - second_class_labels.squeeze(dim=-1), feats)
        self.third_class_weights -= self.opt_params.learning_rate * (1 / feats.shape[0]) * torch.matmul(
            output_scores[:, 2].squeeze(dim=-1) - third_class_labels.squeeze(dim=-1), feats)

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        return (
            torch.tensor(
                [self.first_class_weights[:-1], self.second_class_weights[:-1], self.third_class_weights[:-1]]),
            torch.tensor([self.first_class_weights[-1], self.second_class_weights[-1], self.third_class_weights[-1]]))

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should receive a torch.Tensor of shape [batch, channels, time] (float tensor)
        and an output batch of corresponding labels [B, 1] (integer tensor)
        """
        wavs = wavs.squeeze(1)
        feats = self.exctract_feats(wavs)
        scores = self.forward(feats)
        return torch.argmax(scores, dim=1).unsqueeze(1)


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        # Opening JSON file
        module = MusicClassifier(OptimizationParameters())
        train_loader = ClassifierHandler.load_train_set_in_batches(training_parameters, module)
        test_data, test_labels = ClassifierHandler.load_test_set(training_parameters)
        for epoch in range(training_parameters.num_epochs):
            for batch in train_loader:
                feats = batch[0]
                labels = batch[1]
                scores = module.forward(feats)
                module.backward(feats, scores, labels)
        ClassifierHandler.compute_accuracy(test_data, test_labels, module)
        ClassifierHandler.print_tensor_of_weights_as_list(module.first_class_weights)
        ClassifierHandler.print_tensor_of_weights_as_list(module.second_class_weights)
        ClassifierHandler.print_tensor_of_weights_as_list(module.third_class_weights)
        return module

    @staticmethod
    def load_test_set(training_parameters: TrainingParameters):
        test_paths_file = open(training_parameters.test_json_path)
        test_paths_dict = json.load(test_paths_file)
        test_data = torch.tensor([])
        test_labels = torch.tensor([])
        random.shuffle(test_paths_dict)
        for inner_dict in test_paths_dict:
            audio, sr = librosa.load(inner_dict['path'])
            cur_data_torch = torch.tensor(audio).unsqueeze(0)
            label = Genre[inner_dict['label'].upper().replace('-', '_')]
            cur_labels_torch = torch.tensor([float(label.value)]).unsqueeze(0)
            test_data = torch.cat((test_data, cur_data_torch.clone()))
            test_labels = torch.cat((test_labels, cur_labels_torch.clone()))
        print("finished loading test data")
        # Closing file
        test_paths_file.close()
        return test_data, test_labels

    @staticmethod
    def load_train_set_in_batches(training_parameters: TrainingParameters, module):
        train_paths_file = open(training_parameters.train_json_path)
        train_paths_dict = json.load(train_paths_file)
        train_loader = list()
        batch_counter = 0
        random.shuffle(train_paths_dict)
        cur_data_torch = torch.tensor([])
        cur_labels_torch = torch.tensor([])
        for inner_dict in train_paths_dict:
            batch_counter += 1
            audio, sr = librosa.load(inner_dict['path'])
            cur_data_torch = torch.cat((cur_data_torch, torch.tensor(audio).unsqueeze(0)))
            label = Genre[inner_dict['label'].upper().replace('-', '_')]
            cur_labels_torch = torch.cat((cur_labels_torch, torch.tensor([float(label.value)]).unsqueeze(0)))
            if batch_counter % training_parameters.batch_size == 0:
                train_loader.append([module.exctract_feats(cur_data_torch.clone()), cur_labels_torch.clone()])
                cur_data_torch = torch.tensor([])
                cur_labels_torch = torch.tensor([])
        train_paths_file.close()
        return train_loader

    @staticmethod
    def print_tensor_of_weights_as_list(tensor):
        print(", ".join([str(x) for x in tensor.tolist()]))

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyper-parameters and return the loaded model
        """
        module = MusicClassifier(OptimizationParameters())
        module.set_weights(torch.tensor([-0.011372833512723446, 0.020494626834988594, -0.06394007802009583, -0.12641745805740356,
                                         -0.030416306108236313, -0.09144598990678787, -0.036103568971157074,
                                         -0.03880093991756439, 0.006737630348652601, -0.038998037576675415, 0.010391443967819214,
                                         -0.022182587534189224, -0.007290207780897617, 0.00145872519351542, 0.01924142614006996,
                                         0.017280176281929016, 0.047399379312992096, 0.00899857934564352, -0.005552022252231836,
                                         0.02932639606297016, -0.007758110295981169, 0.015352180227637291,
                                         0.005761635955423117, 0.021408889442682266, -0.0013940792996436357, 0.019870156422257423,
                                         -0.002837357809767127, 0.03088158369064331, 0.006957559380680323, -0.005237983539700508,
                                         -0.00010570632730377838, -0.00036704339436255395, -0.001711581484414637]),
                           torch.tensor([0.029980603605508804, -0.03357984498143196, -0.028617775067687035, 0.09350784868001938,
                                         -0.003332694061100483, -0.02301018126308918, 0.02920541912317276, 0.005449625663459301,
                                         0.008561192080378532, 0.014836831949651241, -0.034509606659412384, 0.01626128889620304,
                                         -0.05597783252596855, -0.02782018668949604, -0.019905220717191696, -0.008135818876326084,
                                         0.003259392688050866, -0.0390542633831501, 0.011196278035640717, -0.008717233315110207,
                                         -0.016262013465166092, -0.018635433167219162, -0.006841812748461962, -0.025590455159544945,
                                         -0.016384687274694443, -0.041856374591588974, -0.025775324553251266, -0.0010226838057860732,
                                         0.000834767590276897, -0.0024862566497176886, -0.00010015319276135415, -0.0007900249911472201,
                                         -0.0007668042671866715]),
                           torch.tensor([0.0033042789436876774, 0.002013015327975154, 0.04173900932073593, -0.013068262487649918,
                                         0.04679708927869797, 0.1103283166885376, 0.028105955570936203, 0.018613619729876518,
                                         -0.00980818085372448, 0.016670173034071922, 0.04528491571545601, -0.03309603035449982,
                                         0.08379297703504562, 0.011602255515754223, 0.04151991382241249, -0.03373072296380997,
                                         -0.08778582513332367, 0.04505482688546181, -0.0528569370508194, -0.028757384046912193,
                                         0.03602603077888489, -0.012033781036734581, -0.03611057996749878, 0.017053689807653427,
                                         0.008144245482981205, 0.0343603678047657, 0.022465474903583527, -0.02168974280357361,
                                         -0.028664376586675644, 0.002392763737589121, -0.0002478927781339735, 0.0005581853911280632,
                                         0.003654906991869211]))
        return module

    @staticmethod
    def compute_accuracy(wavs: torch.Tensor, labels: torch.Tensor, module: MusicClassifier):
        """
        computes the accuracy of each class
        """
        wavs = wavs.squeeze(1)
        output_labels = module.classify(wavs)
        first_class_acc = torch.sum(
            (output_labels == float(Genre.CLASSICAL.value)) & (labels == float(Genre.CLASSICAL.value))) / torch.sum(
            labels == float(Genre.CLASSICAL.value))
        second_class_acc = torch.sum(
            (output_labels == float(Genre.HEAVY_ROCK.value)) & (labels == float(Genre.HEAVY_ROCK.value))) / torch.sum(
            labels == float(Genre.HEAVY_ROCK.value))
        third_class_acc = torch.sum(
            (output_labels == float(Genre.REGGAE.value)) & (labels == float(Genre.REGGAE.value))) / torch.sum(
            labels == float(Genre.REGGAE.value))
        print("classical accuracy: ", first_class_acc.item())
        print("hard rock class accuracy: ", second_class_acc.item())
        print("reggae class accuracy: ", third_class_acc.item())
        total_acc = torch.sum(output_labels == labels).item() / labels.shape[0]
        print("total accuracy: ", total_acc)
        return total_acc

ClassifierHandler.train_new_model(TrainingParameters())

