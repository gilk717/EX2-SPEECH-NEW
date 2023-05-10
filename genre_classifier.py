from abc import abstractmethod
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import librosa
import json
import numpy as np
import random

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
    num_epochs: int = 25
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.01


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
        self.num_features = 10
        self.first_class_weights = torch.zeros(self.num_features, requires_grad=False)
        self.second_class_weights = torch.zeros(self.num_features, requires_grad=False)
        self.third_class_weights = torch.zeros(self.num_features, requires_grad=False)

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        feats = torch.zeros(wavs.shape[0], self.num_features)
        ## compute mfcc into a colimn vector with shape (num_features,1)
        for i, wav in enumerate(wavs):
            audio_np = wav.squeeze().numpy()
            mfcc = librosa.feature.mfcc(y=audio_np)
            # put the features in the feats tensor
            feats[i] = torch.tensor([np.mean(mfcc), np.std(mfcc), np.max(mfcc), np.min(mfcc), np.median(mfcc), np.mean(mfcc[1:]), np.std(mfcc[1:]), np.max(mfcc[1:]), np.min(mfcc[1:]), np.median(mfcc[1:])])
        # for i,wav in enumerate(wavs):
        #     audio_np = wav.squeeze().numpy()
        #
        #     # Compute the amplitude envelope
        #     amplitude_envelope = np.abs(audio_np)
        #
        #     # Compute the energy
        #     energy = np.sum(audio_np ** 2)
        #
        #     # Compute the zero-crossing rate
        #     zero_crossings = librosa.zero_crossings(audio_np, pad=False)
        #     zero_crossing_rate = np.mean(zero_crossings)
        #     # put the features in the feats tensor
        #     feats[i] = torch.tensor([zero_crossing_rate , np.max(amplitude_envelope), np.mean(amplitude_envelope)])
        return feats

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass through logistic regression model, outputting scores for every class of.
        the three classes
        feats: batch of extracted faetures
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
        self.first_class_weights -= self.opt_params.learning_rate * (1 / feats.shape[0]) * (torch.matmul(output_scores[:, 0].squeeze(dim=-1) - first_class_labels.squeeze(dim=-1), feats))
        self.second_class_weights -= self.opt_params.learning_rate * (1 / feats.shape[0]) * torch.matmul(
            output_scores[:, 1].squeeze(dim=-1) - second_class_labels.squeeze(dim=-1), feats)
        self.third_class_weights -= self.opt_params.learning_rate * (1 / feats.shape[0]) * torch.matmul(
            output_scores[:, 2].squeeze(dim=-1) - third_class_labels.squeeze(dim=-1), feats)

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        raise NotImplementedError("function is not implemented")

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor) 
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        wavs = wavs.squeeze(1)
        feats = self.exctract_feats(wavs)
        scores = self.forward(feats)
        ## see shpae
        return torch.argmax(scores, dim=1).unsqueeze(1)


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        # Opening JSON file
        train_paths_file = open(training_parameters.train_json_path)
        train_paths_dict = json.load(train_paths_file)
        train_loader = list()
        batch_counter = 0
        cur_data_torch = torch.tensor([])
        cur_labels_torch = torch.tensor([])
        random.shuffle(train_paths_dict)
        for inner_dict in train_paths_dict:
            batch_counter += 1
            audio, cr = librosa.load(inner_dict['path'])
            cur_data_torch = torch.cat((cur_data_torch, torch.tensor(audio).unsqueeze(0)))
            label = Genre[inner_dict['label'].upper().replace('-', '_')]
            cur_labels_torch = torch.cat((cur_labels_torch, torch.tensor([float(label.value)]).unsqueeze(0)))
            if batch_counter % training_parameters.batch_size == 0:
                train_loader.append([cur_data_torch.clone(), cur_labels_torch.clone()])
                cur_data_torch = torch.tensor([])
                cur_labels_torch = torch.tensor([])
        print("finished loading data")
        # Closing file
        train_paths_file.close()
        ## print accuracy
        module = MusicClassifier(OptimizationParameters())
        for epoch in range(training_parameters.num_epochs):
            for batch in train_loader:
                feats = module.exctract_feats(batch[0])
                labels = batch[1]
                scores = module.forward(feats)
                module.backward(feats, scores, labels)
            if epoch == 70:
                pass
        ClassifierHandler.compute_accuracy(train_loader[1][0], train_loader[1][1], module)

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyperparameters and return the loaded model
        """
        raise NotImplementedError("function is not implemented")

    @staticmethod
    def compute_accuracy(wavs: torch.Tensor, labels: torch.Tensor, module: MusicClassifier ):
        """
        """
        wavs = wavs.squeeze(1)
        output_labels = module.classify(wavs)
        first_class_acc = torch.sum((output_labels == float(Genre.CLASSICAL.value)) & (labels == float(Genre.CLASSICAL.value))) / torch.sum(labels == float(Genre.CLASSICAL.value))
        second_class_acc = torch.sum((output_labels == float(Genre.HEAVY_ROCK.value)) & (labels == float(Genre.HEAVY_ROCK.value))) / torch.sum(labels == float(Genre.HEAVY_ROCK.value))
        third_class_acc = torch.sum((output_labels == float(Genre.REGGAE.value)) & (labels == float(Genre.REGGAE.value))) / torch.sum(labels == float(Genre.REGGAE.value))
        print("first class accuracy: ", first_class_acc)
        print("second class accuracy: ", second_class_acc)
        print("third class accuracy: ", third_class_acc)


ClassifierHandler.train_new_model(TrainingParameters())
