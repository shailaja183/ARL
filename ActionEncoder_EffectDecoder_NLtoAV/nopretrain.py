import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dataset

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('nopretrain_max_updates', 50, 'max updates')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Vocabulary(object):
    def __init__(self):
        self.vocab = []
        self.vocab_id_map = {}

        for session_id in dataset.get_session_ids():
            for (_, language, _) in dataset.get_session_data(session_id):
                tokens = language.split(' ')
                for token in tokens:
                    if token not in self.vocab_id_map:
                        new_id = len(self.vocab)
                        self.vocab.append(token)
                        self.vocab_id_map[token] = new_id

    def token_ids(self, language):
        return [self.vocab_id_map[t] for t in language.split(' ')]

    def get_vocab_size(self):
        return len(self.vocab)

class Model(object):
    def __init__(self):
        self.vocab = Vocabulary()
        self.language_module = dataset.LSTMLanguageModule(FLAGS.flattened_message_size,self.vocab.get_vocab_size()).to(device)
        self.decoder = dataset.Decoder(FLAGS.flattened_message_size).to(device)
        all_params = list(self.language_module.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(all_params, weight_decay=1e-5)
        self.training_examples = []

    def predict(self, state, command):
        self.language_module.eval()
        self.decoder.eval()
        token_ids = self.vocab.token_ids(command)
        command_variable = torch.LongTensor(token_ids).unsqueeze(0).to(device)
        state_variable = dataset.state_to_variable(state).to(device)
        encoder_output = self.language_module.forward(command_variable)
        decoder_input = encoder_output if FLAGS.continuous_message else discrete_util.discrete_transformation(encoder_output)
        prediction = self.decoder.forward(state_variable, decoder_input)
        return dataset.output_from_variable(prediction, state)

    def optimizer_step(self):
        self.language_module.train()
        self.decoder.train()
        random.shuffle(self.training_examples)
        for state, command, target in self.training_examples:
            self.optimizer.zero_grad()
            state_variable = dataset.state_to_variable(state).to(device)
            target_variable = dataset.output_to_variable(target, state).to(device)
            token_ids = self.vocab.token_ids(command)
            command_variable = torch.LongTensor(token_ids).unsqueeze(0).to(device)

            encoder_output = self.language_module.forward(command_variable)
            decoder_input = encoder_output if FLAGS.continuous_message else discrete_util.discrete_transformation(encoder_output)
            prediction = self.decoder.forward(state_variable, decoder_input, target_variable)

            loss = dataset.loss(prediction, target_variable)

            loss.backward()
            self.optimizer.step()

    def training_accuracy(self):
        n_correct = 0
        for state, command, target in self.training_examples:
            prediction = self.predict(state, command)
            if prediction == target:
                n_correct += 1
        return n_correct / len(self.training_examples)

    def update(self, state, command, target_output, num_updates=None):
        if num_updates is None:
            num_updates = FLAGS.nopretrain_max_updates
        self.training_examples.append((state, command, target_output))
        for _ in range(num_updates):
            self.optimizer_step()

