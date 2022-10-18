import numpy as np
import random
import sys
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dataset

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pretrain_prefix', '', 'pretrain model file prefix')
flags.DEFINE_integer('pretrain_iterations', 500000, 'training iterations')
flags.DEFINE_float('pretrain_kl_weight', 0.0, 'kl weight')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def batch_iterator(iterator, n):
    b = []
    for i in iterator:
        b.append(i)
        if len(b) == n:
            yield b
            b = []
    if len(b) > 0:
        yield b

def train():
    encoder = dataset.Encoder(FLAGS.flattened_message_size).to(device)
    decoder = dataset.Decoder(FLAGS.flattened_message_size).to(device)
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, weight_decay=1e-5)
    losses = []

    encoder.train()
    decoder.train()

    for iter_idx, batch in enumerate(batch_iterator(dataset.get_states_and_actions(),20)):
        iteration = iter_idx + 1

        states = [s for s,t in batch]
        targets = [t for s,t in batch]
            
        optimizer.zero_grad()
        state_variable = dataset.state_to_variable_batch(states).to(device)
        target_variable = dataset.output_to_variable_batch(targets, states).to(device)
        encoder_output = encoder.forward(state_variable, target_variable)
        decoder_input = encoder_output 

        prediction = decoder.forward(state_variable, decoder_input, target_variable)
        prediction_loss = dataset.loss(prediction, target_variable)

        loss = prediction_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 1000 == 0:
            print('===== itr %s =====' % (iteration))
            print('loss %s ' % (np.mean(losses)))
            losses = []
            torch.save(encoder.state_dict(), FLAGS.pretrain_prefix + 'encoder_parameters.pt')
            torch.save(decoder.state_dict(), FLAGS.pretrain_prefix + 'decoder_parameters.pt')
            print('sg2 reconstruction accuracy:', reconstruction_accuracy(encoder, decoder))

        if FLAGS.pretrain_iterations is not None and iteration >= FLAGS.pretrain_iterations:
            break

def reconstruction_accuracy(encoder, decoder, n=20):
    encoder.eval()
    decoder.eval()

    batch = next(batch_iterator(dataset.get_states_and_actions(),n))

    states = [s for s,t in batch]
    targets = [t for s,t in batch]

    state_variable = dataset.state_to_variable_batch(states).to(device)
    target_variable = dataset.output_to_variable_batch(targets, states).to(device)
    encoder_output = encoder.forward(state_variable, target_variable)
    decoder_input = encoder_output 
    
    prediction = decoder.forward(state_variable, decoder_input)

    outputs = dataset.output_from_variable_batch(prediction, states)

    correct = 0
    for i in range(n):
        if outputs[i] == targets[i]:
            correct += 1

    encoder.train()
    decoder.train()

    return correct / n

def saved_model_exists():
    return os.path.exists(FLAGS.pretrain_prefix + 'encoder_parameters.pt') and \
        os.path.exists(FLAGS.pretrain_prefix + 'decoder_parameters.pt')

def load_saved_encoder():
    encoder = dataset.Encoder(FLAGS.flattened_message_size)
    encoder.load_state_dict(torch.load(FLAGS.pretrain_prefix + 'encoder_parameters.pt'))
    return encoder

def load_saved_decoder():
    decoder = dataset.Decoder(FLAGS.flattened_message_size)
    decoder.load_state_dict(torch.load(FLAGS.pretrain_prefix + 'decoder_parameters.pt'))
    return decoder

if __name__ == '__main__':
    FLAGS(sys.argv)
    dataset.load()
    train()
