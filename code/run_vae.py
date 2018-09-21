#!/usr/bin/env python3

import mxnet as mx
import numpy as np
import math
import urllib.request
import os, logging, sys
from typing import Optional, List
from argparse import ArgumentParser
from os.path import join, exists
from numpy import genfromtxt
from matplotlib import cm, pyplot as plt
from vae import construct_vae, ElboMetric

# from matplotlib import pyplot as plt

DEFAULT_LEARNING_RATE = 0.0003

data_names = ['train', 'valid', 'test']
train_set = ['train']
test_set = ['test']
data_dir = join(os.curdir, "binary_mnist")


# def plot_digit(digit: np.array) -> None:
#     '''
#     Plots an mnist digit encoded in a pixel array.
#
#     :param digit: An array of pixels.
#     '''
#     size = np.sqrt(digit.shape[0])
#     digit.reshape((size, size))
#
#     plt.imshow(digit, cmap='gray')
#     plt.show()


def load_data(train: bool = True, logger: Optional[logging.Logger] = logging) -> dict:
    '''
    Download binarised mnist data set and load it into memory.

    :param: Whether to load training or test data.
    :param: A logger for the data loading process.
    :return: Binarised mnist data.
    '''
    if not exists(data_dir):
        os.mkdir(data_dir)
    for data_set in data_names:
        file_name = "binary_mnist.{}".format(data_set)
        goal = join(data_dir, file_name)
        if exists(goal):
            logger.info("Data file {} exists".format(file_name))
        else:
            logger.info("Downloading {}".format(file_name))
            link = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
                data_set)
            urllib.request.urlretrieve(link, goal)
            logger.info("Finished")

    data = {}
    data_sets = train_set if train else test_set
    for data_set in data_sets:
        file_name = join(data_dir, "binary_mnist.{}".format(data_set))
        logger.info("Reading {} into memory".format(file_name))
        data[data_set] = mx.nd.array(genfromtxt(file_name))
        logger.info("{} contains {} data points".format(file_name, data[data_set].shape[0]))

    return data


def train_model(generator_layers: List[int],
                inference_layers: List[int],
                latent_size: int,
                batch_size: int,
                epochs: int = 10,
                learning_rate: float = DEFAULT_LEARNING_RATE,
                optimiser: str = "adam",
                ctx: mx.context = mx.cpu(),
                logger: Optional[logging.Logger] = logging):
    """
    Train a variational autoencoder model.

    :param generator_layers:
    :param inference_layers:
    :param latent_size:
    :param batch_size:
    :param ctx:
    :param logger:
    :return:
    """
    mnist = load_data(train=True, logger=logger)
    train_iter = mx.io.NDArrayIter(data=mnist['train'], data_name="data", label=mnist["train"], label_name="label",
                                   batch_size=batch_size, shuffle=True)

    vae = construct_vae(latent_type="gaussian", likelihood="bernoulliProd", generator_layer_sizes=generator_layers,
                        infer_layer_size=inference_layers, latent_variable_size=latent_size,
                        data_dims=mnist['train'].shape[1], generator_act_type='tanh', infer_act_type='tanh')

    module = mx.module.Module(vae.train(mx.sym.Variable("data"), mx.sym.Variable('label')),
                              data_names=[train_iter.provide_data[0][0]],
                              label_names=[train_iter.provide_label[0][0]], context=ctx,
                              logger=logger)

    logger.info("Starting to train")
    #
    module.fit(train_data=train_iter, optimizer=optimiser, force_init=True, force_rebind=True, num_epoch=epochs,
               optimizer_params={'learning_rate': learning_rate},
               validation_metric=ElboMetric(),
               # eval_data=val_iter,
               batch_end_callback=mx.callback.Speedometer(frequent=20, batch_size=batch_size),
               epoch_end_callback=mx.callback.do_checkpoint('vae'),
               eval_metric="Loss")



def reconstruct_from_model(
                generator_layers: List[int],
                inference_layers: List[int],
                latent_size: int,
                samples: int, 
                ctx: mx.context = mx.cpu(),
                logger: Optional[logging.Logger] = logging
                ):
  mnist = load_data(train=False) # to get the dimension
  test_set = mnist['test']
  random_idx = np.random.randint(test_set.shape[0])
  random_picture = test_set[random_idx, :]
  width = height = int(math.sqrt(test_set.shape[1]))
  plot, canvas = plt.subplots(1, figsize=(5,5))
  canvas.imshow(np.reshape(random_picture.asnumpy(), (width, height)), cmap = cm.Greys)
  plt.show()

  logger.info("Loading saved module: vae")
  vae = construct_vae(latent_type="gaussian", likelihood="bernoulliProd", generator_layer_sizes=generator_layers,
                      infer_layer_size=inference_layers, latent_variable_size=latent_size,
                      data_dims=test_set.shape[1], generator_act_type='tanh', infer_act_type='tanh')
  sym, arg_params, aux_params =  mx.model.load_checkpoint("vae", 20)
  arg_params["random_digit"] = random_picture
  reconstructions = mx.sym.Group([vae.generate_reconstructions(mx.sym.Variable("random_digit"), samples)])
  reconstructions_exec = reconstructions.bind(ctx=ctx, args=arg_params)

  logger.info("Generating {} samples".format(samples))
  digits = reconstructions_exec.forward()

  digits = digits[0].asnumpy()

  #plot
  width = height = int(math.sqrt(mnist['test'].shape[1]))
  cols = 3
  rows = int(samples/cols)
  plot, axes = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(30,6))
  sample = 0
  for row in range(rows):
    for col in range(cols):
      axes[row][col].imshow(np.reshape(digits[sample,:], (width, height)), cmap=cm.Greys)
      sample += 1
  plt.show()
  

def sample_from_model(
                generator_layers: List[int],
                inference_layers: List[int],
                latent_size: int,
                samples: int, 
                ctx: mx.context = mx.cpu(),
                logger: Optional[logging.Logger] = logging
                ):
  logger.info("Loading saved module: vae")
  mnist = load_data(train=False) # to get the dimension
  vae = construct_vae(latent_type="gaussian", likelihood="bernoulliProd", generator_layer_sizes=generator_layers,
                      infer_layer_size=inference_layers, latent_variable_size=latent_size,
                      data_dims=mnist['test'].shape[1], generator_act_type='tanh', infer_act_type='tanh')
  sym, arg_params, aux_params =  mx.model.load_checkpoint("vae", 20)
  logger.info("Generating {} samples".format(samples))
  

  # We group the outputs of phantasize to be able to process them as a single symbol
  dream_digits = mx.sym.Group([vae.phantasize(samples)])
  dream_exec = dream_digits.bind(ctx=ctx, args=arg_params)

  # run the computation
  digits = dream_exec.forward()

  # transform into numpy arrays
  digits = digits[0].asnumpy()

  #plot
  width = height = int(math.sqrt(mnist['test'].shape[1]))
  cols = 3
  rows = int(samples/cols)
  plot, axes = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(30,6))
  sample = 0
  for row in range(rows):
    for col in range(cols):
      axes[row][col].imshow(np.reshape(digits[sample,:], (width, height)), cmap=cm.Greys)
      sample += 1
  plt.show()



def main():
    command_line_parser = ArgumentParser("Train a VAE on binary mnist and generate images of random digits.")

    command_line_parser.add_argument('-b', '--batch-size', default=500, type=int,
                                     help="Training batch size. Default: %(default)s.")
    command_line_parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="adam",
                                     help="The optimizer to use during training. "
                                          "Choices: %(choices)s. Default: %(default)s")
    command_line_parser.add_argument('-e', '--epochs', default=20, type=int,
                                     help="Number of epochs to run during training. Default: %(default)s.")
    command_line_parser.add_argument('--latent-dim', type=int, default=300,
                                     help='Dimensionality of the latent variable. Default: %(default)s.')
    command_line_parser.add_argument('--num-gpus', type=int, default=0,
                                     help="Number of GPUs to use. CPU is used if set to 0. Default: %(default)s.")
    command_line_parser.add_argument('-s', '--sample-random-digits', type=int, default=0,
                                     help="Load parameters of a previously trained VAE and randomly generate "
                                          "image digits from it.")
    command_line_parser.add_argument('-r', '--reconstruct-random-digits', type=int, default=0,
                                     help="Load parameters of a previously trained VAE and reconstruct "
                                          "a random digit from it.")

    args = command_line_parser.parse_args()

    ctx = mx.cpu() if args.num_gpus <= 0 else [mx.gpu(i) for i in range(args.num_gpus)]
    opt = args.opt
    epochs = args.epochs
    batch_size = args.batch_size

    generator_layers = [400, 600]
    inference_layers = [600, 400]
    latent_size = args.latent_dim

    training = not args.sample_random_digits and not args.reconstruct_random_digits

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    if training:
        train_model(generator_layers=generator_layers, inference_layers=inference_layers, latent_size=latent_size,
                    batch_size=batch_size, epochs=epochs, optimiser=opt, ctx=ctx)
    elif args.sample_random_digits:
      sample_from_model(generator_layers=generator_layers, inference_layers=inference_layers, latent_size=latent_size,
                    samples=args.sample_random_digits, ctx=ctx)
    else:
      reconstruct_from_model(generator_layers=generator_layers, inference_layers=inference_layers, latent_size=latent_size,
                    samples=args.reconstruct_random_digits, ctx=ctx)



if __name__ == "__main__":
    main()
