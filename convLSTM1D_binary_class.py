from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ConvLSTM1D
from tensorflow.keras.utils import to_categorical
import numpy as np




def convlstm_binary_class(input_shape, output_shape,
                     **kwargs):




    # list current settings to allow for default model settings
    # here, you should define default values for all the model settings.
    config = {}
    config['model_type'] = 'convlstms1D_binary_class'
    config['activation'] = kwargs.get('activation', 'LeakyReLU')
    config['activation_second'] = kwargs.get('activation_second', 'ReLU')
    config['activation_third'] = kwargs.get('activation_second', 'softmax')
    config['activation_last_layer'] = kwargs.get('activation_last_layer', 'sigmoid')
    config['filters'] = kwargs.get('filters', 64)
    config['kernel_size'] = kwargs.get('kernel_size', (1))
    config['loss'] = kwargs.get('loss', 'categorical_crossentropy')
    config['metrics'] = kwargs.get('metrics', ['accuracy'])
    config['num_hidden_layers'] = kwargs.get('num_hidden_layers', 1)
    config['optimizer'] = kwargs.get('optimizer', 'adam')


    print_summary = kwargs.get('print_summary', False)

    input_shape = input_shape[:1] + (1,) + input_shape[1:]


    # compile the model
    model = Sequential()
    model.add(ConvLSTM1D(filters=config['filters'], kernel_size=config['kernel_size'], activation=config['activation'],
                         input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation=config['activation_second']))
    model.add(Dense(output_shape, activation=config['activation_third']))
    model.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=config['metrics'])

    # print a model summary if print_summary == True
    if print_summary:
        print('model summary', model.summary())

    # return the model and its configuration
    return model, config

if __name__ == '__main__':


    # define variables for
    epochs = 20
    verbose = 1
    n_batch = 64
    n_timesteps = 200


    # generate some random sample data: 3 channels for 100 time steps
    x1 = np.expand_dims(np.sin(np.arange(start=0, stop=2 * np.pi, step=(2 * np.pi) / n_timesteps)), (0, 2))
    x2 = np.expand_dims(np.cos(np.arange(start=0, stop=2 * np.pi, step=(2 * np.pi) / n_timesteps)), (0, 2))
    x3 = x1 * x2
    X = np.concatenate([x1, x2, x3], axis=-1)
    X = np.repeat(X, n_batch, axis=0)
    y = np.random.randn(n_batch)
    for i in range(n_batch):
        X[i, :, :] = X[i, :, :] * y[i]
    y[y > 0] = 1
    y[y <= 0] = 0
    y = np.expand_dims(y, axis=1)
    y = to_categorical(y)
    X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])    #ConvLSTM1D requires 4D tensor (samples, time, rows, channels)

    print(f'shape of input: {X.shape}')
    print(f'shape of output: {y.shape}')



    n_in_channels = np.shape(X)[-1]
    n_out_channels = np.shape(y)[-1]

    # define input an output shapes:
    input_shape = (n_timesteps, n_in_channels)
    output_shape = (n_out_channels)

    # now build and train model
    model, config = convlstm_binary_class(input_shape, output_shape, print_summary=True)

    # fit network
    history = model.fit(x=X, y=y, epochs=epochs, batch_size=n_batch, verbose=verbose)
