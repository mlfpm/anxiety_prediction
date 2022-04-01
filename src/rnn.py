import numpy as np
import torch
from torch import nn


class AnxietyRNN(nn.Module):
    """
    The RNN model that will be used to perform anxiety analysis.
    """

    def __init__(
            self,
            input_dim_rnn,
            hidden_dim_rnn,
            n_layers_rnn,
            input_dim_fc,
            hidden_dim_fc,
            output_dim,
    ):
        """
        Initialise the model by setting up the layers.

        :param int input_dim_rnn: number of input features
        :param int hidden_dim_rnn: number of RNN units
        :param int n_layers_rnn: number of RNN layers
        :param int input_dim_fc: input dimension of the fully connected network
        :param int hidden_dim_fc: hidden dimension of the fully connected network; if zero, a single fully connected
            layer is used, otherwise two
        :param int output_dim: output dimension
        """

        super(AnxietyRNN, self).__init__()

        self.n_layers_rnn = n_layers_rnn
        self.hidden_dim = hidden_dim_rnn
        self.hidden_dim_fc = hidden_dim_fc
        self.output_dim = output_dim

        # embedding and LSTM layers
        if n_layers_rnn > 1:
            self.rnn = nn.RNN(
                input_dim_rnn,
                hidden_dim_rnn,
                n_layers_rnn,
                dropout=0.2,
                batch_first=True,
            )
        else:
            self.rnn = nn.RNN(
                input_dim_rnn, hidden_dim_rnn, n_layers_rnn, batch_first=True
            )

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and relu layers
        if hidden_dim_fc:
            self.fc1 = nn.Linear(1 + input_dim_fc, hidden_dim_fc)
            self.relu1 = nn.ReLU()

            self.fc2 = nn.Linear(hidden_dim_fc, output_dim)
        else:
            self.fc = nn.Linear(1 + input_dim_fc, output_dim)

        # sigmoid activation for the outputs
        self.sig = nn.Sigmoid()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x_t, x_s):
        """
        Perform a forward pass of our model on some input and hidden state.

        :param Tensor x_t: temporal input to the network
        :param Tensor x_s: static input to the network
        :return: network output
        """

        if len(x_t.size()) == 2:
            batch_size, seq_len = x_t.size()
            n_features = 1
        else:
            batch_size, seq_len, n_features = x_t.size()

        # initialise hidden states of LSTM
        hidden = self.init_hidden(batch_size)

        # embeddings and rnn_out
        rnn_out, hidden = self.rnn(x_t.view(batch_size, seq_len, n_features), hidden)

        # stack up lstm outputs
        rnn_out = rnn_out.contiguous().view(batch_size, -1)  # [:, -1].view(-1, 1)

        # dropout
        rnn_out = self.dropout(rnn_out)[:, -1].view(-1, 1)

        # concatenate the EMA features
        x_concat = torch.cat((rnn_out, x_s), 1)

        # fully-connected layer; reshape to be batch_size first
        if self.hidden_dim_fc:
            fc_out = self.relu1(self.fc1(x_concat))
            out = self.sig(self.fc2(fc_out)).view(batch_size, -1)
        else:
            out = self.sig(self.fc(x_concat)).view(batch_size, -1)

        return out

    def init_hidden(self, batch_size):
        """
        Initializes hidden state of the RNN to zero with sizes n_layers_rnn x batch_size x hidden_dim.

        :param int batch_size: batch size
        :return: initialised RNN weights
        """

        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (
                weight.new(self.n_layers_rnn, batch_size, self.hidden_dim).zero_().cuda()
            )
        else:
            hidden = weight.new(self.n_layers_rnn, batch_size, self.hidden_dim).zero_()

        return hidden


class AnxietyRNNExtended(AnxietyRNN):
    """
    Extended version of the AnxietyRNN.
    """

    def __init__(
            self,
            input_dim_rnn,
            hidden_dim_rnn,
            n_layers_rnn,
            input_dim_fc,
            hidden_dim_fc,
            output_dim,
    ):
        """
        Class initializer.

        :param int input_dim_rnn: number of input features
        :param int hidden_dim_rnn: number of RNN units
        :param int n_layers_rnn: number of RNN layers
        :param int input_dim_fc: input dimension of the fully connected network
        :param int hidden_dim_fc: hidden dimension of the fully connected network; if zero, a single fully connected
            layer is used, otherwise two
        :param int output_dim: output dimension
        """
        super(AnxietyRNNExtended, self).__init__(
            input_dim_rnn,
            hidden_dim_rnn,
            n_layers_rnn,
            input_dim_fc,
            hidden_dim_fc,
            output_dim,
        )
        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )

        self.criterion = nn.BCELoss()

        self.loss_during_training = []

        self.valid_loss_during_training = []

        self.to(self.device)

    def train_loop(
            self,
            train_loader,
            valid_loader=None,
            verbose=True,
            max_nb_epochs=100,
            print_every=10,
            clip=10,
            model_path=None,
            save_all=False,
    ):
        """
        Method to train the model.

        :param DataLoader train_loader: the training data set
        :param DataLoader valid_loader: the validation set
        :param bool verbose: if True, information about the training evolution is printed
        :param int max_nb_epochs: maximum number of epochs to perform
        :param int print_every: how frequently to print the training details
        :param float clip: gradient clipping threshold
        :param str model_path: full path where to save the models during training
        :param bool save_all: flag to indicate whether to save the models from
            each step or only when the valid_loss decreases
        """
        if self.valid_loss_during_training:
            valid_loss_min = np.amin(self.valid_loss_during_training)
        else:
            valid_loss_min = np.Inf  # track change in validation loss

        # SGD Loop
        for epoch in range(1, max_nb_epochs + 1):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            ###################
            # train the model #
            ###################
            self.train()

            # Batch loop
            for temp_inputs, ema_inputs, gad_labels in train_loader:
                # move tensors to GPU if CUDA is available
                temp_inputs, ema_inputs, gad_labels = (
                    temp_inputs.to(self.device),
                    ema_inputs.to(self.device),
                    gad_labels.to(self.device),
                )

                # clear the gradients of all optimized variables
                self.optimiser.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.forward(temp_inputs, ema_inputs)

                # calculate the batch loss
                loss = self.criterion(output.squeeze(), gad_labels)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), clip)

                # perform a single optimization step (parameter update)
                self.optimiser.step()

                # update training loss
                train_loss += loss.item() * temp_inputs.size(0)

                # calculate the accuracy
                train_acc += self.accuracy(
                    output, gad_labels
                ).item() * temp_inputs.size(0)

            # calculate average training loss and store
            train_loss = train_loss / len(train_loader.sampler)
            self.loss_during_training.append(train_loss)

            # calculate average training accuracies
            train_acc = train_acc / len(train_loader.sampler)

            ######################
            # validate the model #
            ######################
            if valid_loader is not None:
                self.eval()
                for temp_inputs, ema_inputs, gad_labels in valid_loader:
                    # move tensors to GPU if CUDA is available
                    temp_inputs, ema_inputs, gad_labels = (
                        temp_inputs.to(self.device),
                        ema_inputs.to(self.device),
                        gad_labels.to(self.device),
                    )

                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = self.forward(
                        temp_inputs, ema_inputs,
                    )

                    # calculate the batch loss
                    loss = self.criterion(output.squeeze(), gad_labels)

                    # update average validation loss
                    valid_loss += loss.item() * temp_inputs.size(0)

                    # calculate the accuracy
                    valid_acc += self.accuracy(
                        output, gad_labels
                    ).item() * temp_inputs.size(0)

                # calculate average validation loss and save
                valid_loss = valid_loss / len(valid_loader.sampler)
                self.valid_loss_during_training.append(valid_loss)

                # calculate average validation accuracy
                valid_acc = valid_acc / len(valid_loader.sampler)

                # print training/validation info
                if verbose and epoch % print_every == 0:
                    print(
                        "Epoch: {}/{}\t".format(epoch, max_nb_epochs),
                        "Trn. Loss: {:.4f}\t".format(train_loss),
                        "Trn. Acc: {:.2f}% \t".format(train_acc),
                        "Val. Loss: {:.4f}\t".format(valid_loss),
                        "Val. Acc: {:.2f}%".format(valid_acc),
                    )
            else:
                # print training info
                if verbose and epoch % print_every == 0:
                    print(
                        "Epoch: {}/{}\t".format(epoch, max_nb_epochs),
                        "Trn. Loss: {:.4f}\t".format(train_loss),
                        "Trn. Acc: {:.2f}% \t".format(train_acc),
                    )

            # save model if validation loss has decreased
            if verbose and (valid_loss <= valid_loss_min) and (model_path is not None):
                print(
                    "Epoch: {}\t".format(epoch),
                    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    ),
                )
                torch.save(self, model_path + "earl_stp.pt")
                valid_loss_min = valid_loss

            if save_all and (model_path is not None):
                torch.save(self, model_path + "epoch_" + str(epoch + 1) + ".pt")

    @staticmethod
    def accuracy(y_probs, y):
        """
        Compute the accuracy given the probabilities and expected labels.

        :param Tensor y_probs: predicted probabilities
        :param Tensor y: true labels
        :return: computed accuracy percentage
        """
        # Get predictions from the maximum value
        y_hat = torch.round(y_probs.data)

        # Total number of labels
        total = y.size(0)

        # Total correct predictions
        correct = (y_hat == y).sum()

        acc = 100.0 * correct / total

        return acc

    def predict(self, x_temp, x_ema):
        """
            Compute prediction given a single observation.
        """
        self.eval()

        # move tensors to GPU if CUDA is available
        x_temp, x_ema = (x_temp.to(self.device), x_ema.to(self.device))

        # forward pass: compute predicted outputs by passing inputs to the model
        y_prob = self.forward(x_temp, x_ema)

        self.train()

        # Get predictions from the maximum value
        y_hat = torch.round(y_prob.data)

        return y_hat.numpy().reshape(-1, 1)

    def predict_for_eval(self, test_loader):
        """
        Compute predictions for each observation in the testloader and
        return all true labels, predicted labels and predicted probabilities
        concatenated in a list.

        :param DataLoader test_loader: the test set
        :return: return all the results concatenated
        """
        y_true_list = []
        y_pred_list = []
        y_score_list = []

        self.eval()
        for i, (temp_inputs, ema_inputs, gad_labels) in enumerate(test_loader):
            y_true_list.append(gad_labels.numpy().reshape(-1))

            # move tensors to GPU if CUDA is available
            temp_inputs, ema_inputs, gad_labels = (
                temp_inputs.to(self.device),
                ema_inputs.to(self.device),
                gad_labels.to(self.device),
            )

            # forward pass: compute predicted outputs by passing inputs to the model
            y_probs = self.forward(temp_inputs, ema_inputs)

            # Get predictions from the maximum value
            y_hat = torch.round(y_probs.data)

            y_pred_list.append(y_hat.cpu().detach().numpy().reshape(-1))
            y_score_list.append(y_probs.cpu().detach().numpy())

        self.train()
        return (
            np.hstack(y_true_list),
            np.hstack(y_pred_list),
            np.array(
                [
                    y_score_list[i][j]
                    for i in range(len(y_score_list))
                    for j in range(len(y_score_list[i]))
                ]
            ),
        )
