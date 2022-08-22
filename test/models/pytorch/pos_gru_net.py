import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx as onnx

from torch.utils.data import DataLoader
from torchtext.datasets import UDPOS

t.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return t.tensor(idxs, dtype=t.long)


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64
HIDDEN_DIM = 64


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn_cell = nn.GRU(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        reshaped_embeds = embeds.view(sentence.shape[0], 1, -1)
        lstm_out, _ = self.rnn_cell(reshaped_embeds)
        tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def initHidden(self):
        return t.zeros(1, 1, self.hidden_dim)


if __name__ == '__main__':

    train_set = UDPOS(root='./data', split='train')

    training_data = []
    for sent, pos, _ in train_set:
        assert len(sent) == len(pos)
        training_data.append((sent, pos))

    word_to_ix = {}
    tag_to_ix = {}

    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    #print(word_to_ix)
    #print(tag_to_ix)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(0):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with t.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)

        print(tag_scores)

    sentence_in = prepare_sequence(training_data[0][0], word_to_ix)
    onnx.export(model, sentence_in, 'pytorch/saved_models/my_gru.onnx', export_params=True, opset_version=13, training=onnx.TrainingMode.EVAL)