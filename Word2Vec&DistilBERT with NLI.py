import random
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers



# ########################## PART 1: PROVIDED CODE ##############################
def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize_w2v(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.c
    

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


def collate_cbow(batch):
    """
    Collate function for the CBOW model. This is needed only for CBOW but not skip-gram, since
    skip-gram indices can be directly formatted by DataLoader. For more information, look at the
    usage at the end of this file.
    """
    sources = []
    targets = []

    for s, t in batch:
        sources.append(s)
        targets.append(t)

    sources = torch.tensor(sources, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)

    return sources, targets


def train_w2v(model, optimizer, loader, device):
    """
    Code to train the model. See usage at the end.
    """
    model.train()

    for x, y in tqdm(loader, miniters=20, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)

        loss = F.cross_entropy(y_pred, y)
        loss.backward()

        optimizer.step()

    return loss


class Word2VecDataset(torch.utils.data.Dataset):
    """
    Dataset is needed in order to use the DataLoader. See usage at the end.
    """

    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets
        assert len(self.sources) == len(self.targets)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]


# ######################## PART 2: PROVIDED CODE ########################


def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.
    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.
    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        dd = data_dict

        if len(dd["premise"]) != len(dd["hypothesis"]) or len(dd["premise"]) != len(
            dd["label"]
        ):
            raise AttributeError("Incorrect length in data_dict")

    def __len__(self):
        return len(self.data_dict["premise"])

    def __getitem__(self, idx):
        dd = self.data_dict
        return dd["premise"][idx], dd["hypothesis"][idx], dd["label"][idx]


def train_distilbert(model, loader, device):
    model.train()
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in tqdm(loader):
        optimizer.zero_grad()

        inputs = model.tokenize(premise, hypothesis).to(device)
        target = target.to(device, dtype=torch.float32)
        
        pred = model(inputs)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_distilbert(model, loader, device):
    model.eval()

    targets = []
    preds = []

    for premise, hypothesis, target in loader:
        preds.append(model(model.tokenize(premise, hypothesis).to(device)))

        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


# ######################## PART 1: YOUR WORK STARTS HERE ########################


def build_current_surrounding_pairs(indices: "list[int]", window_size: int = 2):
    """
    Pairs up each current word w(t) with its surrounding words
    (...w(t+2), w(t+1), w(t-1), w(t-2)...), with respect to a window size.

    Parameters
    ----------
    indices: list of int
        The indices output by tokens_to_ix for a single sample (not for the entire dataset).
    window_size: int
        The number of indices from each side to choose from.
        The context should be 2 times the window size (e.g. 2*5 = 10)

    Returns
    -------
    surrounding_indices: list[list[int]]
        Indices nearby the current word but does not include itself. Make sure
        that the length of the inner surrounding is consistent (i.e. 2 * window_size)
    current_indices: list[int]
        Indices of the current words in the middle of a context. denoted as
        w(t) in the paper. The length of this list should be
        "sample_indices - 2 * window_size"

    Notes
    -----
    To ensure that current indices has a consistent inner length, you will need
    to omit the start and end of the sample_indices list from the surrounding,
    since it would otherwise have unbalanced sides.

    Examples
    --------
    >>> text = "dogs and cats are playing".split()
    >>> surroundings, currents = build_current_surrounding_pairs(text, window_size=1)
    >>> print(currents)
    ['and', 'cats', 'are']
    >>> print(surroundings)
    [['dogs', 'cats'], ['and', 'are'], ['cats', 'playing']]

    >>> indices = [word_to_index[t] for t in text]
    >>> surroundings, currents = build_current_surrounding_pairs(indices, window_size=1)
    >>> print(currents)
    [3, 4887, 11]
    >>> print(surroundings)
    [[110, 4887], [3, 11], [4887, 31]]
    """
    # TODO: your work here
    currents=indices[window_size:-window_size]
    surroundings=[]
    if len(currents)==0:
        return surroundings,currents
    else:
     for i in range(window_size,len(indices)-window_size):
         former=[indices[f_number] for f_number in range(i-window_size,i)]
         after=[indices[a_number] for a_number in range(i+1,i+window_size+1)]
         temp=former+after
         surroundings.append(temp)
     return surroundings,currents

def expand_surrounding_words(
    ix_surroundings: "list[list[int]]", ix_current: "list[int]"
):
    """
    Using the output of build_current_surrounding_pairs(), convert the
    surrounding into pairs of context-target pair. The resulting lists should be longer.

    Parameters
    ----------
    ix_surroundings: list of list of int
        Aka the context from a window. Those are the indices of words around the current word.
    ix_current: list of int
        The indices of the current words. Denoted as w(t) in the paper.

    Returns
    -------
    ix_surroundings_expanded: list of int
        The indices of a each surrounding word (after expansion).
    ix_current_expanded: list of int
        The indices of the current word (after expansion) matching
        each single surrounding word.

    Example
    -------
    >>> # dogs and cats are playing
    >>> surroundings = [['dogs', 'cats'], ['and', 'are'], ['cats', 'playing']]
    >>> currents = ['and', 'cats', 'are']
    >>> surroundings_expanded, current_expanded = expand_surrounding_words(surroundings, currents)
    >>> print(surroundings_expanded)
    ['dogs', 'cats', 'and',  'are',  'cats', 'playing']
    >>> print(current_expanded)
    ['and',  'and',  'cats', 'cats', 'are',  'are']

    >>> ix_surroundings = [[110, 4887], [3, 11], [4887, 31]]
    >>> ix_currents = [3, 4887, 11]
    >>> ix_surr_expanded, ix_curr_expanded = expand_surrounding_words(ix_surroundings, ix_currents)
    >>> print(ix_surr_expanded)
    [110, 4887, 3, 11, 4887, 31]
    >>> print(ix_curr_expanded)
    [3, 3, 4887, 4887, 11, 11]
    """
    # TODO: your work here
    out=[]
    for lst in ix_surroundings:
      ind=ix_surroundings.index(lst)
      temp=[ix_current[ind]]*len(lst)
      out.append(temp) 
    ix_curr_expanded = [j for i in out for j in i] 
    ix_surr_expanded = [j for i in ix_surroundings for j in i]
    return ix_surr_expanded, ix_curr_expanded


def cbow_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    """
    Use the build_current_surrounding_pairs function you used above to complete this
    function. The difference is that the input is a list of indices (so a nested
    list), but the output format should be the same.

    Parameters
    ----------
    indices_list: list of list of int
        A nested list of indices. The inner list contains the indices of a single sample.
        The outer list contains all the samples in the dataset.
    window_size: int
        The number of indices from each side to choose from.

    Returns
    -------
    sources: list of list of int
        The inputs to the CBOW model. The inner list contains the indices of surrounding words
        and the outer list contains the samples (from a batch or from whole dataset).
    targets: list of int
        The targets of the CBOW model. Contains the indices of the target word w(t).
    """
    # TODO: your work here
    sources=[]
    targets=[]
    for lst in indices_list:
        surroundings,currents=build_current_surrounding_pairs(lst,window_size)
        sources.extend(surroundings)
        targets.extend(currents)
    
    return sources,targets


def skipgram_preprocessing(
    indices_list: "list[list[int]]", window_size: int = 2
):
    """
    Use the build_current_surrounding_pairs function you used above to complete
    this function. The difference is that the input is a list of indices (so a
    nested list), but the output format should be the same.

    Parameters
    ----------
    indices_list: list of list of int
        A nested list of indices. The inner list contains the indices of a single
        sample. The outer list contains all the samples in the dataset.
    window_size: int
        The number of indices from each side to choose from.

    Returns
    -------
    sources: list of int
        The inputs to the Skip-gram model. List of indices of the target word w(t).
    targets: list of int
        The targets of the CBOW model. List of indices of a single surrounding word.

    Notes
    -----
    Here, you need to return all possible pairs between a word w(t) and its
    surroundings. In the paper, it's a sampling method based on distance, but we
    will not do that for simplicity, instead we'll just use everything.
    """
    # TODO: your work here
    sources=[]
    targets=[]
    for lst in indices_list:
        surroundings,currents=build_current_surrounding_pairs(lst,window_size)
        surr_expand,curr_expand=expand_surrounding_words(surroundings,currents)
        sources.extend(surr_expand)
        targets.extend(curr_expand)
    return sources,targets


class SharedNNLM:
    def __init__(self, num_words: int, embed_dim: int):
        """
        SkipGram and CBOW actually use the same underlying architecture,
        which is a simplification of the NNLM model (no hidden layer)
        and the input and output layers share the same weights. You will
        need to implement this here.

        Notes
        -----
          - This is not a nn.Module, it's an intermediate class used
            solely in the SkipGram and CBOW modules later.
          - Projection does not have a bias in word2vec
        """

        # TODO: your work here
        
        self.embedding = nn.Embedding(num_words,embed_dim)
        self.projection = nn.Linear(embed_dim, num_words,bias=False)
        

        self.bind_weights()

    def bind_weights(self):
        """
        Bind the weights of the embedding layer with the projection layer.
        This mean they are the same object (and are updated together when
        you do the backward pass).
        """
        emb = self.get_emb()
        proj = self.get_proj()

        proj.weight = emb.weight

    def get_emb(self):
        return self.embedding

    def get_proj(self):
        return self.projection


class SkipGram(nn.Module):
    """
    Use SharedNNLM to implement skip-gram. Only the forward() method differs from CBOW.
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: tensor of shape [batch_size]
            The indices of the target word w(t)

        Returns
        -------
        tensor of shape [batch_size]
            The predicted distribution of the index of a surrounding word.
        """
        # TODO: your work here
        output = self.emb(x)
        output = self.proj(output)
    
        return output


class CBOW(nn.Module):
    """
    Use SharedNNLM to implement CBOW. Only the forward() method differs from SkipGram.
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: tensor of shape [batch_size, 2 * window_size]
            The indices of the surrounding words, i.e. 
            w(t - window_size), ..., w(t + window_size).

        Returns
        -------
        tensor of shape [batch_size]
            The predicted distribution of the index of w(t).
        """
        # TODO: your work here
        x = self.emb(x)
        x = x.sum(dim=1)
        x = self.proj(x)
        return x



def compute_topk_similar(
    word_emb: torch.Tensor, w2v_emb_weight: torch.Tensor, k
) -> list:
    """
    Compute the cosine similarity between the embedding of a single word and
    the embedding of all words, then return the indices of the top k most
    similar results (excluding the word itself).

    Parameters
    ----------
    word_emb: tensor of shape [1, embed_dim]
        The embedding representation of a single word
    w2v_emb_weight: tensor of shape to
        The entire tensor representing the weight of the
        nn.Embedding of all words in the vocabulary.

    Returns
    -------
    list of int
        The indices of the top k most similar words.
    """
    # TODO: your work here
    cos = nn.CosineSimilarity(dim=1)
    similarity=cos(word_emb,w2v_emb_weight)
    copy=similarity.tolist()
    out=sorted(range(len(copy)), key=lambda i: copy[i], reverse=True)[:k+1]
    out.pop(0)
    return out


@torch.no_grad()
def retrieve_similar_words(
    model: nn.Module,
    word: str,
    index_map: "dict[str, int]",
    index_to_word: "dict[int, str]",
    k: int = 5,
) -> "list[str]":
    """
    Using compute_topk_similar() and a w2v model to find the k most similar words 
    from your vocab. You will need to disable gradient calculations and set your 
    model to evaluation mode.

    Parameters
    ----------
    model: nn.Module
        Either the SkipGram or CBOW model you created previously.
    word: str
        Some word that exists in index_map (or the list of its keys)
    index_map: dict of {str: int}
        A dictionary mapping a word to its index
    index_to_word: dict of {int: str}
        A dictionary mapping an index to the word (reverse of index_map)

    Returns
    -------
    list of str
        The k most similar words to the input word.

    Notes
    -----
    compute_topk_similar() takes as input a word_emb of shape [1, embed_dim].
    You will need to handle that.
    """
    # TODO: your work here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        word_ind=index_map[word]
        word_tensor=torch.tensor([word_ind])
        word_emb=model.emb(word_tensor.to(device))
        vocab_emb=model.emb.weight
        top_similar_indices=compute_topk_similar(word_emb,vocab_emb,k)
        top_similar_words=[index_to_word[w] for w in top_similar_indices]
    return top_similar_words



@torch.no_grad()
def word_analogy(
    model: nn.Module,
    word_a: str,
    word_b: str,
    word_c: str,
    index_map: "dict[str, int]",
    index_to_word: "dict[int, str]",
    k: int = 5,
) -> "list[str]":
    """
    Using compute_topk_similar() and a w2v model,
    you want to compute the following analogy:
        > "word_a" is to "word_b" what "?" is to "word_c"

    It can also be represented as:
        > word_a - word_b + word_c = ?

    You will have to find the k most similar words to "?" from your vocab. You will need to
    disable gradient calculations and set your model to evaluation mode.

    Parameters
    ----------
    model: nn.Module
        Either the SkipGram or CBOW model you created previously.
    word_a: str
        Some word that exists in index_map (or the list of its keys)
    index_map: dict of {str: int}
        A dictionary mapping a word to its index
    index_to_word: dict of {int: str}
        A dictionary mapping an index to the word (reverse of index_map)

    Returns
    -------
    list of str
        The k most similar words to the unknown word "?".

    Notes
    -----
    compute_topk_similar() takes as input a word_emb of shape [1, embed_dim].
    You will need to handle that.
    """
    # TODO: your work here
    print(type(model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    print('Successfully set the model to `.eval()` mode')
    with torch.no_grad():
        
        a_ind=index_map[word_a]
        a_tensor=torch.tensor([a_ind])
        a_emb=model.emb(a_tensor.to(device))
        
        b_ind=index_map[word_b]
        b_tensor=torch.tensor([b_ind])
        b_emb=model.emb(b_tensor.to(device))
        
        c_ind=index_map[word_c]
        c_tensor=torch.tensor([c_ind])
        c_emb=model.emb(c_tensor.to(device))
        
        out_emb=a_emb-b_emb+c_emb
        vocab_emb=model.nnlm.get_emb().weight
        
        top_similar_indices=compute_topk_similar(out_emb,vocab_emb,k)
        top_similar_words=[index_to_word[w] for w in top_similar_indices]
    
    return top_similar_words



# ######################## PART 2: YOUR WORK STARTS HERE ########################
class CustomDistilBert(nn.Module):
    def __init__(self):
        """
        - Load the DistilBERT model's pretrained "base uncased" weights from
          the Huggingface repository. We want the bare encoder outputting 
          hidden-states without any specific head on top.
        - Load the corresponding pre-trained tokenizer using the same method.
        - self.pred_layer takes the output of the model and predicts a single 
          score (binary, 1 or 0), then pass the output to the sigmoid layer
        - self.sigmoid should return torch's sigmoid activation.
        - self.criterion should be the binary cross-entropy loss. You 
          may use torch.nn here.

        Link:
            https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/distilbert#transformers.DistilBertModel
        """
        super().__init__()

        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.pred_layer = nn.Linear(768,1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    # DO NOT CHANGE BELOW THIS LINE
    def get_distilbert(self):
        return self.distilbert

    def get_tokenizer(self):
        return self.tokenizer

    def get_pred_layer(self):
        return self.pred_layer

    def get_sigmoid(self):
        return self.sigmoid
    
    def get_criterion(self):
        return self.criterion
    # DO NOT CHANGE ABOVE THIS LINE

    def assign_optimizer(self, **kwargs):
        """
        This assigns the Adam optimizer to this model's parameters (self) and returns the 
        optimizer.

        Parameters
        ----------
        **kwargs
            The arguments passed to the optimizer.

        Returns
        -------
        torch optimizer
            An Adam optimizer bound to the model
        """
        # TODO: your work here
        optimizer = torch.optim.Adam(self.parameters(),lr= 0.0001)
        
        return optimizer 

    def slice_cls_hidden_state(
        self, x: transformers.modeling_outputs.BaseModelOutput
    ) -> torch.Tensor:
        """
        Using the output of the model, return the last hidden state of the CLS token.

        Parameters
        ----------
        x: transformers BaseModelOutput
            The output of the distilbert model. You need to retrieve the hidden state 
            of the last output layer, then slice it to obtain the hidden representation.
            The last hidden state has shape: [batch_size, sequence_length, hidden_size]

        Returns
        -------
        torch.Tensor of shape [batch_size, hidden_size]
            The last layer's hidden state representing the [CLS] token. Usually, CLS
            is the first token in the sequence.
        """
        # TODO: your work here
        last_hidden_states = x.last_hidden_state
        cls_token = last_hidden_states[:,0,:]  
        return cls_token

    def tokenize(
        self,
        premise: "list[str]",
        hypothesis: "list[str]",
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = True,
    ):
        """
        This function will be applied to the premise and hypothesis (list of str) 
        to obtain the inputs for  your model. You will need to use the Huggingface 
        tokenizer returned by get_tokenizer().

        Parameters
        ----------
        premise: list of str
            The first text to be input in your model.
        hypothesis: list of str
            The second text to be input in your model.

            https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__

        Returns
        -------
        transformers.BatchEncoding
            A dictionary-like object that can be directly given to the model with **
        """
        
        # TODO: your work here
        tokens=self.tokenizer(premise,hypothesis,padding=padding,truncation=truncation,max_length=max_length,return_tensors='pt')
        return tokens

    def forward(self, inputs: transformers.BatchEncoding):
        """
        Parameters
        ----------
        inputs: transformers.BatchEncoding
            The input ingested by our model. Output of tokenizer for a given batch

        Returns
        -------
        tensor of shape [batch_size]
            The output prediction for each element in the batch, with sigmoid
            activation. Make sure the shape is not [batch_size, 1]

        Notes
        -----
        In the original BERT paper, the output representation of CLS is used for
        classification. You will need to slice the output of your DistilBERT to
        obtain the representation before giving it to the last layer with sigmoid 
        activation.
        """
        
        # TODO: your work here
        distilbert_output=self.distilbert(**inputs.to(device))
        cls_token=self.slice_cls_hidden_state(distilbert_output)
        
        pred=self.pred_layer(cls_token)
        sig_act=self.sigmoid(pred)
        return sig_act.reshape(-1)
        
        
      


if __name__ == "__main__":
    random.seed(2022)
    torch.manual_seed(2022)

    # Parameters (you can change them)
    sample_size = 2500  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 2
    num_words = 50000
    # data_path = "../input/a1-data"
    data_path = "../input/a1-data"

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 1: TEST CODE ######################
    print("=" * 80)
    print("Running test code for part 1")
    print("-" * 80)

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets(data_path)
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = (
        train_raw["premise"]
        + train_raw["hypothesis"]
        + valid_raw["premise"]
        + valid_raw["hypothesis"]
    )
    
   
    # Process into indices
    tokens = tokenize_w2v(full_text)

    word_counts = build_word_counts(tokens)
    word_to_index = build_index_map(word_counts, max_words=num_words)
    index_to_word = {v: k for k, v in word_to_index.items()}

    text_indices = tokens_to_ix(tokens, word_to_index)
    
    # Train CBOW
    sources_cb, targets_cb = cbow_preprocessing(text_indices, window_size=2)
    print(len(sources_cb))
    print(len(sources_cb[0]))
    loader_cb = DataLoader(
        Word2VecDataset(sources_cb, targets_cb),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cbow,
    )

    model_cb = CBOW(num_words=len(word_to_index), embed_dim=200).to(device)
    optimizer = torch.optim.Adam(model_cb.parameters())

    for epoch in range(n_epochs):
        loss = train_w2v(model_cb, optimizer, loader_cb, device=device).item()
        print(f"Loss at epoch #{epoch}: {loss:.4f}")

    # Training skip-gram
    
    # TODO: your work here
    model_sg =  SkipGram(200,embed_dim=200).to(device)
    optimizer = torch.optim.Adam(model_sg.parameters())
    # RETRIEVE SIMILAR WORDS
    word = "man"

    similar_words_cb = retrieve_similar_words(
        model=model_cb,
        word=word,
        index_map=word_to_index,
        index_to_word=index_to_word,
        k=5,
    )

    similar_words_sg = retrieve_similar_words(
        model=model_sg,
        word=word,
        index_map=word_to_index,
        index_to_word=index_to_word,
        k=5,
    )

    print(f"(CBOW) Words similar to '{word}' are: {similar_words_cb}")
    print(f"(Skip-gram) Words similar to '{word}' are: {similar_words_sg}")

    # COMPUTE WORDS ANALOGIES
    a = "man"
    b = "woman"
    c = "girl"

    analogies_cb = word_analogy(
        model=model_cb,
        word_a=a,
        word_b=b,
        word_c=c,
        index_map=word_to_index,
        index_to_word=index_to_word,
    )
    analogies_sg = word_analogy(
        model=model_sg,
        word_a=a,
        word_b=b,
        word_c=c,
        index_map=word_to_index,
        index_to_word=index_to_word,
    )

    print(f"CBOW's analogies for {a} - {b} + {c} are: {analogies_cb}")
    print(f"Skip-gram's analogies for {a} - {b} + {c} are: {analogies_sg}")

    # ###################### PART 2: TEST CODE ######################
    print("=" * 80)
    print("Running test code for part 2...")
    from sklearn.metrics import f1_score

    print("-" * 80)

    train_loader = torch.utils.data.DataLoader(
        NLIDataset(train_raw), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        NLIDataset(valid_raw), batch_size=batch_size, shuffle=False
    )

    model = CustomDistilBert().to(device)
    optimizer = model.assign_optimizer(lr=1e-4)

    for epoch in range(n_epochs):
        loss = train_distilbert(model, train_loader, device=device)

        preds, targets = eval_distilbert(model, valid_loader, device=device)
        preds = preds.round()

        score = f1_score(targets.cpu(), preds.cpu())
        print("Epoch:", epoch)
        print("Training loss:", loss)
        print("Validation F1 score:", score)
        print()
        