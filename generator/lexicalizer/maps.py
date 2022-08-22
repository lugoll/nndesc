
from enum import Enum
from generator.selector.selection import CNN, RNN, FFN

class Languages(Enum):
    EN = 'en'
    DE = 'de'

layer_type_lexicalizer = {
    'en': {
        'conv': 'Convolutional Layer',
        'maxpool': 'MaxPool Layer',
        'dense': 'Dense Layer',
        'flatten': 'Flatten Layer',
        'gru': 'Gated Recurrent Unit Cell',
        'lstm': 'Long Short Term Memory Cell',
    },
    'de': {
        'conv': 'Faltungschicht',
        'maxpool': 'MaxPool Poolingschicht',
        'dense': 'vollverbundene Schicht',
        'flatten': 'Flachungsschicht',
        'gru': 'Gated Recurrent Unit Cell',
        'lstm': 'Long Short Term Memory Cell',
    }
}

activation_type_lexicalizer = {
    'en': {
        'relu': 'ReLU',
        'softmax': 'Softmax',
        'logsoftmax': 'logarithmic Softmax',
        'sigmoid': 'Sigmoid',
    },
    'de': {
        'relu': 'ReLU',
        'softmax': 'Softmax',
        'logsoftmax': 'logarithmischen Softmax',
        'sigmoid': 'Sigmoid',
    }
}

pos_word_map = {
    'en': {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
        6: 'sixth',
        7: 'seventh',
        8: 'eighth',
        9: 'ninth',
        10: 'thenth',
        11: 'eleventh',
        12: 'twelveth'
    },
}

network_type_map = {
    'en': {
        CNN: 'convolutional Neural Network',
        RNN: 'recurrent Neural Network',
        FFN: 'feed-forward Neural Network'
    },
}


def get_pos_word_map_default(number: int, lang: Languages = Languages.EN):
    if number in pos_word_map[lang.value].keys():
        return pos_word_map[lang.value][number]
    elif number % 10 == 1:
        return f'{number}st'
    elif number % 10 == 2:
        return f'{number}nd'
    elif number % 10 == 1:
        return f'{number}rd'
    else:
        return f'{number}th'
