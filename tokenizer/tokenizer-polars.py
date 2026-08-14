'''

Tokenizing CSV full in memory using Polars.

This is the second step in processing, we are finally getting a Parquet file
wich is an column-oriente data format designed for efficient data storage
and retrieval wich gives compression and encoding schemes with enhanced 
performance.

'''

import polars as pl
import torch


df = pl.read_parquet("./preprocessing/csic_cleaned.parquet")


'''
    Create a tensor with the target column 
    using pytorch.
    Then defining all the labels for the text columns and asign it a maximum 
    length.
'''

y = torch.tensor(df["Label"].to_list(), dtype=torch.float32)


cols_text = ['Method', 'User-Agent', 'Pragma', 
              'Cache-Control', 'Accept', 'Accept-encoding', 'Accept-charset', 
              'language', 'host', 'cookie', 'content-type', 'connection', 'content', 'URL', 'URL_decoded']


cols_num = [
    "lenght", "classification"
]



max_lens = {
    'Method' :7 ,
    'User-Agent':  100, 
    'Pragma' : 50,
    'Cache-Control': 50 ,
    'Accept': 200,
    'Accept-encoding': 200,
    'Accept-charset': 100, 
    'language': 5, 
    'host': 120, 
    'cookie': 250, 
    'content-type': 225,
    'connection': 20 ,
    'content': 250 ,
    'URL': 250 , 
    'URL_decoded': 250 
}


'''
    The second step in tokenization process is to create an alphabet with all diferent
    characters in dataframe.

'''


unic_chars = set()

for col in cols_text:
    for text in df[col]:
        unic_chars.update(text)

sorted_chars = sorted(list(unic_chars))

alphabet = { char: i+2 for i, char in enumerate(sorted_chars)}

alphabet["<UNK>"] = 0
alphabet["<PAD>"] = 1


'''

    Third step is the actual tokenization of dataset, for this we are going to define a function.
    Which will map each character to an ID in `alphabet` that we create in the previous step.

    Input : Text of column, Max lenght of column
    Output: List of ID's
'''

def text_to_id ( text:str, max_length : int ) -> list[int]: 
    tokens = [ alphabet.get(char, alphabet["<UNK>"]) for char in text[:max_length]]

    if len(tokens) < max_length:
        padding = (max_length - len(tokens) )
        tokens += [alphabet['<PAD>']] * padding

    return tokens[:max_length]



'''

    This is the actual implementation of the function, the workflow is:

    Creates an empty array, then visits every single column of our labels array
    an applying our `text to id` function.
    
'''



tensor_blocks = []

for col in cols_text:
    texts = df[col].to_list()
    m_len = max_lens[col]

    seq_col = [text_to_id(text, m_len) for text in texts]

    tensor_column = torch.tensor(seq_col, dtype=torch.long)
    tensor_blocks.append(tensor_column)


for col in cols_num:
    tensor_num = torch.tensor(df[col].to_list(), dtype =torch.long).unsqueeze(1)
    tensor_blocks.append(tensor_num)

X_final = torch.cat(tensor_blocks, dim=1)

torch.save({
    'X': X_final,
    'y': y,
    'vocab': alphabet,
    'vocab_size': len(alphabet) 
}, "csic_rnn_inputs.pt")

print(
    f'''
        'X': {X_final},
        'y': {y},
        'vocab': {alphabet},
        'vocab_size': {len(alphabet)} 
    '''
)


'''
prev results:



        'X': tensor([[41, 39, 54,  ...,  1,  0,  0],
        [41, 39, 54,  ...,  1,  0,  0],
        [50, 49, 53,  ...,  1, 68,  0],
        ...,
        [41, 39, 54,  ...,  1,  0,  1],
        [41, 39, 54,  ...,  1,  0,  1],
        [41, 39, 54,  ...,  1,  0,  1]]),



        'y': tensor([0., 0., 0.,  ..., 1., 1., 1.]),



        'vocab': {' ': 2, '!': 3, '"': 4, '#': 5, '$': 6, '%': 7, '&': 8, "'": 9, '(': 10, ')': 11, '*': 12, '+': 13, ',': 14, '-': 15, '.': 16, '/': 17, '0': 18, '1': 19, '2': 20, '3': 21, '4': 22, '5': 23, '6': 24, '7': 25, '8': 26, '9': 27, ':': 28, ';': 29, '<': 30, '=': 31, '>': 32, '?': 33, '@': 34, 'A': 35, 'B': 36, 'C': 37, 'D': 38, 'E': 39, 'F': 40, 'G': 41, 'H': 42, 'I': 43, 'J': 44, 'K': 45, 'L': 46, 'M': 47, 'N': 48, 'O': 49, 'P': 50, 'Q': 51, 'R': 52, 'S': 53, 'T': 54, 'U': 55, 'V': 56, 'W': 57, 'X': 58, 'Y': 59, 'Z': 60, '_': 61, 'a': 62, 'b': 63, 'c': 64, 'd': 65, 'e': 66, 'f': 67, 'g': 68, 'h': 69, 'i': 70, 'j': 71, 'k': 72, 'l': 73, 'm': 74, 'n': 75, 'o': 76, 'p': 77, 'q': 78, 'r': 79, 's': 80, 't': 81, 'u': 82, 'v': 83, 'w': 84, 'x': 85, 'y': 86, 'z': 87, '|': 88, '~': 89, '�': 90, '<UNK>': 0, '<PAD>': 1},
        
        
        
        'vocab_size': 91 


''' 