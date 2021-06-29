# CharRNN-pytorch

## Pytorch implementation of Character level recurrent neural network
#### To generate text

### train.py 

Use your own text data and name it `data.txt`
Then run 
```
python train.py
```

Your model will be saved in `pretrained` folder

| Arguement | Parse | Default |
| ----------- | ----------- | ----------- |
| Embedding Dimension | --embed_dim | 65 |
| Dropout | --dropout | 0.1 |
| Hidden Size | --hidden_size | 50 |
| Number of LSTM layers | --num_layers | 2 |
| Batch Size | --batch_size | 100 |
| Chunk Size | --chunk_size | 200 |
| Epochs | --epochs | 200 |
| Learning Rate | --lr | 0.01 |
| Root directory of data | --root | data.txt |
| Save the model every N epochs | --save_every | 50 |

### generate.py 

| Arguement | Parse | Default |
| ----------- | ----------- | ----------- |
| Initial Text | --initialtext | Where |
| Temperature | --temperature | 1 |
| Number of Letter to Generate | --n | 100 |
| Root directory of data | --root | data.txt |


### Output

```
python generate.py --n 200

Where of call, let all not no Risent us thy do.

BETHE:
Ay, that we onoth be miled to did; and
That here and manesty
Nabtalmy most evancion they stash?
You bick'd gelolot part? then bloinopt.        

LADY MACBETH

```








