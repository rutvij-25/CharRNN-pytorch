# CharRNN-pytorch

## Pytorch implementation of Character level recurrent neural network

### train.py 

Use your own text data and name it `data.txt` \n
Then run 
```
>python train.py
```

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

### generate.py 

After training 

| Arguement | Parse | Default |
| ----------- | ----------- | ----------- |
| Initial Text | --initialtext | Where |
| Temperature | --temperature | 1 |
| Number of Letter to Generate | --n | 100 |
| Root directory of data | --root | data.txt |








