import torch
import torch.nn as nn


cell_1 = nn.RNNCell(input_size=10, hidden_size=20, nonlinearity='tanh')
cell_2 = nn.LSTMCell(input_size=10, hidden_size=20)
cell_3 = nn.GRUCell(input_size=10, hidden_size=20)

cell_1 = nn.LSTMCell(10, 20)
cell_2 = nn.LSTMCell(20, 20)

full_cell = nn.Sequential(cell_1, cell_2)

input = torch.randn(2, 3, 10)   # (time_steps, batch, input_size)
hx_init = torch.randn(3, 20)    # hidden state of size: (batch_size, hidden_size)
# output of output gate
cx_init = torch.randn(3, 20)    # hidden state of size: (batch_size, hidden_size)
# output of write gate
output = []

hx, cx = hx_init, cx_init
for t in range(input.size()[0]):
    hx, cx = cell_1(input[t], (hx, cx))
    hx2, cx2 = cell_2(hx, (hx, cx))
    output.append(hx2)
output = torch.stack(output, dim=0)     # shape is (time_steps, batch_size, input_size)
print(f'output: {output}')


multi_layer_rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, nonlinearity='tanh')
multi_layer_lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)

output2 = []

hx, cx = hx_init, cx_init
for t in range(input.size()[0]):
    hx, cx = cell_1(input[t], (hx, cx))
    hx2, cx2 = cell_2(hx, (hx, cx))
    output2.append(hx2)
output2 = torch.stack(output2, dim=0)     # shape is (time_steps, batch_size, input_size)
print(f'output2: {output2}')

print(f'torch.all(output == output2): {torch.all(output == output2)}')


multi_layer_rnn_dr = nn.RNN(input_size=10, hidden_size=20, num_layers=2,
                            nonlinearity='tanh', batch_first=False, dropout=0.5)
multi_layer_lstm_dr = nn.LSTM(input_size=10, hidden_size=20, num_layers=2,
                              batch_first=False, dropout=0.5)

input_size = 32

cell_1 = nn.LSTM(input_size, hidden_size=10, num_layers=2, dropout=1)
rnn = nn.LSTM(input_size=32, hidden_size=20, num_layers=1, batch_first=False)

inputs = torch.randn((32, 32, 32))
output, states = rnn(inputs)
print(f'output: {output}')


input = torch.randn(5, 3, 10)       # (time_steps, batch, input_size)
h_0 = torch.randn(2, 3, 20)         # (n_layers, batch_size, hidden_size)
c_0 = torch.randn(2, 3, 20)         # (n_layers, batch_size, hidden_size)

rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
output_n, (hn, cn) = rnn(input, (h_0, c_0))
print(f'output_n: {output_n}')
