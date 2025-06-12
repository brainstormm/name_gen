import torch
from model import load_model
from config import input_size, hidden_size, sequence_length, device
from data import get_vocab_size, encode, decode

model = load_model(
    "model.pth",
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=get_vocab_size(),
)

for j in range (500):
    seed_string = " "
    generated_text = ""

    init_hidden = model.init_hidden(batch_size=1)
    for i in range(100):
        input = torch.tensor(encode(seed_string))
        input = input.reshape(1, len(seed_string), input_size) / float(get_vocab_size())
        input = input.to(device)
        output, hidden = model(input, init_hidden)
        prob = torch.softmax(output, dim=1)
        predicted_index = torch.multinomial(prob, num_samples=1).item()
        predicted_char = decode([predicted_index])

        if(len(seed_string) < sequence_length):
            seed_string += predicted_char
        else:
            seed_string = seed_string[1:] + predicted_char

        if predicted_char == '>':
            break
        generated_text += predicted_char
    print(generated_text)
