import torch
from config import device


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = torch.nn.RNN(
            input_size, hidden_size, batch_first=True, num_layers=num_layers
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.to(device)  # Move model to the appropriate device

    # x shape : [batch_size, sequence_length, input_size]
    # hidden shape : [num_layers, batch_size, hidden_size]
    def forward(self, x, hidden):
        x = x.to(device)  # Move input to device
        hidden = hidden.to(device)  # Move hidden state to device
        output, hidden = self.rnn(x, hidden)
        # output shape : [batch_size, sequence_length, hidden_size]
        # hidden shape : [num_layers, batch_size, hidden_size]
        output = self.fc(output[:, -1, :])
        # output shape : [batch_size, output_size (vocab_size)]
        return output, hidden

    def init_hidden(self, batch_size, num_layers):
        return torch.zeros(num_layers, batch_size, self.hidden_size, device=device)


def load_model(model_path, input_size, hidden_size, output_size, num_layers):
    model = RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    return model
