# Character-Level RNN Text Generation

This project implements a character-level Recurrent Neural Network (RNN) for text generation. The model learns to predict the next character in a sequence based on the preceding characters, allowing it to generate new text that mimics the style of the input data.

## Project Structure

The project is organized as follows:

*   **`input.txt`**: The primary input text file used for training the model. You can replace this with your own text data.
*   **`input1.txt`**: An alternative input text file.
*   **`config.py`**: Contains hyperparameters for the model and training process (e.g., sequence length, batch size, learning rate, number of layers, hidden size).
*   **`data.py`**: Handles data loading, preprocessing, vocabulary creation, and character encoding/decoding.
*   **`model.py`**: Defines the RNN model architecture (class `RNN`) and includes a function to load a pre-trained model.
*   **`train.py`**: The main script for training the RNN model. It handles data preparation, model initialization, the training loop (including validation), saving the trained model (`model.pth`), and logging loss history (`loss_history.csv`).
*   **`generator.py`**: Generates new text using the trained model (`model.pth`). It takes a seed string and predicts subsequent characters.
*   **`analyze_data.py`**: Script for performing analysis on the input data. It can generate visualizations like character frequency, sequence length distribution, and identify sequence patterns. The outputs are saved as `.png` files and a `data_analysis_summary.txt`.
*   **`plot_loss.py`**: Utility script to plot the training and validation loss curves from `loss_history.csv`. The output is saved as `loss_curve.png`.
*   **`requirements.txt`**: Lists the Python dependencies required to run the project.
*   **`model.pth`**: The saved weights of the trained model.
*   **`loss_history.csv`**: CSV file logging the training and validation loss for each epoch.
*   **`*.png`**: Image files for various plots (character frequency, loss curve, etc.).
*   **`data_analysis_summary.txt`**: Text file summarizing the results of the data analysis.
*   **`.gitignore`**: Specifies files and directories that Git should ignore.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

1.  **Prepare your input data:**
    *   Place your training text data in `input.txt`. The model will learn to generate text based on the style and content of this file. You can also use `input1.txt` or modify `data.py` to load data from other sources.

2.  **Configure training parameters (optional):**
    *   Adjust hyperparameters such as `sequence_length`, `batch_size`, `epochs`, `hidden_size`, `num_layers`, and `learning_rate` in `config.py` as needed.

3.  **Run the training script:**
    ```bash
    python train.py
    ```
    *   The script will preprocess the data, train the RNN model, and save the trained model to `model.pth`.
    *   Training progress, including loss values for each epoch, will be printed to the console and saved in a log file (e.g., `training_YYYYMMDD_HHMMSS.log`).
    *   A `loss_history.csv` file will also be generated, containing training and validation losses per epoch.
    *   The script will automatically use the device specified in `config.py` (MPS if available, otherwise CPU).

## Generating Text

Once the model is trained and `model.pth` is available, you can generate new text:

1.  **Run the generation script:**
    ```bash
    python generator.py
    ```
    *   The script will load the trained model from `model.pth`.
    *   It will then use a default seed string ("A") to start the generation process and print the generated text to the console.
    *   You can modify `generator.py` to change the seed string, the length of the generated text, or the number of samples to generate. The generation stops if the model produces a ">" character or reaches the maximum length.

## Additional Scripts and Outputs

*   **`analyze_data.py`**:
    *   Run this script to perform basic analysis on your input data (`input.txt`).
        ```bash
        python analyze_data.py
        ```
    *   It generates:
        *   `character_frequency.png`: A plot of character frequencies.
        *   `sequence_length_distribution.png`: A histogram of sequence lengths (if applicable to your data format).
        *   `sequence_patterns.png`: Visualizations of common character patterns.
        *   `data_analysis_summary.txt`: A text file with a summary of the analysis.

*   **`plot_loss.py`**:
    *   Use this script to visualize the training and validation loss curves after training.
        ```bash
        python plot_loss.py
        ```
    *   It reads `loss_history.csv` and generates `loss_curve.png`.

*   **Log files**:
    *   Training logs are saved in files named like `training_YYYYMMDD_HHMMSS.log`. These logs contain detailed information about the training process, including loss at each batch and epoch.

## Future Improvements & Contributions

This project provides a basic framework for character-level text generation. Here are some potential areas for improvement and contribution:

*   **Experiment with different RNN architectures:** Try LSTMs or GRUs for potentially better performance.
*   **Implement attention mechanisms:** This could help the model focus on relevant parts of the input sequence.
*   **Add support for word-level or subword-level tokenization:** This might be more effective for certain types of text.
*   **Hyperparameter tuning:** Systematically explore different hyperparameter combinations to optimize performance.
*   **More sophisticated data analysis:** Expand `analyze_data.py` with more in-depth text analysis techniques.
*   **Interactive generation:** Create a simple web interface or CLI tool for easier text generation.
*   **Packaging the project:** Make it easier to install and use as a library.

Contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.
