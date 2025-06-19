# PyTorch Linear Regression Example

This project demonstrates a simple linear regression model implemented using PyTorch.
It serves as a basic example to illustrate core PyTorch concepts such as:

- Defining a neural network model (`nn.Module`, `nn.Linear`)
- Generating synthetic data
- Defining a loss function (`nn.MSELoss`)
- Using an optimizer (`optim.SGD`)
- Implementing a training loop
- Basic visualization of results using `matplotlib`

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows: .\.venv\Scripts\activate
    # On macOS/Linux: source .venv/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Script

Once the setup is complete, you can run the script directly:
```bash
python linear_regression.py
```
This will train the model, print the loss at intervals, and display a plot of the original data versus the fitted regression line.