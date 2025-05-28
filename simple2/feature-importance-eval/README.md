# Feature Importance Evaluation Project

This project aims to evaluate feature importance for attributes in a PyTorch Geometric Data object, specifically focusing on the attributes `x`, `pos`, `edge_attr`, and `temperature`. The evaluation will help in understanding the contribution of each feature towards the model's predictions.

## Project Structure

```
feature-importance-eval
├── src
│   ├── main.py               # Entry point of the application
│   ├── feature_importance.py # Functions to compute and evaluate feature importance
│   ├── data_utils.py         # Utility functions for data handling
│   └── types
│       └── index.py          # Custom types and data structures
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd feature-importance-eval
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`. After activating your environment, run:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the feature importance evaluation, execute the following command:
```
python src/main.py
```

## Functionality

- **Data Handling**: The project includes utility functions for loading and preprocessing data in `data_utils.py`.
- **Feature Importance Evaluation**: The core functionality for computing and evaluating feature importance is implemented in `feature_importance.py`.
- **Custom Types**: The project defines custom types and data structures in `types/index.py` to facilitate data handling and feature evaluation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.