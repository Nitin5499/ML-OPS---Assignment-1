# ML Ops Assignment 1

## Setup Instructions

1. Create and activate the conda environment:
   ```sh
   conda create -n mlops2 python=3.11 -y
   conda activate mlops2
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. To train and evaluate the DecisionTreeRegressor:
   ```sh
   python train.py
   ```
4. To train and evaluate the KernelRidge model:
   ```sh
   python train2.py
   ```

## Project Structure
- `requirements.txt`: Python dependencies
- `misc.py`: Utility functions for data loading, preprocessing, training, and evaluation
- `train.py`: DecisionTreeRegressor workflow
- `train2.py`: KernelRidge workflow

## GitHub Actions
- On the `kernelridge` branch, GitHub Actions will automatically run both models and display their MSEs on every push.
