How to Run the code:
1. The first step is to clone the repository by writing in the terminal: git clone https://github.com/franciscomldsantos/DSAP_Project

2. Then the dependencies need to be installed: conda env create -f environment.yml

3. (optional) The data loader can be run, but due to inconsistencies with yfinance API, the
    dataset is also provided inside the folder data. To request the data again, the following
    input can be used: python src/data_loader.py (Downloads sp500 data)
    
4. The final step is to execute the pipeline with the orchestrator: python src/main.py