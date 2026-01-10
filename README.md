How to Run the code:
1. The first step is to clone the repository by writing in the terminal: git clone https://github.com/franciscomldsantos/DSAP_Project

2. Unzip the data.zip and make sure the "data" folder (which has the CSV data files) is inside the project folder

3. Then the dependencies need to be installed: conda env create -f environment.yml

4. (optional) The data loader can be run, but due to inconsistencies with yfinance API, the
    dataset is also provided inside the folder data. To request the data again, the following
    input can be used: python src/data_loader.py (Downloads sp500 data)
    
5. The final step is to execute the pipeline with the orchestrator: python src/main.py
