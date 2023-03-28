# Stock Price Forecasting Flask Web App

### Predicting Appple's Adj. Close Price For The Next 7 Days 
### WEB APP - https://aritheanalyst.com/ir

### Recommendation
* 0 seems to be the best parameters for p and q with 1 as the order of differencing to use when forecasting AAPL stock dataset but I recommend using an autoarima model to be sure the best parameters are picked before fitting in the training data.

## Setup
- Install the requirements and setup the development environment.

	`pip3 install -r requirements.txt`
	`make install && make dev`

- Run the application.

		`python3 main.py`

- Navigate to `localhost:5000`.

## Future Work
   * Use a simple LSTM model to forecast 7 days out then do the same with a Multivariate LSTM model. 
