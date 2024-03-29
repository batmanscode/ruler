# Ruler 📏

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/batmanscode/ruler/main/ruler.py)

This web app uses machine learning to generate recommendations ([association rule learning](https://en.wikipedia.org/wiki/Association_rule_learning)) from an input CSV.

The input CSV needs the following input columns:

* Unique transaction ID (like an invoice number)
* Item name

To see what the output looks like, click 'Get rules' and rules will be generated using built in sample data.\
Sample data used is a variation of: [Online Retail Data Set](https://archive.ics.uci.edu/ml/datasets/online+retail)


### Demo
![demo](https://github.com/batmanscode/ruler/blob/main/demo.gif)

## How to Run Locally
Clone this repository into a new environment, install dependencies from `requirements.txt` and enter the following in your terminal:
```shell
streamlit run ruler.py
```
This will create a local web server which should open in your default browser. If not, just use one of the links returned in your terminal.

## Additional Info
* Web app built with [Streamlit](https://github.com/streamlit/streamlit).
* Associaton rules generated using Pycaret's [Association Rules module](https://pycaret.readthedocs.io/en/latest/api/arules.html).
