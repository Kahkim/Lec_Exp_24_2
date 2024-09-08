import pandas as pd

demand_data = pd.read_csv('demand_data.csv')
demand_data = demand_data.head(5)

print(demand_data['DEMAND_QTY'].mean())
