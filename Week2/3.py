import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('./customer.csv')
except FileNotFoundError:
    print("Error: 'company_sales_data.csv' not found. Please make sure the file is in the same directory.")
    exit()
print("DataFrame head:")
print(df.head())
print("\nDataFrame info:")
df.info()
month_data = df['month_number']


total_profit_data = df['total_profit']

plt.figure(figsize=(10, 6))
plt.plot(month_data, total_profit_data, marker='o', linestyle='-')

plt.xlabel('Month Number')
plt.ylabel('Total profit')

plt.title('Total Profit Per Month')
plt.grid(True)
plt.savefig('total_profit_line_plot.png')
plt.show()

print("\nLine plot 'total_profit_line_plot.png' generated successfully!")