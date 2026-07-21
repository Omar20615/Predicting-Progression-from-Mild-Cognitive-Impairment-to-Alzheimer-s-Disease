import pandas as pd

# Read the original CSV file
df = pd.read_csv('output_complete.csv')

selected_columns = df.columns[:12].tolist()
selected_columns += ['ST40CV', 'ST30SV', 'ST88SV', 'ST60CV', 'ST99CV', 'ST83TA', 'ST83CV']
df_selected = df[selected_columns]

df_selected.to_csv('new_output_data.csv', index=False)
