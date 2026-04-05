import pandas as pd
import numpy as np
import uuid

# Set seed for reproducibility
np.random.seed(42)
n_rows = 5000

print("Generating deeply flawed dataset...")

# 1. Base Data
data = {
    # High cardinality string (Fully Unique - useless for ML)
    "employee_id": [str(uuid.uuid4()) for _ in range(n_rows)], 
    "department": np.random.choice(["IT", "HR", "Sales", "Engineering"], n_rows),
    "years_experience": np.random.uniform(1, 25, n_rows),
    # Constant column (Low Variance - 0 predictive power)
    "company_status": ["Active"] * n_rows, 
}
df = pd.DataFrame(data)

# 2. Multicollinearity Trap (Highly Correlated Pairs)
# Salary is tightly mathematically coupled to years of experience
df["base_salary"] = 40000 + (df["years_experience"] * 3500) + np.random.normal(0, 1500, n_rows)
# Bonus is directly tied to salary, creating a multicollinearity nightmare
df["annual_bonus"] = df["base_salary"] * 0.15 + np.random.normal(0, 300, n_rows) 

# 3. Severe Outlier Trap
# Most have 5-20 days off, but we inject extreme data entry errors (e.g., 500 days)
df["pto_days_taken"] = np.random.normal(14, 3, n_rows)
outlier_indices = np.random.choice(n_rows, 200, replace=False)
df.loc[outlier_indices, "pto_days_taken"] = np.random.uniform(300, 600, 200)

# 4. Missing Values Trap
# Inject 22% missing data into 'base_salary'
missing_salary_idx = np.random.choice(n_rows, int(n_rows * 0.22), replace=False)
df.loc[missing_salary_idx, "base_salary"] = np.nan

# Inject 8% missing data into 'department'
missing_dept_idx = np.random.choice(n_rows, int(n_rows * 0.08), replace=False)
df.loc[missing_dept_idx, "department"] = np.nan

# Save it to the workspace
file_name = "messy_corporate_data.parquet"
df.to_parquet(file_name, index=False)

print(f"Successfully created '{file_name}'.")
print("Traps laid: Missing Values, IQR Outliers, Constant Columns, and Multicollinearity.")