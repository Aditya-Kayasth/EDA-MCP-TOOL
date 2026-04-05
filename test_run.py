from advanced_eda import AdvancedEDA
eda = AdvancedEDA.from_parquet("biometric.parquet")
print(eda.to_json())
