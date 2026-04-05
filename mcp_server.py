import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from advanced_eda import AdvancedEDA

mcp = FastMCP("AdvancedEDA_Server")

@mcp.tool()
def analyze_dataset(file_path: str, output_directory: str = ".", corr_method: str = "pearson") -> str:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on a structured dataset.
    
    Args:
        file_path: The path to the source data file (.csv or .parquet).
        output_directory: The folder to save the detailed JSON and Excel reports.
        corr_method: Method for correlation analysis.
        
    Returns:
        A structured string detailing specific data quality violations (missing values, 
        outliers, constant columns) so the LLM can determine the necessary cleaning steps.
    """
    path = Path(file_path)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not path.exists():
        return f"Error: The file {file_path} does not exist."

    try:
        # Load data
        if path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            eda = AdvancedEDA(df, name=path.stem)
        elif path.suffix.lower() == ".parquet":
            eda = AdvancedEDA.from_parquet(path)
        else:
            return "Error: Unsupported file format."

        # Run analysis and save artifacts for deep-dives
        report = eda.run_full_report(corr_method=corr_method)
        json_path = out_dir / f"{path.stem}_eda_report.json"
        eda.save_json(json_path, corr_method=corr_method)
        
        # ---------------------------------------------------------
        # ACTIONABLE INTELLIGENCE EXTRACTION FOR THE LLM
        # ---------------------------------------------------------
        
        overview = report.get("dataset_overview", {})
        
        # 1. Extract Missing Values
        missing_data = report.get("missing_value_report", {}).get("columns", [])
        missing_str = "\n".join([
            f"  - '{col['feature']}': {col['missing_pct']}% missing ({col['severity']} severity)"
            for col in missing_data if col['missing_pct'] > 0
        ]) or "  - None"

        # 2. Extract Severe Outliers (Only returning Moderate/High/Critical to save tokens)
        outliers_data = report.get("outlier_summary", {}).get("columns", [])
        outliers_str = "\n".join([
            f"  - '{col['feature']}': {col['iqr_outlier_pct']}% IQR outliers ({col['severity']} severity)"
            for col in outliers_data if col['severity'] in ["Moderate", "High", "Critical"]
        ]) or "  - None"

        # 3. Extract Constant/Useless Columns
        meta_data = report.get("column_metadata", [])
        constant_cols = [c['feature'] for c in meta_data if c.get('is_constant', False)]
        fully_unique_cols = [c['feature'] for c in meta_data if c.get('is_fully_unique', False) and c.get('dtype_kind') in ['O', 'U']] # Likely useless IDs
        
        useless_str = ""
        if constant_cols: useless_str += f"  - Constant Columns (1 unique value): {', '.join(constant_cols)}\n"
        if fully_unique_cols: useless_str += f"  - Fully Unique Object/String Columns (likely IDs): {', '.join(fully_unique_cols)}\n"
        if not useless_str: useless_str = "  - None\n"

        # 4. Extract High Correlations (Multicollinearity risk)
        corr_data = report.get("correlation_analysis", {}).get("high_correlation_pairs", [])
        corr_str = "\n".join([
            f"  - {pair['feature_a']} & {pair['feature_b']}: {pair['correlation']} correlation"
            for pair in corr_data
        ]) or "  - None"

        # Assemble the final LLM prompt payload
        summary = f"""
✅ EDA Complete for '{path.name}'
Shape: {overview.get('rows')} rows, {overview.get('columns')} columns
Duplicate Rows: {overview.get('duplicates', {}).get('duplicate_rows')}

ACTIONABLE DATA QUALITY ISSUES:

[1] Missing Values:
{missing_str}

[2] Outliers (IQR Method):
{outliers_str}

[3] Low Variance / High Cardinality (Candidates for Dropping):
{useless_str.strip()}

[4] Multicollinearity (High Correlation Pairs):
{corr_str}

Instruction to Agent: Review the issues above. Select and execute the appropriate data cleaning tools (e.g., imputation, dropping columns, handling outliers) before proceeding with analysis. The full detailed JSON report is available locally at: {json_path.absolute()}
"""
        return summary.strip()

    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"

if __name__ == "__main__":
    mcp.run()