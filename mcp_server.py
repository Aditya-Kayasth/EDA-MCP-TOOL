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
        A complete, token-efficient metadata roster of all columns, plus critical warnings, 
        allowing the LLM to autonomously decide the best data cleaning strategy.
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

        # Run analysis and save artifacts
        report = eda.run_full_report(corr_method=corr_method)
        json_path = out_dir / f"{path.stem}_eda_report.json"
        eda.save_json(json_path, corr_method=corr_method)
        
        # ---------------------------------------------------------
        # THE "COMPLETE PICTURE" EXTRACTION FOR THE LLM
        # ---------------------------------------------------------
        
        overview = report.get("dataset_overview", {})
        meta_data = report.get("column_metadata", [])
        num_summary = report.get("numerical_summary", {})
        quality = report.get("quality_scorecard", {})
        
        # 1. COMPLETE COLUMN ROSTER (The macro view for the LLM)
        roster_lines = []
        for col in meta_data:
            c_name = col.get("feature")
            c_type = col.get("dtype")
            missing = col.get("missing_pct", 0)
            
            # If it's a numeric column, give the LLM the distribution stats
            if c_name in num_summary and isinstance(num_summary[c_name], dict):
                stats = num_summary[c_name]
                skew = round(stats.get("skewness") or 0, 2)
                outliers = stats.get("iqr_outlier_pct", 0)
                mean = round(stats.get("mean") or 0, 2)
                median = round(stats.get("median") or 0, 2)
                roster_lines.append(f"  - {c_name} [{c_type}]: {missing}% missing | {outliers}% outliers | Skew: {skew} | Mean: {mean}, Median: {median}")
            else:
                # If categorical, give the LLM cardinality
                cardinality = col.get("cardinality", 0)
                roster_lines.append(f"  - {c_name} [{c_type}]: {missing}% missing | Cardinality: {cardinality} unique values")
                
        roster_str = "\n".join(roster_lines)

        # 2. SEVERE WARNINGS (To ensure the LLM doesn't miss the biggest traps)
        useless_cols = [c['feature'] for c in meta_data if c.get('is_constant', False) or (c.get('is_fully_unique', False) and c.get('dtype_kind') in ['O', 'U'])]
        
        corr_data = report.get("correlation_analysis", {}).get("high_correlation_pairs", [])
        corr_str = "\n".join([f"  - {p['feature_a']} & {p['feature_b']}: {p['correlation']} correlation" for p in corr_data]) or "  - None"

        # Assemble the final LLM prompt payload
        summary = f"""
EDA Complete for '{path.name}'
Shape: {overview.get('rows')} rows, {overview.get('columns')} columns
Duplicate Rows: {overview.get('duplicates', {}).get('duplicate_rows')}
Overall Health Grade: {quality.get('overall_grade')} ({quality.get('overall_score')}/100)

COMPLETE COLUMN ROSTER (Review all data before planning):
{roster_str}

CRITICAL SYSTEM WARNINGS:
- High Risk Candidates for Dropping (Constant or Fully Unique): {', '.join(useless_cols) if useless_cols else 'None'}
- Severe Multicollinearity Detected:
{corr_str}

Instruction to Agent: 
1. Review the Complete Column Roster above to understand the full context of the dataset.
2. Formulate a comprehensive data cleaning plan. 
3. Use the distribution stats (skewness, mean vs median) to logically justify your imputation strategies.
4. Execute the appropriate data cleaning tools.
"""
        
        # --- SAVE PROMPT TO FILE FOR DEBUGGING ---
        # with open(out_dir / "latest_llm_payload.txt", "w", encoding="utf-8") as f:
        #     f.write(summary.strip())
        # -----------------------------------------

        return summary.strip()

    except Exception as e:
        import traceback
        return f"An error occurred during analysis: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

if __name__ == "__main__":
    mcp.run()