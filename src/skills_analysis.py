import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import Normalizer
import warnings
import json
import os
import math
import sys
import logging
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Import visualization function from separate module
from visualizations import visualize_skills_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('integration_analysis.log')  # File output
    ]
)
logger = logging.getLogger(__name__)



def preprocess_data(source_data, llm_output_file="occupation_sector_llm.csv", save=False, my_company_rows=20, load_files=True, create_visualizations=True, n_skills=20):
    """
    Process job data and generate recommendations using LLM classification.
    
    Args:
        source_data (dict): Job data from SKILLAB Tracker API
        llm_output_file (str): Output filename for LLM classification results
        save (bool): Whether to save results to Excel file
        my_company_rows (int): Number of rows to assign to "my_company" dataset (default: 20)
        load_files (bool): Whether to load existing files or regenerate (default: True)
        create_visualizations (bool): Whether to create visualizations (default: True)
        n_skills (int): Number of top skills to compare and visualize (default: 20)
        
    Returns:
        dict: Dictionary containing 'comparison', 'my_company_data', 'competition_data'
    """
    try:
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "Data", "integration")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        temp_skills_data = os.path.join(output_dir, "raw_skills_data.csv")

        # Load skills mapping
        skill_labels_dict = create_skills_level_dict()
        logger.debug(f"Loaded {len(skill_labels_dict)} skill labels")
        # Show first 5 skill labels as a sample
        sample_skills = dict(list(skill_labels_dict.items())[:5])
        logger.info(f"Sample skill labels: {sample_skills}")

        # Process source data
        record_count = source_data.get("count", 0)
        logger.info(f'Retrieving {record_count} records from the SKILLAB Tracker API')
        number_of_pages = int(math.ceil(record_count / 300))
        logger.debug(f"Number of pages: {number_of_pages}")

        # Create DataFrame and save raw data
        df = pd.DataFrame(source_data["items"])
        logger.info(f"Created DataFrame with {len(df)} rows")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame head:\n{df.head()}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        """DataFrame columns: ['skills', 'occupations', 'id', 'organization', 'title', 'description', 'experience_level', 'type', 'location', 'location_code', 'upload_date', 'source', 'source_id']"""
        # breakpoint()

        # skills_data = df[["id","skills", "source_id", "location_code"]]
        
        # Create skills matrix using MultiLabelBinarizer (efficient approach)
        logger.info("Creating skills matrix using MultiLabelBinarizer")
        
        # Ensure skills is a list type
        skills_lists = df["skills"].apply(lambda x: x if isinstance(x, list) else ([x] if x else []))
        logger.info(f"Processing {len(skills_lists)} job entries")
        
        # Create binary matrix
        mlb = MultiLabelBinarizer()
        skills_binary = mlb.fit_transform(skills_lists)
        
        logger.info(f"Found {len(mlb.classes_)} unique skills across all entries")
        logger.debug(f"Sample skills: {list(mlb.classes_[:10])}")
        
        # Create DataFrame with id column and skill columns
        skills_binary_df = pd.DataFrame(
            skills_binary,
            columns=mlb.classes_
        )

        # Concatenate with id, source_id, location_code columns
        skills_matrix_df = pd.concat([
            df[["id", "source_id", "location_code"]].reset_index(drop=True),
            skills_binary_df
        ], axis=1)
        
        logger.info(f"Created skills matrix with shape: {skills_matrix_df.shape} (rows x columns)")
        logger.debug(f"Skills matrix columns (first 10): {list(skills_matrix_df.columns[:11])}...")
        logger.debug(f"Skills matrix head:\n{skills_matrix_df.head()}")
        

        
        # Save skills matrix to CSV
        skills_binary_matrix = os.path.join(output_dir, "skills_matrix.csv")
        skills_matrix_df.to_csv(skills_binary_matrix, index=False, encoding='utf-8')
        logger.info(f"Saved skills matrix to: {skills_binary_matrix}")
        
        # Split skills matrix into "my_company" and "competition" datasets
        logger.info(f"Splitting skills matrix: first {my_company_rows} rows -> my_company, rest -> competition")
        
        # Validate my_company_rows parameter
        total_rows = len(skills_matrix_df)
        if my_company_rows > total_rows:
            logger.warning(f"my_company_rows ({my_company_rows}) exceeds total rows ({total_rows}). Using all rows for my_company.")
            my_company_rows = total_rows
        
        # Split the dataframe
        my_company_df = skills_matrix_df.iloc[:my_company_rows].copy()
        competition_df = skills_matrix_df.iloc[my_company_rows:].copy()
        
        logger.info(f"my_company dataset shape: {my_company_df.shape}")
        logger.info(f"competition dataset shape: {competition_df.shape}")
        

                
        # # Save split datasets
        # my_company_path = os.path.join(output_dir, "skills_matrix_my_company.csv")
        # competition_path = os.path.join(output_dir, "skills_matrix_competition.csv")
        
        # my_company_df.to_csv(my_company_path, index=False, encoding='utf-8')
        # competition_df.to_csv(competition_path, index=False, encoding='utf-8')
        
        # logger.info(f"Saved my_company dataset to: {my_company_path}")
        # logger.info(f"Saved competition dataset to: {competition_path}")
        
        # Create aggregated dataframes with sum of all rows
        logger.info("Creating aggregated sum dataframes")
        
        # Identify metadata columns (non-skill columns)
        metadata_cols = ["id", "source_id", "location_code"]
        skill_cols = [col for col in skills_matrix_df.columns if col not in metadata_cols]
        
        # Define file paths
        my_company_sum_path = os.path.join(output_dir, "skills_matrix_my_company_sum.csv")
        competition_sum_path = os.path.join(output_dir, "skills_matrix_competition_sum.csv")
        
        if not load_files:
            # Create aggregated dataframes using the external function
            my_company_sum = create_aggregated_skills_dataframe(my_company_df, skill_cols)
            competition_sum = create_aggregated_skills_dataframe(competition_df, skill_cols)

            logger.info(f"my_company aggregated sum shape: {my_company_sum.shape}")
            logger.info(f"competition aggregated sum shape: {competition_sum.shape}")
            
            # Save aggregated datasets
            my_company_sum.to_csv(my_company_sum_path, index=False, encoding='utf-8')
            competition_sum.to_csv(competition_sum_path, index=False, encoding='utf-8')
        else:
            my_company_sum = pd.read_csv(my_company_sum_path)
            competition_sum = pd.read_csv(competition_sum_path)
            logger.info(f"Loaded my_company aggregated sum from: {my_company_sum_path}")
            logger.info(f"Loaded competition aggregated sum from: {competition_sum_path}")
        
        skills_comparison = compare_skills(my_company_sum, competition_sum, n_skills=n_skills)
        logger.info(f"Skills comparison completed for top {n_skills} skills")
        
        # Create visualizations if requested
        if create_visualizations:
            vis_output_dir = os.path.join(output_dir, "visualizations")
            logger.info(f"Creating visualizations in {vis_output_dir}")
            visualize_skills_comparison(
                comparison_results=skills_comparison,
                my_company_df=my_company_sum,
                competition_df=competition_sum,
                output_dir=vis_output_dir,
                n_skills=n_skills,
                normalize=True,  # Normalize by default
                plots_to_create=None  # Create all plots by default (or specify ['differences', 'unique', 'bar_chart', 'radar', 'bubble'])
            )
        
        # Return results and dataframes for further analysis
        return {
            'comparison': skills_comparison,
            'my_company_data': my_company_sum,
            'competition_data': competition_sum
        }


        
    except KeyError as e:
        logger.error(f"Missing expected key in source data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in recommend_jobs: {e}", exc_info=True)
        raise




def create_skills_level_dict():
    """Create a dictionary mapping skill URIs to their preferred labels."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_file = os.path.join(base_dir, 'new_ESCO_mapping.xlsx')
        logger.info(f"Loading skills mapping from: {mapping_file}")
        
        skills_mapping = pd.read_excel(mapping_file)
        logger.debug(f"Loaded {len(skills_mapping)} skills from mapping file")
        
        # Convert strings to lists
        skills_mapping['skills_levels'] = skills_mapping['skills_levels'].apply(eval)
        skills_mapping['knowledge_levels'] = skills_mapping['knowledge_levels'].apply(eval)
        skills_mapping['traversal_levels'] = skills_mapping['traversal_levels'].apply(eval)
        skills_mapping['skills_ancestors'] = skills_mapping['skills_ancestors'].apply(eval)
        skills_mapping['knowledge_ancestors'] = skills_mapping['knowledge_ancestors'].apply(eval)
        skills_mapping['traversal_ancestors'] = skills_mapping['traversal_ancestors'].apply(eval)
        skills_mapping['children'] = skills_mapping['children'].apply(eval)

        skill_labels_dict = {}
        for index, row in skills_mapping.iterrows():
            skill_labels_dict[row['conceptUri']] = [row['preferredLabel'], row['skills_levels'], row['knowledge_levels'], row['traversal_levels']]
        
        logger.info(f"Created skills dictionary with {len(skill_labels_dict)} entries")
        return skill_labels_dict
        
    except FileNotFoundError as e:
        logger.error(f"Skills mapping file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating skills level dict: {e}")
        raise


def create_aggregated_skills_dataframe(df, skill_cols):
    """
    Create an aggregated dataframe with sum of skills and additional level rows.
    
    Args:
        df (pd.DataFrame): Input dataframe with skill columns
        skill_cols (list): List of skill column names to aggregate
        
    Returns:
        pd.DataFrame: Aggregated dataframe with sum and level information
    """
    # Create sum dataframe with only skill columns
    sum_df = df[skill_cols].sum().to_frame().T
    
    # Get skills level dictionary
    skills_level_dict = create_skills_level_dict()
    skills_level_dict = {k: v for (k, v) in skills_level_dict.items() if k in sum_df.columns}
    
    # Add rows for skills_levels, knowledge_levels, and traversal_levels
    for i in [1, 2, 3]:
        new_row = {key: value[i] for key, value in skills_level_dict.items()}
        new_row_df = pd.DataFrame([new_row])
        sum_df = pd.concat([sum_df, new_row_df], axis=0, ignore_index=True)
    
    # Rename columns to preferred labels
    sum_df = sum_df.rename(columns=lambda x: skills_level_dict.get(x)[0])
    
    # Sort columns by sum values in descending order
    sorted_columns = sum_df.iloc[0].sort_values(ascending=False).index
    sum_df = sum_df[sorted_columns]
    
    # Insert SKILL LEVEL column at the beginning
    sum_df.insert(0, 'SKILL LEVEL', ["sum", "skills_levels", "knowledge_levels", "traversal_levels"])
    
    return sum_df


def compare_skills(sorted_my_ad_df: pd.DataFrame = None, 
                    renamed_sorted_normalized_df: pd.DataFrame = None,
                    n_skills: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Compare top n skills between job ad and sector average using the approach from filtered_jobs_analysis.
    
    Args:
        sorted_my_ad_df (pd.DataFrame, optional): Job ad skills data. If None, uses self.sorted_my_ad_df
        renamed_sorted_normalized_df (pd.DataFrame, optional): Sector skills data. If None, uses self.renamed_sorted_normalized_df
        n_skills (int): Number of top skills to compare
        
    Returns:
        Dict containing DataFrames for common, only-in-ad, and only-in-sector skills
    """

        
    if sorted_my_ad_df is None or renamed_sorted_normalized_df is None:
        print("Both job ad and sector data must be processed first.")
        return None
    
    # Get only the first row (sum row) and exclude 'SKILL LEVEL' column if present
    my_ad_first_row = sorted_my_ad_df.iloc[0].copy()
    sector_first_row = renamed_sorted_normalized_df.iloc[0].copy()
    
    # Remove 'SKILL LEVEL' column if it exists
    if 'SKILL LEVEL' in my_ad_first_row.index:
        my_ad_first_row = my_ad_first_row.drop('SKILL LEVEL')
    if 'SKILL LEVEL' in sector_first_row.index:
        sector_first_row = sector_first_row.drop('SKILL LEVEL')
    
    # Convert to numeric (handles string values from CSV)
    my_ad_first_row = pd.to_numeric(my_ad_first_row, errors='coerce')
    sector_first_row = pd.to_numeric(sector_first_row, errors='coerce')
    
    # Get skill columns (first n_skills)
    my_ad_skills = my_ad_first_row.index[:n_skills]
    sector_skills = sector_first_row.index[:n_skills]
    
    # Find common skills - well represented
    common_columns = my_ad_skills.intersection(sector_skills)
    print("Common skills:", common_columns)
    
    # Find columns unique to my company
    only_in_my_company = my_ad_skills.difference(sector_skills)
    print("\nSkills only in my job ad:", only_in_my_company)
    
    # Find columns unique to the sector - missing in my company
    only_in_sector = sector_skills.difference(my_ad_skills)
    print("\nMissing skills (only in sector):", only_in_sector)
    
    # Create results dictionary
    results = {}
    
    # For common skills, calculate and sort by probability difference
    if len(common_columns) > 0:  # Only create DataFrame if there are common skills
        common_skills_diff = pd.DataFrame()
        common_skills_diff['skill'] = common_columns
        common_skills_diff['my_ad_prob'] = my_ad_first_row[common_columns].values
        common_skills_diff['sector_prob'] = sector_first_row[common_columns].values
        common_skills_diff['difference'] = common_skills_diff['my_ad_prob'] - common_skills_diff['sector_prob']
        common_skills_diff = common_skills_diff.sort_values('difference', ascending=False)
        
        print("\nCommon skills sorted by probability difference:")
        print(common_skills_diff)
        results['common'] = common_skills_diff
    else:
        print("\nNo common skills found between the two distributions")
        results['common'] = pd.DataFrame()
    
    # Skills only in job ad
    if len(only_in_my_company) > 0:
        only_ad_df = pd.DataFrame()
        only_ad_df['skill'] = only_in_my_company
        only_ad_df['my_ad_prob'] = my_ad_first_row[only_in_my_company].values
        only_ad_df = only_ad_df.sort_values('my_ad_prob', ascending=False)
        results['only_in_ad'] = only_ad_df
    else:
        results['only_in_ad'] = pd.DataFrame()
    
    # Skills only in sector (missing skills)
    if len(only_in_sector) > 0:
        only_sector_df = pd.DataFrame()
        only_sector_df['skill'] = only_in_sector
        only_sector_df['sector_prob'] = sector_first_row[only_in_sector].values
        only_sector_df = only_sector_df.sort_values('sector_prob', ascending=False)
        results['only_in_sector'] = only_sector_df
    else:
        results['only_in_sector'] = pd.DataFrame()
    
    # Print summary
    print(f"\nSkills comparison summary (top {n_skills}):")
    print(f"Common skills: {len(common_columns)}")
    print(f"Skills only in job ad: {len(only_in_my_company)}")
    print(f"Missing skills (only in sector): {len(only_in_sector)}")
    
    return results



# ----------------- LLM Classification -----------------

# -----------------

def classify_jobs_with_llm(
    input_file,
    output_file=None,
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    use_quantization=False
):
    """
    Classify jobs using LLM to extract occupational and sector labels.
   
    Args:
        input_file (str): Path to input CSV file with columns: title, description, source_id
        output_file (str, optional): Path to save output CSV. If None, results are returned but not saved.
        model_name (str): Name of the LLM model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
        use_quantization (bool): Whether to use quantization (default: False)
       
    Returns:
        pd.DataFrame: DataFrame with columns: source_id, occupations, sectors
       
    Example usage:
        # Without saving (returns DataFrame only)
        df_results = classify_jobs_with_llm(
            input_file="Data/filtered_jobs/occupations_SE_GAP_analysis.csv",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            use_quantization=True
        )
       
        # With saving
        df_results = classify_jobs_with_llm(
            input_file="Data/filtered_jobs/occupations_SE_GAP_analysis.csv",
            output_file="Data/classified_ads/filtered_jobs_classifications_ESCO.csv",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            use_quantization=True
        )
    """
    try:
        logger.info(f"Starting LLM classification with model: {model_name}")
        logger.debug(f"Input file: {input_file}")
        logger.debug(f"Output file: {output_file}")
        logger.debug(f"Quantization enabled: {use_quantization}")
        
        # Import the classification function from the organizational_skills_recommender module
        base_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        org_skills_path = os.path.join(base_parent_dir, 'organizational_skills_recommender')
        sys.path.insert(0, org_skills_path)
        logger.debug(f"Added to sys.path: {org_skills_path}")
        
        from llm_classification import classify_jobs
        
        # Run classification
        df_results = classify_jobs(
            input_file=input_file,
            output_file=output_file,
            model_name=model_name,
            use_quantization=use_quantization
        )
        
        logger.info(f"LLM classification completed. Results shape: {df_results.shape}")
        return df_results
        
    except ImportError as e:
        logger.error(f"Failed to import llm_classification module: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in classify_jobs_with_llm: {e}", exc_info=True)
        raise


def recommend_jobs(source_data, llm_output_file="occupation_sector_llm.csv", save=False, my_company_rows=50):
    """
    Process job data and generate recommendations using LLM classification.
    
    Args:
        source_data (dict): Job data from SKILLAB Tracker API
        llm_output_file (str): Output filename for LLM classification results
        save (bool): Whether to save results to Excel file
        my_company_rows (int): Number of rows to assign to "my_company" dataset (default: 50)
        
    Returns:
        DataFrame: LLM classification results
    """
    try:
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "Data", "integration")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        temp_raw_data = os.path.join(output_dir, "raw_data.csv")

        # Load skills mapping
        skill_labels_dict = create_skills_level_dict()
        logger.debug(f"Loaded {len(skill_labels_dict)} skill labels")
        # Show first 5 skill labels as a sample
        sample_skills = dict(list(skill_labels_dict.items())[:5])
        logger.info(f"Sample skill labels: {sample_skills}")

        # Process source data
        record_count = source_data.get("count", 0)
        logger.info(f'Retrieving {record_count} records from the SKILLAB Tracker API')
        number_of_pages = int(math.ceil(record_count / 300))
        logger.debug(f"Number of pages: {number_of_pages}")

        # Create DataFrame and save raw data
        df = pd.DataFrame(source_data["items"])
        logger.info(f"Created DataFrame with {len(df)} rows")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame head:\n{df.head()}")
        breakpoint()
        
        raw_data = df[["source_id", "title", "location", "description"]]
        raw_data.to_csv(temp_raw_data, index=False, encoding='utf-8')
        logger.info(f"Saved raw data to: {temp_raw_data}")
        
        # Run LLM classification and save results
        output_file = os.path.join(output_dir, llm_output_file)
        logger.info(f"Starting LLM classification, output file: {output_file}")

        llm_results = classify_jobs_with_llm(
            temp_raw_data,
            output_file=output_file,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            use_quantization=False
        )
        logger.info("LLM classification completed successfully")

        # Optional: Save results to Excel
        if save:
            logger.info("Saving results to Excel...")
            logger.debug(f"DataFrame head:\n{df.head()}")

            # Flatten lists (skills, occupations) to comma-separated strings for easier viewing
            df["skills"] = df["skills"].apply(lambda x: ", ".join(x))
            df["occupations"] = df["occupations"].apply(lambda x: ", ".join(x))
            logger.debug(f"Flattened columns - sample:\n{df[['id', 'title', 'location', 'skills']].head()}")
            
            # Save first 50 entries to Excel
            output_path = os.path.join(output_dir, "first_50_jobs.xlsx")
            df.head(50).to_excel(output_path, index=False)
            logger.info(f"Saved first 50 entries to: {output_path}")
        
        return llm_results
        
    except KeyError as e:
        logger.error(f"Missing expected key in source data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in recommend_jobs: {e}", exc_info=True)
        raise