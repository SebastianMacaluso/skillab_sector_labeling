
# CUDA_VISIBLE_DEVICES=3 python main.py --use_quantization True --max_samples 100

import argparse
from typing import Union
import os
import time
import torch
from datetime import datetime

from fastapi import FastAPI
import pandas as pd
from external_queries import api_get_jobs
from main_inference_qwen32b import LLMInference, process_output_llms, fixed_prompt

app = FastAPI()

# System message for LLM inference
SYSTEM_MESSAGE = """You are an expert career advisor specializing in company sector classification. Your task is to analyze job titles and descriptions to identify the COMPANY'S SECTOR (industry), not the job role itself.

CRITICAL INSTRUCTIONS:
1. You MUST select exactly ONE label from the list below - no exceptions.
2. Focus on the COMPANY'S industry/sector based on clues in the job description, not the job position.
3. For example: An "accountant" position at a pharmaceutical company should be classified as "Human health and social work activities" (the pharma sector), NOT "Consultancy, marketing, accounting and legal services" (the job role).
4. Look for company context clues: products, services, company name, industry keywords in the description.
5. DO NOT include any thinking, reasoning, or explanation in your 
response.
6. DO NOT use <think> tags or show your thought process.
7. Output ONLY the sector label in the exact format shown in the examples.

You MUST choose exactly ONE label from the following list:

company sector labels = [
"Manufacturing",
"Administrative and support service activities",
"Consultancy, marketing, accounting and legal services",
"Human health and social work activities",
"Wholesale and retail trade",
"Repair of motor vehicles and motorcycles",
"Information and communication",
"Employment activities",
"Education",
"Transportation and storage",
"Financial and insurance activities",
"Accommodation and food service activities",
"Construction",
"Public administration and defence",
"Other service activities",
"Other professional activities",
"Technical, engineering and R&D activities",
"Electricity, gas, steam and air conditioning supply",
"Real estate activities",
"Arts entertainment and recreation",
"Water supply, sewerage, waste management and remediation activities",
"Agriculture, forestry and fishing",
"Mining and quarrying"]

"""


def perform_sector_inference(df, model, model_name="meta-llama/Meta-Llama-3-8B-Instruct", use_quantization=True, max_samples=None):
    """
    Perform sector classification inference on job data.
    
    Args:
        df: DataFrame with 'title' and 'description' columns
        model: Loaded LLM model (None if using GPT)
        model_name: Name of the model being used
        use_quantization: Whether quantization is used
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        DataFrame with added 'sector' column
    """
    print(f"\n{'='*60}")
    print(f"Starting sector classification inference")
    print(f"Model: {model_name}")
    print(f"Quantization: {use_quantization}")
    print(f"{'='*60}\n")
    
    results = []
    k_total = 0
    
    # Determine number of samples to process
    num_samples = min(max_samples, len(df)) if max_samples else len(df)
    print(f"Processing {num_samples} out of {len(df)} job listings\n")
    
    start_time = time.time()
    
    for i in range(num_samples):
        title = df.iloc[i]['title']
        description = df.iloc[i]['description']
        
        # Create prompt
        prompt = fixed_prompt + "Position title: " + str(title) + "\n Description: " + str(description)
        
        # Get model output
        if model_name != 'gpt':
            output = model.model_generate_pipeline(prompt, system_message=SYSTEM_MESSAGE)
            output = list(output)
        else:
            from main_inference_qwen32b import get_response_gpt
            output = get_response_gpt(prompt)
            output = [output]
        
        # Process output
        try:
            sector_label, k_total = process_output_llms(output, k=k_total)
            results.append(sector_label)
        except ValueError as e:
            print(f"Error processing row {i}: {e}")
            results.append("NaN")
            k_total += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} jobs...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per job: {total_time/num_samples:.2f} seconds")
    print(f"Number of non-matched outputs: {k_total}")
    print(f"{'='*60}\n")
    
    # Add sector column to dataframe
    df_result = df.iloc[:num_samples].copy()
    df_result['sector'] = results
    
    return df_result


@app.get("/get_recommendations/{item_id}")
def get_jobs_recommendations(item_id: int, options: Union[str, None] = None):
    """
    Get job recommendations with skills analysis.
    
    Args:
        item_id: Item identifier
        options: Additional options
    """
    result = api_get_jobs(
            page=1,
            page_size=97,
            keywords=["python", "java"],
            keywords_logic="and",
            skill_ids_logic="or",
            occupation_ids_logic="or",
        )
    print("result: ", result.keys())
    df = pd.DataFrame(result["items"])
    
    # Save dataframe to Data/sector directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "Data", "sector")
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_data_path = os.path.join(output_dir, f"raw_job_data_from_api_{timestamp}.csv")
    df.to_csv(raw_data_path, index=False, encoding='utf-8')
    print(f"Saved raw data to: {raw_data_path}")
    
    return df
    



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Get job data and perform sector classification")
    parser.add_argument("--item_id", type=int, default=1, help="Item identifier (default: 1)")
    parser.add_argument("--options", type=str, default=None, help="Additional options (optional)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # default="Qwen/Qwen3-8B",
        # default="Qwen/Qwen3-32B",
        help="LLM model name (default: Qwen/Qwen2.5-32B-Instruct)"
        # Other model options:
        # - "meta-llama/Meta-Llama-3-8B-Instruct"
        # - "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # - "meta-llama/Llama-2-7b-chat-hf"
        # - "mistralai/Mistral-7B-Instruct-v0.2"
        # - "gpt" (for OpenAI GPT models)
    )
    parser.add_argument(
        "--use_quantization",
        type=str,
        default='True',
        help="Use quantization (True/False, default: True)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of jobs to process (default: all)"
    )
    parser.add_argument(
        "--skip_inference",
        action='store_true',
        help="Skip inference and only download/save data"
    )
    
    args = parser.parse_args()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Get job data
    print("Fetching job data from API...")
    try:
        df = get_jobs_recommendations(args.item_id, args.options)
        print(f"Retrieved {len(df)} jobs from API")
        print(f"DataFrame columns: {list(df.columns)}\n")
    except Exception as e:
        print(f"API request failed: {e}")
        print("Loading data from local fallback file...")
        
        # Fallback to local CSV file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_path = os.path.join(base_dir, "Data", "sector", "raw_job_data_from_api.csv")
        
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
            print(f"Successfully loaded {len(df)} jobs from fallback file: {fallback_path}")
            print(f"DataFrame columns: {list(df.columns)}\n")
        else:
            print(f"Error: Fallback file not found at {fallback_path}")
            print("Cannot proceed without data. Exiting.")
            exit(1)
    
    if not args.skip_inference:
        # Load model
        model_name = args.model_name
        use_quantization = str(args.use_quantization).lower() == 'true'
        
        print(f"Loading model: {model_name}")
        print(f"Quantization: {use_quantization}\n")
        
        if model_name != 'gpt':
            model = LLMInference.load_model_from_pretrained_pipeline(
                model_name, 
                use_quantization=use_quantization
            )
        else:
            model = None
        
        # Perform inference
        df_with_sectors = perform_sector_inference(
            df, 
            model, 
            model_name=model_name,
            use_quantization=use_quantization,
            max_samples=args.max_samples
        )
        
        # Save results
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "Data", "sector")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with model name and number of jobs
        model_name_clean = model_name.replace('/', '_').replace('.', '_')
        num_jobs = len(df_with_sectors)
        output_filename = f"job_data_with_sectors_{model_name_clean}_{num_jobs}_jobs.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        df_with_sectors.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved classified data to: {output_path}")
        
        # Display sample results
        print("\nSample results:")
        print(df_with_sectors[['title', 'sector']].head(10))
    else:
        print("Skipping inference (--skip_inference flag set)")

    # To run with uvicorn instead:
    # uvicorn main:app --host 0.0.0.0 --port 8012



# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
#
# Run locally with sector classification inference:
#   python src/main.py
#
# Process only first 10 jobs (for testing):
#   python src/main.py --max_samples 10
#
# Use a different model (Qwen 32B):
#   CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name Qwen/Qwen2.5-32B-Instruct
#
# Use Llama 3.1:
#   CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
#
# Use Mistral:
#   CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name mistralai/Mistral-7B-Instruct-v0.2
#
# Use GPT (OpenAI):
#   python src/main.py --model_name gpt
#
# Skip inference and only download data:
#   python src/main.py --skip_inference
#
# Run without quantization (requires more GPU memory):
#   CUDA_VISIBLE_DEVICES=0,1 python src/main.py --use_quantization False
#
# Note: If API connection fails, the script automatically falls back to:
#   Data/sector/raw_job_data_from_api.csv
#
# ==============================================================================
# FastAPI Server Usage:
# ==============================================================================
#
# Start the server:
#   uvicorn main:app --host 0.0.0.0 --port 8012
#
# Access endpoints:
#   - Main endpoint: http://localhost:8012/get_recommendations/1
#   - Interactive API docs (Swagger UI): http://localhost:8012/docs
#   - Alternative docs (ReDoc): http://localhost:8012/redoc
#
# Using Swagger UI (/docs):
#   1. Navigate to http://localhost:8012/docs
#   2. Find /get_recommendations/{item_id} endpoint
#   3. Click "Try it out"
#   4. Fill in parameters (item_id, options)
#   5. Click "Execute"
#   6. View response with job data
#
# ==============================================================================