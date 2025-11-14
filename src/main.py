import argparse
from typing import Union
import os
import time
import torch

from fastapi import FastAPI
import pandas as pd
from external_queries import api_get_jobs
from main_inference_qwen32b import LLMInference, process_output_llms, fixed_prompt

app = FastAPI()

# System message for LLM inference
SYSTEM_MESSAGE = """You are an expert career advisor specializing in company sector classification. Your task is to analyze job titles and descriptions and provide a concise, standardized company sector label. The label should be chosen from the list detailed below, choosing the label that represents the best match. 

The company sector label should be the best candidate among the following list (in python format):

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
    
    # Save dataframe to Data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "Data")
    os.makedirs(output_dir, exist_ok=True)
    
    raw_data_path = os.path.join(output_dir, "raw_job_data_from_api.csv")
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
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="LLM model name (default: meta-llama/Meta-Llama-3-8B-Instruct)"
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
    df = get_jobs_recommendations(args.item_id, args.options)
    print(f"Retrieved {len(df)} jobs")
    print(f"DataFrame columns: {list(df.columns)}\n")
    
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
        output_dir = os.path.join(base_dir, "Data")
        output_path = os.path.join(output_dir, "job_data_with_sectors.csv")
        
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
# Use a different model:
#   CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name Qwen/Qwen2.5-32B-Instruct
#
# Skip inference and only download data:
#   python src/main.py --skip_inference
#
# Run without quantization (requires more GPU memory):
#   CUDA_VISIBLE_DEVICES=0,1 python src/main.py --use_quantization False
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