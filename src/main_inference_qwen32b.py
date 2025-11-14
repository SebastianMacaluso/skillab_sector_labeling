import argparse
import pathlib
import sys

import torch
import time
import os
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(path))

import pandas as pd

# import pathlib
# import sys
import re
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


from os import listdir
from os.path import isfile, join


def load_data(path, filename):
    
    df = pd.read_csv(path+"/"+filename, encoding='utf-8')
    occupations = list(df['title'])
    descriptions = list(df['description'])
    source_id = list(df['source_id'])

    my_prompts = {}
    my_prompts['occupations'] = occupations
    my_prompts['descriptions'] = descriptions
    my_prompts['source_id'] = source_id
    # print(my_prompts['source_id'][0:10])

    return my_prompts



fixed_prompt = """Based on the following job title and description, provide a single, concise company sector label that represents the best match. Only provide the label as shown in the examples, nothing else.


Example 1:
Position title: Software Engineer (React).\n 
Description: Join INTRALOT as a Software Engineer (React)!\nYour Role:\n\nAs a Software Engineer (React) at INTRALOT, you'll play a pivotal role in our Digital iLottery Projects!\n\nMore specifically, some of your main tasks will be: \n\n Participating in all phases of the development life cycle, focusing on coding and unit testing.\n Developing and maintaining functional and stable applications to meet company's needs.\n Designing and implementing highly scalable solutions for unpredictable traffic patterns.\n Following standard code practices and building reusable code and libraries for future use.\n Implementing security and data protection mechanisms.\n Optimizing the application for speed and scalability.\n Contributing to application strategy and planning.\nStaying up to date with emerging technologies and formulating concepts and ideas for additional products, tools and services to be provided.\nTo get this role, you should match the following criteria:\n\n You hold a University or College Degree in Computer Science, Software Engineering or relevant technology-related field.\n At least three (3) years of experience in Software Development.\n\n\n Strong working experience in Javascript (ES6), HTML5, CSS3, SASS and React.JS/MobX\n Experience in using version control software such as Git.\n Excellent understanding of the entire web development process (Design, Development and Deployment).You have an excellent command of the Greek & English language, both written and spoken\n  We work better as a TEAM. So, you also have to be an excellent team player.\n\nNice to have, but not necessary:\nExperience in developing and maintaining high-quality React Native applications using clean code.\n\nAbout Us: INTRALOT, a publicly listed company established in 1992, is a leading gaming solutions supplier and operator active in 39 regulated jurisdictions worldwide. With a global workforce of approximately 1,700 employees, INTRALOT is committed to redefine innovation and quality of services in the lottery and gaming sector, while supporting operators in raising funds for good causes. In 2023, INTRALOT was distinguished as a top ten most attractive employer brands in Greece.\nINTRALOT has been awarded the prestigious WLA Responsible Gaming Framework Certification by the World Lottery Association. We always act with integrity and enjoy what we do.\n\nWhy Join Us:\n A competitive compensation package combined with additional benefits.\n Hybrid working model.\n Private Insurance for you and your family. \n Modern facilities, parking inside, enticing restaurant and corporate bus.\n An extensive training program upon induction and throughout employment.\n Exceptional opportunities to learn and develop in a multinational environment.\n Prospects for professional growth both locally and globally. \nEqual Opportunity Employer: We celebrate diversity and are proud to be an Equal Opportunity Employer. We welcome applicants from all backgrounds and do not discriminate based on race, color, religion, gender, sexual orientation, national origin, age, marital status, medical condition, disability, or any other protected status. Our premises in Paiania, Attica, are accessible to individuals with disabilities. Join us in shaping the future of gaming and lottery solutions!\n 
Sector label: Arts entertainment and recreation


Example 2:
Position title: Data project manager 
Description: Company descriptionGTT is the world leader in containment technologies for maritime transport and storage of Liquefied Natural Gas (LNG). Today, energy transition and innovation are at the heart of our concerns. GTT achieves more than 300M€ turnover per year with more than 500 employees spread across the world. To support the growth of the group, we are recruiting a Data Project Manager, attached to the IT Department.Job description CONTEXT:At within the Digital and Information Systems Department, under the responsibility of the AMOA Manager, the Data project manager is responsible for designing, developing and implementing a global data management strategy aimed at optimizing data management within of GTT. Its role is essential to ensure the collection, storage, security and efficient use of data in compliance with business needs and current regulations. MAIN RESPONSIBILITIES: Business Needs Analysis: Work in collaboration with the various business teams to identify their data needs.Define data management solutions adapted to support the company's objectives.Definition of the Data Management Strategy:Develop a global data management strategy aligned with the objectives of GTT.Establish KPIs to evaluate and measure the effectiveness of the strategy. Data Management: Establish processes for the collection, quality, storage and security of data. Ensure the consistency and integrity of data across the company.Selection and Implementation of Technological Tools:Evaluate, choose and deploy the appropriate technological tools and platforms for data management.Collaborate with technical teams to guarantee effective integration.Communication and Training:Make employees aware of the importance of data management.Organize training as needed and communicate regularly on progress.Monitoring and Evaluation:Monitor KPIs and adjust the strategy accordingly.Present periodic reports to management on the progress of the project.Qualifications SIGNIFICANT SKILLS: Technical skills: IT project and program management, certification in PMP or Prince 2 is a plus; In-depth knowledge of data technologies: (BI tools, data security, Cloud Computing, Storage system, MDM, Tools of ETL..Etc.);In-depth knowledge of data regulations;Experience in managing integration projects (CDC, AO, contractualization, integration, deployment);Proficiency in oral and written English is desired. Human skills: Excellent communication skills; Ability to work in a team and collaborate with different stakeholders; Rigor, ability to solve complex problems and results orientation. LEVEL OF STUDIES & EXPERIENCE: Graduate engineering school in engineering IT (specialized in data or equivalent); Significant experience in data management as a project manager or project manager; Additional information CONTRACT AND BENEFITS: This position is to be filled on a permanent basis You will benefit from the following package: Base salary (To be defined according to profile)Bonus on individual objective (20%)Incentive (15%)Participation (8%)Company contribution to the PEE of €3,700You will benefit from a hybrid teleworking schedule (2 days/week)Our green campus of 8 hectares has a company restaurantWe are connected to the RER B and C as well as the Transilien N and U lines, and provide you with free shuttles (at the St Remy-les Chevreuse and Versailles Chantier stations)We We also offer one-off access to our Parisian premises. Join a company involved in the environmental transition!
Sector label: Transportation and storage

Example 3:
Position title: NURSE IN OPERATING ROOM (M/F) 
Description: Want to escape? Going towards new horizons? Oh no you are not dreaming! Vitalis Médical Var invites you to escape to the Isle of Beauty! Recruitment specialist in the field of Health, VITALIS MEDICAL is recruiting an IBODE or Block IDE (M/F) for a fixed-term contract; Permanent contract possible...! At Vitalis Médical, you can benefit from: Loyalty program from the first hour of mission Company mutual insurance Local management based on trust, autonomy, transparency, and good humor! Sponsorship bonus Your missions Under the Responsibility of the Head of Block, your main missions are: - Ensure and monitor maintenance and instrumentation - Ensure the quality of the premises - Provide equipment, instrumentation, specific installation for the patient - Ensure care adapted to patients (reception, verification of identity, comfort, installation, control of the patient's pre-operative preparation). - Check the proper functioning of the equipment before its use (including respirators in the operating room). - Prepare and check the equipment specific to the instrumentation (sterile linen, ligature, etc.) - Master the procedures and respect the established protocols. You have free accommodation (apartment hotel) and travel expenses are covered. Morning schedule or afternoon with on-call duty once a week and one weekend per month. Profile sought State-certified operating room nurses or IDE with at least 2 years of experience in the operating room or versatile IDE with experience as a whole. You are motivated, demonstrate adaptability, involvement and great professional rigor, then this position is made for you. Possibility of continuing on other missions. The VITALIS Team is here for YOU, candidate, to take the time to listen to you, to understand your career path and your professional project, in order to better guide you towards the opportunities that suit YOU. Do not hesitate to forward this announcement to those around you. Additional information Type Contract: Fixed-term contract Working time: Full-time Salary: €20 - €30 per hour
Sector label: Human health and social work activities

Now, please provide a sector label for the following input. (Note: the output format should match exactly the following: [Sector label: label] and nothing else.)
"""


    # Job Title: Cyber Security Consulting, Associate Manager/ Manager
    # Description: We are seeking a highly skilled and experienced Assistant Manager/Manager in Cybersecurity Consulting to join our dynamic team. The ideal candidate will possess deep technical knowledge, strong leadership skills, and a proven track record in managing cybersecurity projects. This role involves working closely with clients to identify security risks, develop mitigation strategies, and implement solutions to protect their information systemsThe role involves: Lead and manage cybersecurity consulting engagements, ensuring the delivery of high-quality services to clients. Conduct comprehensive security assessments, maturity assessments, and risk assessments. Develop and implement cybersecurity strategies, policies, and procedures tailored to client needs. Collaborate with cross-functional teams to design and deploy security solutions. Stay current with emerging cybersecurity trends, threats, and technologies to inform client solutions. Prepare detailed reports and presentations for clients, summarizing findings, recommendations, and action plans. Mentor and develop junior team members, fostering a culture of continuous learning and professional growth. Assist in business development activities, including proposal writing, client presentations, and identifying new opportunities.The ideal candidate should possess: Bachelor's degree in computer science, Information Technology, Cybersecurity, or a related field preferred. Relevant certifications such as CISSP, CISM, CISA, or equivalent. 5+ years of experience in cybersecurity consulting, with a focus on areas such as risk management, incident response, and security architecture. Strong understanding of regulatory frameworks and standards, including NIST, and ISO 27001, etc. Excellent project management skills with the ability to manage multiple engagements simultaneously. Strong analytical and problem-solving skills, with the ability to think strategically and act tactically. Exceptional communication skills, both written and verbal, with the ability to convey complex technical concepts to non-technical stakeholders. Proven ability to build and maintain client relationships, demonstrating a commitment to delivering exceptional client service. Ability to travel as required to meet client needs.Only shortlisted candidates will be contacted by KPMG Talent Acquisition team, personal data collected will be used for recruitment purposes only.At KPMG in Singapore we are committed to creating a diverse and inclusive workplace. We believe that diversity of thought, background and experience strengthens relationships and delivers meaningful benefits to our people, our clients and communities. As an equal opportunity employer, all qualified applicants will receive consideration for employment regardless of age, race, gender identity or expression, colour, marital status, religion, sexual orientation, disability, or other non-merit factors. We celebrate the different talents that our people bring and support every staff member in their journey to achieve personal and professional growth. One of the ways we do this is through Take Charge: Flexi-work, our flexible working framework which enables agile and innovative teams to help deliver our business goals."""


def process_output_llms(output:list, k=0)->list:
    # Regular expression to extract the sector label
    # Pattern 1: Match "Sector label: xxx" format (with or without brackets)
    pattern1 = r"Sector label:\s*\[?\s*(.*?)(?:\]|$)"
    # Pattern 2: Match just quoted labels as fallback (e.g., '"Manufacturing"')
    pattern2 = r'^[\'"](.*?)[\'"]$'
    
    print("output: ", output)
    
    # Try pattern 1 first (standard format)
    match = re.search(pattern1, output[0], re.DOTALL | re.IGNORECASE)
    if match:
        sector_label = match.group(1).strip().strip(".").strip('"').strip("'")
    else:
        # Try pattern 2 for quoted labels without prefix
        match = re.search(pattern2, output[0].strip(), re.DOTALL)
        if match:
            sector_label = match.group(1).strip().strip(".")
        else:
            sector_label = "NaN"
            k+=1
            raise ValueError("Warning: Could not find the sector label")
    
    print("Sector Label:", sector_label)
    print("--------------------------------")

    return sector_label, k






    # # Extract occupational label
    # occupation_regex = r'Occupational label:\s*(.*?)(?:\n|$)'
    # print(f"occupation_regex: {occupation_regex}")
    # occupation_match = re.search(occupation_regex, output[0])
    # if occupation_match:
    #     occupation = occupation_match.group(1)
    # else:
    #     raise ValueError("Warning: Could not find the occupation")

    # # Extract sector label
    # sector_regex = r'Sector label:\s*(.*?)(?:\n|$)'
    # sector_match = re.search(sector_regex, output[0])
    # if sector_match:
    #     sector = sector_match.group(1)
    # else:
    #     raise ValueError("Warning: Could not find the sector")

    # print("Occupation:", occupation)  # Output: Cybersecurity specialist
    # print("Sector:", sector)         # Output: Financial services

        # # Extract SOC values from LLM output using regex
        # regex = r'SOC:\s*\[([\d\.,\s]+)\]'  
        # regex = r'(?i)optimal[_ ]soc"?\s*:\s*\[([\d\.,\s]+)\]'
        # matches = re.findall(regex, output)
        
        # if matches:
        #     # Split values and convert to float, handling whitespace
        #     sigma_soc = [float(x.strip()) for x in matches[0].split(',')]
            
        #     # Validate we got 24 hourly values
        #     if len(sigma_soc) != 24:
        #         raise ValueError(f"Warning: Expected 24 SOC values but got {len(sigma_soc)}")
        #         sigma_soc = []
        # else:
        #     print("Warning: Could not find SOC values in LLM output")
        #     print(output)
        #     regex = r'\[([\d\.,\s]+)\]'
        #     matches = re.findall(regex, output)
        #     if matches:
        #         sigma_soc = [float(x.strip()) for x in matches[0].split(',')]
        #         if len(sigma_soc) != 24:
        #             raise ValueError(f"Warning: Expected 24 SOC values but got {len(sigma_soc)}")
        #     else:
        #         raise ValueError("Warning: Could not find SOC values in LLM output")
        # return sigma_soc

def get_response_gpt(prompt):
    client = OpenAI(api_key="")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that would."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

class LLMInference:
    def __init__(self, tokenizer, model="", pipeline=""):
        if model == "" and pipeline == "":
            raise ValueError("You must provide a model or a pipeline")
        self.model = model
        self.pipeline = pipeline
        self.tokenizer = tokenizer


    @staticmethod
    def format_chat(tokenizer, prompt, system_message=""):
        if system_message == "":
            data = [{"role": "user", "content": prompt}]
        else:
            data = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        # The function tok should never generate the EOS token, however FastChat (used in vLLM) sends the full prompt as a string which might lead to incorrect tokenization of the EOS token and prompt injection. Users are encouraged to send tokens instead as described above.
        return tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)

    def model_generate_pipeline(
        self, prompts, max_gen_len=0, temperature=0.01, top_p=1, max_new_tokens=1000, system_message=""
    ):
        if self.pipeline == "":
            raise ValueError("You must provide a pipeline")
        if system_message == "":
            system_message = "You must generate complete, valid JSON. Always finish all arrays and objects with proper closing brackets."
        else:
            system_message += " You must generate complete, valid JSON. Always finish all arrays and objects with proper closing brackets."
    
        if isinstance(prompts, str):
            prompts = [
                {"role": "user", "content": prompts},
            ]
        elif isinstance(prompts, list):
            if not isinstance(prompts[0], dict):
                prompts = [{"role": "user", "content": p} for p in prompts]
                
        prompts = [{"role": "system", "content": system_message}] + prompts
        sequences = self.pipeline(
            prompts,
            # do_sample=True,
            # top_k=top_p,
            # temperature=temperature,
            # num_return_sequences=1,
            # eos_token_id=self.tokenizer.eos_token_id,
            # max_length=max_gen_len,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            truncation=True,
        )
        results = []
        # print(f"prompts: {prompts}")
        for seq in sequences:
            results.append(seq["generated_text"])
            # print(f"Result: {seq['generated_text'][:200]}")
            # print("----------------------------------------")
        return results

    @classmethod
    def load_model_from_pretrained_pipeline(
        cls,
        model_name,
        use_quantization=False,
        quantization_bits=4
    ):
        print(f"------Loading model: {model_name}---------")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            try:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            except:
                pass

        # Update quantization configuration
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True if quantization_bits == 4 else False,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if use_quantization else None,
            device_map="auto",
            trust_remote_code=True,
        )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     load_in_4bit=True if use_quantization and quantization_bits == 4 else False, 
        #     # quantization_config=bnb_config if use_quantization else None,
        #     torch_dtype=torch.float16 if use_quantization else None,
        #     device_map="auto",
        #     trust_remote_code=True,
        # )
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # torch_dtype=torch.float16,
            device_map="auto",
        )
        return cls(pipeline=pipeline, tokenizer=tokenizer)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Company sector classification inference using Qwen3-32B model")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="name of the model",
        # default="Qwen/Qwen2.5-32B-Instruct",
        # default="Qwen/Qwen3-32B-Instruct",
        # Alternative Qwen models:
        # default="Qwen/Qwen2.5-32B",
        # default="Qwen/Qwen-72B-Chat",
        # Other models (for reference):
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        # default="mistralai/Mistral-7B-v0.1",
        # default="meta-llama/Llama-3.1-8B-Instruct",
        # default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        # default="gpt",
    )
    parser.add_argument("--use_quantization", type=str, help="use quantization", default='True')
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of job listings to process (default: all)",
        default=None
    )
    #parse arguments
    return parser.parse_args()

# Usage examples:
# CUDA_VISIBLE_DEVICES=0 python main_inference_qwen32b.py --use_quantization True
# CUDA_VISIBLE_DEVICES=0 python main_inference_qwen32b.py --model_name Qwen/Qwen2.5-32B-Instruct --use_quantization True
# CUDA_VISIBLE_DEVICES=0,1 python main_inference_qwen32b.py --use_quantization False  # For full precision across multiple GPUs
# CUDA_VISIBLE_DEVICES=0 python main_inference_qwen32b.py --use_quantization True --max_samples 10  # Process only first 10 job listings

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_arguments()
    model_name = args.model_name
    print(f"Using model: {model_name}")
    print(f"Quantization setting: {args.use_quantization}")
    use_quantization = True if str(args.use_quantization).lower() == 'true' else False

    # Load model
    if model_name != 'gpt':
        model = LLMInference.load_model_from_pretrained_pipeline(model_name, use_quantization=use_quantization)
   
    # Define system message and the prompt
    # system_message = """You are an expert career advisor specializing in job classification. Your task is to analyze job titles and descriptions and provide a concise, standardized occupational label that best represents the position as well as a company sector label that best represents the sector of the company."""
    # prompt = """Based on the following job title and description, provide a single, concise occupational label that best represents this position as well as a company sector label that best represents the sector of the company. Only provide the label, nothing else.



    # occupational labels = [
    # "Software developer",
    # "Data scientist",
    # "Data engineer",
    # "Cybersecurity specialist",
    # "Artificial intelligence specialist",
    # "Machine learning engineer",
    # "Cloud computing specialist",
    # "IT project manager",
    # "Systems analyst",
    # "Database administrator",
    # "Network administrator",
    # "Web developer",
    # "Mobile application developer",
    # "DevOps engineer",
    # "IT support technician",
    # "Game developer",
    # "Blockchain developer",
    # "UX/UI designer",
    # "IT business analyst",
    # "Embedded systems developer",
    # "Project manager",
    # "Nurse",
    # "Healthcare specialist",
    # "HR manager",
    # "Sales manager",
    # "Marketing manager",
    # "Financial analyst",
    # "Accountant",
    # "Audit manager",
    # "Risk manager",
    # "IT manager",
    # ]

#  company sector labels = [
#     "Agriculture, Forestry, Fishery",
#     "Arts, entertainment and recreation",
#     "Hospitality and Tourism",
#     "Human health and social services activities",
#     "ICT service activities",
#     "Manufacturing of food, beverages and tobacco",
#     "Manufacturing of Textile, Apparel, Leather, Footwear and related products",
#     "Mining and heavy industry",
#     "Transportation and storage",
#     "Veterinary activities",
#     "Wholesale and retail trade, renting and leasing",
#     "Business administration",
#     "Chemical industry",
#     "Construction",
#     "Education",
#     "Energy and water supply, sewerage and waste management",
#     "Finance, insurance and real estate",
#     "Manufacturing of consumer goods except food, beverages, tobacco, textile, apparel, leather",
#     "Manufacturing of electrical equipment, computer, electronic and optical products",
#     "Manufacturing of fabricated metal products, except machinery and equipment"
# ]

    system_message = """You are an expert career advisor specializing in company sector classification. Your task is to analyze job titles and descriptions and provide a concise, standardized company sector label. The label should be chosen from the list detailed below, choosing the label that represents the best match. 
    
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

  

    
    # system_message = """You are a teacher at high school. Answer student questions about the following topic: """
    # prompt = """What is the capital of France?"""
    
    

    my_prompts = load_data("../Data/filtered_jobs", "occupations_SE_GAP_analysis.csv")
    my_prompts['source_id'] = list(my_prompts['source_id'])
    
    results = {"source_id": [], "sectors": []}
    k_total=0

    # Determine number of samples to process
    max_samples = args.max_samples if args.max_samples else len(my_prompts['occupations'])
    num_samples = min(max_samples, len(my_prompts['occupations']))
    print(f"Processing {num_samples} out of {len(my_prompts['occupations'])} job listings")

    start_time = time.time()
    for i in range(num_samples):
        prompt = fixed_prompt + "Position title:" + my_prompts["occupations"][i] + "\n Description:" + my_prompts["descriptions"][i]

        # print("Prompt: ", prompt)
        if model_name != 'gpt':
            output = model.model_generate_pipeline(prompt, system_message=system_message)
            output = list(output)
            # print("Model output: ", output)
        else:
            output = get_response_gpt(prompt)


        try:
            llm_sec, k_total = process_output_llms(output, k=k_total)
            results["source_id"].append(my_prompts['source_id'][i])
            results["sectors"].append(llm_sec)
            
            # print("results: ", results)
        except ValueError as e:
            print(e)
            error = {'error': str(e), 'output': output}

        # if i>2:
        #     break
            


    end_time = time.time()
    total_time = end_time - start_time
    print(f'total_time: {total_time}')
    print(f'Number of non-matched llm string outputs = {k_total}')
    print("--------------------------------")
    # print("Results: ", results)
    df = pd.DataFrame(results)
    df.to_csv('../Data/classified_ads/filtered_jobs_sectors_ESCO_qwen32b.csv', 
          index=False,
          encoding='utf-8')

