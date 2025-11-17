import os
import pandas as pd
import google.generativeai as genai
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from dotenv import load_dotenv
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import threading
from functools import lru_cache
from typing import List, Dict, Tuple
from tqdm import tqdm

# --- GLOBAL CONFIGURATION & SETUP ---
print("🚀 Starting the UNIFIED Candidate Evaluation Pipeline...")
PIPELINE_START_TIME = time.time()
load_dotenv()
# Configure Google Generative AI
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    # Trim whitespace and remove quotes if present
    if GEMINI_API_KEY:
        GEMINI_API_KEY = GEMINI_API_KEY.strip()
        # Remove surrounding quotes if present
        if (GEMINI_API_KEY.startswith('"') and GEMINI_API_KEY.endswith('"')) or \
           (GEMINI_API_KEY.startswith("'") and GEMINI_API_KEY.endswith("'")):
            GEMINI_API_KEY = GEMINI_API_KEY[1:-1].strip()
    
    # Check if key is still a placeholder
    if GEMINI_API_KEY and GEMINI_API_KEY.lower() in ['your_api_key_here', 'your_actual_api_key_here', '']:
        print("⚠️  WARNING: API key appears to be a placeholder.")
        print("   Please replace 'your_api_key_here' in the .env file with your actual Gemini API key.")
        print("   Get your API key from: https://aistudio.google.com/app/apikey")
        GEMINI_API_KEY = None
    
    if not GEMINI_API_KEY:
        env_file = '.env'
        if not os.path.exists(env_file) or os.path.getsize(env_file) == 0:
            # Create a template .env file if it doesn't exist or is empty
            with open(env_file, 'w') as f:
                f.write("# Google Generative AI API Key\n")
                f.write("GEMINI_API_KEY=your_api_key_here\n")
            print(f"📝 Created template {env_file} file. Please add your GEMINI_API_KEY.")
        else:
            print(f"⚠️  {env_file} file exists but GEMINI_API_KEY is not set.")
        print("\n❌ FATAL ERROR: GEMINI_API_KEY not found in .env file.")
        print("   Please add your API key to the .env file in the format:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        exit(1)
    configure(api_key=GEMINI_API_KEY)
    
    # Validate API key by making a test API call
    print("🔍 Validating API key...")
    # Show first and last 4 characters for debugging (without exposing full key)
    if len(GEMINI_API_KEY) > 8:
        key_preview = f"{GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}"
        print(f"   API key preview: {key_preview} (length: {len(GEMINI_API_KEY)})")
    try:
        test_model = GenerativeModel("gemini-2.5-flash")
        test_response = test_model.generate_content("Say 'OK' if you can read this.")
        if not test_response or not hasattr(test_response, 'text'):
            raise ValueError("API key validation failed: Invalid response from API")
        print("✅ Google Generative AI configured and validated successfully.")
    except Exception as validation_error:
        error_msg = str(validation_error).lower()
        if "api key" in error_msg or "invalid" in error_msg or "401" in error_msg or "403" in error_msg or "permission" in error_msg:
            print(f"❌ FATAL ERROR: Invalid or unauthorized API key.")
            print(f"   Error details: {validation_error}")
            print("   Please check your GEMINI_API_KEY in the .env file.")
            print("   Make sure the key is correct and has the necessary permissions.")
        else:
            print(f"❌ FATAL ERROR: Could not validate API key. Reason: {validation_error}")
        exit(1)
except ValueError as e:
    print(f"❌ FATAL ERROR: {e}")
    exit(1)
except Exception as e:
    print(f"❌ FATAL ERROR: Could not configure Google Generative AI. Reason: {e}")
    exit(1)

# --- COLUMN NAME CONSTANTS ---
# Define all column names here for easy management
COL_JOB_DESC = "Grapevine Job - Job → Description"
COL_INTERVIEW = "Grapevine Aiinterviewinstance → Transcript → Conversation"
COL_DURATION = "Grapevine Aiinterviewinstance → Transcript → Duration"
COL_RESUME = "Grapevine Userresume - Resume → Metadata → Resume Text"
COL_CRITERIA = "Recruiter GPT Response "
COL_CANDIDATE_NAME = "Grapevine Userresume - Resume → Metadata → User Real Name"
COL_COMPANY_NAME = "Title"
COL_RESUME_URL = "Grapevine Userresume - Resume → Resume URL"
COL_PHONE = "Grapevine Userresume - Resume → Metadata → Phone Number"

# Additional input columns for candidate profile
COL_EMAIL = "Grapevine Userresume - Resume → Metadata → Email"
COL_CTC = "Grapevine Userresume - Resume → Metadata → Current Salary"
COL_EXPERIENCE = "Grapevine Userresume - Resume → Metadata → Experience"

# --- ULTRA-FAST GEMINI PROCESSOR CLASS ---
_response_cache = {}
_cache_lock = threading.Lock()

class UltraFastGeminiProcessor:
    """A highly concurrent, batch-processing class for the Gemini API."""
    def __init__(self, max_workers=32, cache_enabled=True):
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self._response_cache = _response_cache
        self._cache_lock = _cache_lock

    def _gemini_generate_single(self, model_name, prompt):
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response)
        except Exception as e:
            error_str = str(e).lower()
            # Check for API key related errors
            if "api key" in error_str or "invalid" in error_str or "401" in error_str or "403" in error_str or "permission" in error_str or "unauthorized" in error_str:
                return f"Error: Invalid API key - {e}"
            return f"Error: {e}"

    def process_prompts_in_parallel(self, model_name: str, prompts: list, task_description: str) -> list:
        if not prompts:
            return []
        print(f"🔥 Starting parallel processing for '{task_description}' ({len(prompts)} prompts)...")
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(self._gemini_generate_single, model_name, prompt): i for i, prompt in enumerate(prompts)}
            with tqdm(as_completed(future_to_index), total=len(prompts), desc=f"⚡ {task_description}") as progress_bar:
                for future in progress_bar:
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        results[index] = f"Error: Task failed with exception: {e}"
        
        # Check for API key errors in results
        api_key_errors = [r for r in results if "Invalid API key" in r or ("Error:" in r and ("api key" in r.lower() or "401" in r or "403" in r))]
        if api_key_errors:
            print(f"⚠️  WARNING: Found {len(api_key_errors)} API key errors in results!")
            print("   This suggests your API key may be invalid or expired.")
            print("   Please check your GEMINI_API_KEY in the .env file.")
        
        print(f"✅ Completed '{task_description}'.")
        return results

# Instantiate processor globally
processor = UltraFastGeminiProcessor(max_workers=32, cache_enabled=True)

# Output columns for pipeline stages
COL_RESUME_EVAL = "Resume Evaluator Agent (RAG-LLM)"
COL_INTERVIEW_EVAL = "Interview Evaluator Agent (RAG-LLM)"
COL_SUMMARIZER = "Resume + Interview Summarizer Agent"
COL_RESULT = "Result[LLM]"
COL_GOOD_FIT = "Good Fit"
COL_PROFILE = "Candidate Profile"

def clean_phone_number(phone_val):
    """
    Cleans a phone number by removing non-digit characters and keeping the last 10 digits.
    Handles various input types like strings, floats (including scientific notation), and NaNs.
    """
    if pd.isna(phone_val):
        return ''
    # Convert to string, ensuring scientific notation on floats is handled properly
    s = '{:.0f}'.format(phone_val) if isinstance(phone_val, (float, int)) else str(phone_val)
    # Remove non-digit characters
    digits_only = re.sub(r'\D', '', s)
    # Return the last 10 digits, which is the standard mobile number length in many places.
    if len(digits_only) > 10:
        return digits_only[-10:]
    return digits_only

@lru_cache(maxsize=1000)
def create_verdict_prompt(text):
    # This prompt is from your first script
    return f"""From the text below, extract only one of these exact words: "Advanced", "Reject", or "Manual Intervention".You are an extraction assistant. Only output one of the following exact words based on the decision in the provided text: "Advanced", "Reject", or "Manual Intervention". Do not output anything else. Do not add explanations, formatting, or extra text. Do not output the word "Advance".



Text: {text}
"""


# --- STAGE 2: DETAILED PROFILING PROMPTS ---

def create_good_fit_prompt(row_data: dict) -> str:
    """Creates the prompt for the 'Good Fit' summary, adapted from your second script."""
    candidate_name = row_data.get(COL_CANDIDATE_NAME, 'The Candidate')
    company_name = row_data.get(COL_COMPANY_NAME, 'the Company')
    job_description = row_data.get(COL_JOB_DESC, '')
    resume_text = row_data.get(COL_RESUME, '')

    # Handle potential NaN values
    if pd.isna(job_description) or pd.isna(resume_text):
        return "Error: Cannot generate summary because Job Description or Resume Text is missing."

    # Convert to string if not already
    job_description = str(job_description) if job_description else ''
    resume_text = str(resume_text) if resume_text else ''

    if not job_description or not resume_text:
        return "Error: Cannot generate summary because Job Description or Resume Text is missing."

    # A simple way to get job title, can be improved
    job_title = "the Role"
    if 'title' in job_description.lower():
        m = re.search(r'title\s*[:\-]\s*([^\n\r]+)', job_description, re.IGNORECASE)
        if m and m.group(1):
            job_title = m.group(1).strip()

    first_name = str(candidate_name).split(' ')[0]

    return f"""
You are a professional recruitment analyst. Analyze the provided information and generate a concise, 4-point bulleted summary explaining why the candidate is a strong fit for the role.

*Context:*
- Candidate: {candidate_name}
- Company: {company_name}
- Role: {job_title}
- Job Description: {job_description}
- Resume: {resume_text}

*Task:*
Generate a 4-point summary. Each point must be a clear, role-specific reason supported by evidence from the resume.
3. *[Core Strength 3]*: [Brief explanation linking candidate's experience to a job requirement].
4. *[Core Strength 4]*: [Brief explanation linking candidate's experience to a job requirement].
"""

def create_candidate_profile_prompt(row_data: dict, good_fit_summary: str) -> str:
    """Creates the prompt for the detailed 'Candidate Profile'."""
    if "Error:" in good_fit_summary:
        return good_fit_summary

    candidate_name = row_data.get(COL_CANDIDATE_NAME, "Not Provided")
    company_name = row_data.get(COL_COMPANY_NAME, "Not Provided")
    job_description = row_data.get(COL_JOB_DESC, "")
    resume_text = row_data.get(COL_RESUME, "")
    resume_url = row_data.get(COL_RESUME_URL, "")
    phone = row_data.get(COL_PHONE, "Not Provided")
    email = row_data.get(COL_EMAIL, "Not Provided")
    ctc = row_data.get(COL_CTC, "Not Provided")
    experience = row_data.get(COL_EXPERIENCE, "Not Provided")

    # Extract LinkedIn URL from resume text if present
    linkedin_url = "Not Present"
    if isinstance(resume_text, str) and pd.notna(resume_text):
        match = re.search(r'(https?://(?:www\.)?linkedin\.com/in/[\w\-/]+)', resume_text)
        if match:
            linkedin_url = match.group(0)

    resume_hyperlink = f"[Resume]({resume_url})" if resume_url else "[Resume Not Provided]"
    linkedin_hyperlink = f"[LinkedIn]({linkedin_url})" if linkedin_url != "Not Present" else "[LinkedIn Not Present]"

    first_name = str(candidate_name).split(" ")[0] if candidate_name and candidate_name != "Not Provided" else "Candidate"

    # Try to infer a job title from the job description (fallback to 'the Role')
    job_title = "the Role"
    if isinstance(job_description, str) and job_description:
        m = re.search(r'(?:title\s*[:\-]\s*)([^\n\r]+)', job_description, re.IGNORECASE)
        if m and m.group(1):
            job_title = m.group(1).strip()

    prompt = f"""You are an expert Talent Agency Analyst. Produce a clean, Notion-friendly candidate profile using only the provided information. If a field is missing, write 'Not Provided'. Do not invent numbers, institutions, company names, or degrees.

Context:
- Candidate: {candidate_name}
- Company: {company_name}
- Role: {job_title}
- Job Description: {job_description}
- Resume excerpt: {resume_text}

Good Fit Summary:
{good_fit_summary}
   Experience: {experience}

2) Why {first_name} for {job_title} at {company_name}:
   1. [Clear, role-specific reason supported by evidence from resume or provided data]
   2. [Clear, role-specific reason]
   3. [Clear, role-specific reason]
   4. [Clear, role-specific reason]

3) Technical Alignment:
   - [Short bullets of relevant technical and soft skills]

4) What Stood Out in the Interview:
   - [Brief narrative or bullets; if transcript not available, base this on resume and JD]

5) Links:
   Resume: {resume_hyperlink}
   LinkedIn: {linkedin_hyperlink}

6) One-sentence personalized summary of candidate fit.

Formatting rules:
- Output plain text only (no markdown code fences).
- Use clear line breaks and bullets as above.
- Do NOT hallucinate or invent any numbers, colleges, companies, or names. If information is missing, state 'Not Provided'.
"""

    return prompt


def create_resume_prompt(job_desc, resume, criteria):
    # This prompt is from your first script
    return f"""Analyze the provided resume against the job role and criteria. Output a JSON object evaluating 'Education and Company Pedigree', 'Skills & Specialties', 'Work Experience', 'Basic Contact Details', and 'Educational Background Details' on specified scales, including a justification for each. Also, provide a summary for 'Input on Job Specific Criteria'. You are an AI hiring manager. Follow the instructions exactly. Only use the information provided in the input. Never invent or assume details. If information is missing, state this in your justification. Always output in the specified JSON format.

- Job Role Description: {job_desc}
- Resume: {resume}
- Job-Specific Criteria: 
Bias for 

- Venture Funded Startup 
- ⁠Top Undergrad Program Business / Engineering / Liberal Arts 
- ⁠Bias for Metrics in resume 
{criteria}

Output your assessment as a JSON object only.
"""


def create_summarizer_prompt(job_desc, resume_eval, interview_eval, criteria):
    """Placeholder for summarizer prompt. Implement as needed."""
    return f"Summarize resume and interview for job: {job_desc}\nResume Eval: {resume_eval}\nInterview Eval: {interview_eval}\nCriteria: {criteria}"

def get_gemini_verdict(jd_text, resume_text, candidate_profile):
    """
    Calls the Gemini API to get a final verdict and justification for a candidate.
    Returns a tuple: (verdict, justification)
    """
    prompt = f"""
You are an expert HR reviewer. You are given a candidate profile, Requiter requirements (RR) and a job description (JD). This candidate was previously marked as "Maybe" by an automated system. Your task is to thoroughly review the candidate’s profile and the JD, and then make a clear, final decision: either Advanced (the candidate is a strong fit and should move forward) or No (the candidate is not a fit and should not move forward).

Instructions:

Review the Job Description (JD): Carefully read the requirements, responsibilities, and qualifications.
Review the Recquirter Requirements  (JD): Carefully read the requirements, must have and should have
Review the Candidate Profile: Examine the candidate’s experience, skills, education, and any other relevant information.
Compare and Analyze: Assess how well the candidate matches the must-have and nice-to-have requirements in the JD.
IMPORTANT: DO NOT RELY ON YEARS OF EXPERIENCE THAT A CANDIDATE MAY HAVE FOR YOUR DECISIONS. CONSIDER THE IMPACT THEY CREATED AT THEIR JOB, AND THE COMPANY THEY WORKED AT.
AS A RULE OF THUMB, SOMEONE WHO HAS WORKED AT A STARTUP[indian first, search web to make a decision if its a startup] is considered to be a good candidate.
prioritise people from these companies : (\"Dream11\" OR \"MPL\" OR \"Mobile Premier League\" OR \"WinZO\" OR \"A23\" OR \"Nazara Technologies\" OR \"Gameskraft\" OR \"RummyCircle\" OR \"Junglee Rummy\" OR \"Zupee\" OR \"PokerBaazi\" OR \"Adda52\" OR \"FanFight\" OR \"BalleBaazi\" OR \"My11Circle\" OR \"HalaPlay\" OR \"Ace2Three\" OR \"Gamezy\" OR \"SkillClash\" OR \"Classic Rummy\" OR \"Pocket52\" OR \"CrickPe\" OR \"One World Nation\" OR \"PokerStars India\" OR \"Fairplay Club\" OR \"Rooter\" OR \"PlayerzPot\" OR \"Fantasy Akhada\" OR \"LeagueX\" OR \"SportsTiger\") 
AND (\"Real Money Gaming\" OR \"RMG\" OR \"Online Gaming\" OR \"iGaming\" OR \"Skill Gaming\" OR \"Fantasy Sports\" OR \"Poker\" OR \"Rummy\" OR \"Cash Tournaments\" OR \"Monetized Gaming\")
AND (\"Startup\" OR \"Product-based company\" OR \"Series A\" OR \"Series B\" OR \"VC funded\" OR \"Growth stage\")
Be Decisive: Avoid ambiguity. Do not select "Maybe." You must choose either Advanced or No.
Justify Your Decision: Provide a brief, clear explanation (2-3 sentences) for your decision, referencing specific requirements or gaps.
Output Format:

Final Decision: [Advanced/No]
Justification: [Your explanation]
Example Output:

Final Decision: Advanced
Justification: The candidate has 5+ years of relevant experience, meets all must-have skills, and has led similar projects as described in the JD.

Job Description:
{jd_text}

Candidate Profile:
{candidate_profile}

Resume Text:
{resume_text}
"""
    try:
        model = GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        verdict = "Parsing Error"
        justification = "Could not parse justification."

        if lines:
            first_line = lines[0].lower()
            if "advanced" in first_line:
                verdict = "Advanced"
            elif "no" in first_line:
                verdict = "No"

        justification_line = next((line for line in lines if line.lower().startswith("justification:")), None)
        if justification_line:
            justification = justification_line.split(":", 1)[-1].strip()

        return verdict, justification
    except Exception as e:
        return "API Error", str(e)
@lru_cache(maxsize=1000)
def create_interview_prompt(job_desc, transcript, criteria, resume_text, duration=None):
    wpm = "Not Available"
    fillers = "Not Available"
    pauses = "Not Available"
    duration_str = str(duration) if duration is not None else "Not Available"
    return f"""You are acting as a senior hiring manager. 
You are evaluating the audio interview where the person speaks for {duration_str} seconds about their work experience. 
They already have a strong resume. Your job is to decide: can they back it up when speaking live?

Important context:
- Most people you evaluate are Indian professionals.
- Do NOT evaluate grammar, accent, or native-speaker fluency.
- Focus only on workplace communication and articulation: can they explain their work clearly, show ownership, and articulate outcomes?
- Pay special attention to whether their answers genuinely back up the resume or expose surface-level understanding / inflation.

Decision rules for verdict_label:
- "Strong Communicator" → Clearly and credibly backs up the resume with specifics, outcomes, and ownership.
- "Adequate" → Understandable but missing depth or outcomes in places. Not a red flag, but not impressive.
- "Needs Work" → Does not back up the resume; answers are vague, surface-level, or inflated. Red flag.

Candidate resume:
{resume_text}

Transcript of answers:
{transcript}

Job-Specific Criteria:
{criteria}

Observed audio signals:
- Duration (seconds): {duration_str}
- Words per minute: {wpm}
- Fillers per minute: {fillers}
- Pauses per minute: {pauses}

Write your evaluation as a hiring manager’s feedback note. 
Be honest and realistic, the way you would debrief a hiring panel after an interview. 
Do not call the person "the candidate" — instead, address them by name and mention companies or projects from the resume when relevant. 
If they back up the resume strongly, say so clearly. If there are gaps or red flags, call them out directly.


Output valid JSON only with this schema:
{{
  "verdict_label": "Strong Communicator" | "Adequate" | "Needs Work",
  "evaluation_note": "string, a concise hiring-manager style paragraph written about the person, addressing them by name and mentioning companies/projects where relevant",
  "strengths": ["list of concrete strengths where the resume was backed up"],
  "red_flags": ["list of gaps, weak answers, or inflation risks revealed in the interview"],
  "examples": ["short transcript excerpt 1", "short transcript excerpt 2"]
}}
"""

def run_initial_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Stage 1: Resume, Interview, Summary, and Verdict evaluations."""
    print("\n" + "="*25 + " STAGE 1: INITIAL EVALUATION " + "="*25)

    # --- Create Prompts ---
    resume_prompts = [create_resume_prompt(r.get(COL_JOB_DESC, ""), r.get(COL_RESUME, ""), r.get(COL_CRITERIA, "")) for _, r in df.iterrows()]
    interview_prompts = [
        create_interview_prompt(
            r.get(COL_JOB_DESC, ""),
            r.get(COL_INTERVIEW, ""),
            r.get(COL_CRITERIA, ""),
            r.get(COL_RESUME, ""),
            r.get(COL_DURATION, None)
        ) if pd.notna(r.get(COL_INTERVIEW)) and r.get(COL_INTERVIEW, "").strip() else ""
        for _, r in df.iterrows()
    ]

    # --- Run Evaluations in Parallel ---
    df[COL_RESUME_EVAL] = processor.process_prompts_in_parallel("gemini-2.5-pro", resume_prompts, "Resume Evaluations")

    valid_interview_prompts = [p for p in interview_prompts if p]
    interview_results_partial = processor.process_prompts_in_parallel("gemini-2.5-pro", valid_interview_prompts, "Interview Evaluations")

    # Map interview results back
    full_interview_results = [""] * len(df)
    interview_result_idx = 0
    for i in range(len(df)):
        if interview_prompts[i]:
            full_interview_results[i] = interview_results_partial[interview_result_idx]
            interview_result_idx += 1
        else:
            full_interview_results[i] = "Interview not conducted or transcript unavailable."
    df[COL_INTERVIEW_EVAL] = full_interview_results

    # --- Create and Run Summarizer and Verdict ---
    summarizer_prompts = [create_summarizer_prompt(row.get(COL_JOB_DESC, ""), row.get(COL_RESUME_EVAL, ""), row.get(COL_INTERVIEW_EVAL, ""), row.get(COL_CRITERIA, "")) for _, row in df.iterrows()]
    df[COL_SUMMARIZER] = processor.process_prompts_in_parallel("gemini-2.5-pro", summarizer_prompts, "Summaries")

    verdict_prompts = [create_verdict_prompt(summary) for summary in df[COL_SUMMARIZER]]
    df[COL_RESULT] = processor.process_prompts_in_parallel("gemini-2.5-flash", verdict_prompts, "Verdicts")

    print("✅ STAGE 1: Initial Evaluation Complete.")
    return df
def run_detailed_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Stage 2: 'Good Fit' and 'Candidate Profile' generation."""
    print("\n" + "="*25 + " STAGE 2: DETAILED PROFILING " + "="*25)

    if df.empty:
        print("ℹ No candidates to process for detailed profiling. Skipping Stage 2.")
        return pd.DataFrame(columns=[COL_GOOD_FIT, COL_PROFILE])

    print(f"Found {len(df)} candidates for detailed profiling.")

    # --- Create and Run "Good Fit" ---
    row_data_list = [row.to_dict() for _, row in df.iterrows()]
    good_fit_prompts = [create_good_fit_prompt(data) for data in row_data_list]
    good_fit_results = processor.process_prompts_in_parallel("gemini-2.5-pro", good_fit_prompts, "Good Fit Summaries")

    # --- Create and Run "Candidate Profile" ---
    profile_prompts = [create_candidate_profile_prompt(row_data_list[i], good_fit_results[i]) for i in range(len(df))]
    profile_results = processor.process_prompts_in_parallel("gemini-2.5-pro", profile_prompts, "Candidate Profiles")

    # Clean up markdown code blocks and special characters from results
    cleaned_profiles = []
    for p in profile_results:
        # Remove markdown code blocks
        cleaned = re.sub(r"markdown\n?|", "", p)
        # Remove flower brackets and other special characters
        cleaned = re.sub(r'[{}]', '', cleaned)  # Remove curly braces
        cleaned = re.sub(r'[^\w\s.,;:!?()-\[\]#*]', '', cleaned)  # Remove other special characters except basic punctuation and markdown
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned_profiles.append(cleaned)

    # Create a results DataFrame to merge back
    results_df = pd.DataFrame({
        COL_GOOD_FIT: good_fit_results,
        COL_PROFILE: cleaned_profiles
    }, index=df.index)

    print("✅ STAGE 2: Detailed Profiling Complete.")
    return results_df
# --- MAIN ORCHESTRATOR ---

def main():
    # --- Load Data ---
    try:
        # Prioritize CSV, then look for Excel
        csv_filename = "test - test.csv"
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
            print(f"📄 Successfully loaded '{csv_filename}' with {len(df)} rows.")
        else:
            excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
            if not excel_files:
                raise FileNotFoundError("No CSV or .xlsx file found.")
            df = pd.read_excel(excel_files[0])
            print(f"📄 Successfully loaded '{excel_files[0]}' with {len(df)} rows.")
    except Exception as e:
        print(f"❌ FATAL ERROR loading data: {e}")
        return

    # --- Title Case for Candidate Name ---
    if COL_CANDIDATE_NAME in df.columns:
        df[COL_CANDIDATE_NAME] = df[COL_CANDIDATE_NAME].astype(str).apply(lambda x: x.title() if pd.notna(x) else x)
    # --- Title Case Resume Text ---
    if COL_RESUME in df.columns:
        df[COL_RESUME] = df[COL_RESUME].astype(str).apply(lambda x: x.title() if pd.notna(x) else x)
    # --- Clean Phone Number ---
    if COL_PHONE in df.columns:
        df[COL_PHONE] = df[COL_PHONE].apply(clean_phone_number)
    # --- Rename Phone Column if Present ---
    if COL_PHONE in df.columns:
        df.rename(columns={COL_PHONE: "phone"}, inplace=True)
    # --- Deduplicate by Resume Link (immediately after loading data) ---
    initial_rows = len(df)
    df.drop_duplicates(subset=[COL_RESUME_URL], keep='first', inplace=True)
    if (removed_count := initial_rows - len(df)) > 0:
        print(f"🔍 Deduplication removed {removed_count} duplicate entries based on resume link.")

    # --- Run Stage 1 ---
    df = run_initial_evaluations(df)

    # --- Filter Rejected Candidates ---
    print("\n" + "="*25 + " FILTERING REJECTED CANDIDATES " + "="*25)
    initial_count = len(df)
    # Normalize verdict text for reliable filtering
    df[COL_RESULT] = df[COL_RESULT].str.strip().str.title()

    non_rejected_df = df[df[COL_RESULT] != 'Reject'].copy()

    print(f"Initial candidates: {initial_count}")
    print(f"Candidates marked 'Rejected': {initial_count - len(non_rejected_df)}")
    print(f"Candidates remaining for profiling ('Advanced' or 'Manual Intervention'): {len(non_rejected_df)}")

    # --- Run Stage 2 ---
    if not non_rejected_df.empty:
        detailed_profiles_df = run_detailed_profiling(non_rejected_df)

        # --- Merge Results ---
        print("\n" + "="*25 + " MERGING RESULTS " + "="*25)
        # Initialize columns if they don't exist
        if COL_GOOD_FIT not in df.columns:
            df[COL_GOOD_FIT] = ""
            df[COL_PROFILE] = ""

        # Merge in the detailed profile results
        df.update(detailed_profiles_df)

        print("✅ Detailed profiling results merged into main dataset.")
    else:
        print("ℹ No candidates to profile, skipping merge step.")
# --- Save results ---
    output_filename = "tf front posteval.csv"
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n🎉 Pipeline Complete! All results saved to '{output_filename}'.")
    except Exception as e:
        print(f"❌ ERROR saving results: {e}")

    total_time = time.time() - PIPELINE_START_TIME
    print(f"⏱ Total pipeline execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

    # --- Manual Intervention Stage ---
    manual_mask = df[COL_RESULT] == "Manual Intervention"
    if manual_mask.any():
        print("🔎 Processing Manual Intervention candidates...")
        for idx, row in df[manual_mask].iterrows():
            verdict, justification = get_gemini_verdict(
                row.get(COL_JOB_DESC, ""),
                row.get(COL_RESUME, ""),
                row.get(COL_PROFILE, "")
            )
            df.at[idx, "Manual Intervention verdict?"] = verdict
            df.at[idx, "Justification"] = justification
        print("✅ Manual Intervention verdicts and justifications added.")
    else:
        # Ensure columns exist even if no manual cases
        df["Manual Intervention verdict?"] = ""
        df["Justification"] = ""
    return df

if __name__ == "__main__":
    main()
