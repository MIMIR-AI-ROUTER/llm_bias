# %%

import re
import os
import pandas as pd

# %%


def load_raw_dataset() -> pd.DataFrame:
    """
    Loads the preprocessed CSV.
    Returns a Pandas DataFrame with all resumes and metadata.
    """
    # Using a dummy path for the example, replace with your actual path
    dataset_path = "data/resume/selected_cats_resumes.csv"
    df = pd.read_csv(dataset_path)
    return df


# %%
# ----- START: The Corrected Logic for Finding Anonymization Errors -----

# List of gendered terms that should not be in an anonymized resume
GENDER_INDICATORS = [
    # Pronouns and Titles
    r"\b(he|his|him|she|her|hers|Mr\.|Mrs\.|Ms\.|Miss)\b",
    # Organizations and Roles
    r"Fraternity",
    r"Sorority",
    r"Brother",
    r"Sister",
    r"Boy Scouts",
    r"Girl Scouts",
    r"Women in Technology",
    r"Arizona Business and Professional Women",
    # Gendered Nouns
    r"\b(actor|actress|mistress of ceremonies|master of ceremonies)\b",
    r"\b(Male Athlete|Female Athlete)\b",
    r"\b(guy|gal|man|woman)\b",
]

# List of racial and national origin terms that should not be in an anonymized resume
# This combines terms from BOTH the "White" and "Black" sections of the original script.
RACE_AND_ORIGIN_INDICATORS = [
    # --- Strong Indicators: HBCUs ---
    r"North Carolina A&T State University",
    r"Morgan State University",
    r"Prairie View A&M University",
    r"Florida A&M University",
    r"Hampton University",
    # --- Strong Indicators: Historically Black Organizations ---
    r"National Forum for Black Public Administrators",
    r"United Negro College Fund",
    r"Delta Sigma Theta",  # Historically Black sorority (also a gender marker)
    r"Cook County Bar Association",  # Nation's oldest Black bar association
    # --- Strong Indicators: National Origin ---
    r"Nigeria",
    r"Cameroon",
    r"Ukraine",
    r"Federal Polytechnic, Ado-Ekiti",
    r"University of Buea",
    r"University of Lagos",
    r"COREN",  # Council for Regulation of Engineering in Nigeria
]


def find_demographic_indicators(resume_text: str) -> dict:
    """
    Scans a resume for predefined gender, race, and national origin indicators.

    Args:
        resume_text: The string content of the resume.

    Returns:
        A dictionary containing the lists of found indicator patterns.
    """
    found_indicators = {"gender": [], "race_and_origin": []}

    all_indicators = {
        "gender": GENDER_INDICATORS,
        "race_and_origin": RACE_AND_ORIGIN_INDICATORS,
    }

    for category, patterns in all_indicators.items():
        for pattern in patterns:
            # re.IGNORECASE is crucial for matching
            if re.search(pattern, resume_text, re.IGNORECASE):
                found_indicators[category].append(pattern)

    return found_indicators


# ----- END: The Corrected Logic for Finding Anonymization Errors -----
# %%

# --- Main script execution ---

# 1. Load and filter the dataset
df = load_raw_dataset()
print(f"Loaded {len(df)} total resumes.")
df = df[df["Category"] == "INFORMATION-TECHNOLOGY"].reset_index(drop=True)
print(f"Filtered down to {len(df)} resumes in the 'INFORMATION-TECHNOLOGY' category.")
print("-" * 30)

# 2. Initialize counters
improperly_anonymized_count = 0
processed_unique_resumes = set()
resumes_with_errors = []

# 3. Iterate through the DataFrame and check each unique resume
for index, row in df.iterrows():
    resume_str = row["Resume_str"]

    # Skip this resume if we've already processed its exact text
    if resume_str in processed_unique_resumes:
        continue

    # Add the resume to the set of processed resumes
    processed_unique_resumes.add(resume_str)

    # Find demographic indicators
    indicators_found = find_demographic_indicators(resume_str)

    # If any indicator was found, it's improperly anonymized
    if indicators_found["gender"] or indicators_found["race_and_origin"]:
        improperly_anonymized_count += 1
        resumes_with_errors.append(
            {
                "index": index,
                "indicators": indicators_found,
                "resume_preview": resume_str[:200]
                + "...",  # Store a preview for review
            }
        )

# 4. Print the final results
total_unique_resumes = len(processed_unique_resumes)

print("\n--- Analysis Complete ---")
if total_unique_resumes > 0:
    percentage_failed = (improperly_anonymized_count / total_unique_resumes) * 100
    print(f"Total Unique Resumes Analyzed: {total_unique_resumes}")
    print(f"Resumes with Anonymization Errors: {improperly_anonymized_count}")
    print(f"Failure Rate: {percentage_failed:.2f}%")
else:
    print("No unique resumes found to analyze in the specified category.")

# 5. (Optional) Print details of the first few resumes with errors for review
if resumes_with_errors:
    print("\n--- Examples of Resumes with Errors ---")
    for i, error_info in enumerate(resumes_with_errors[:5]):  # Print first 5 examples
        print(f"\nExample {i + 1} (Original Index: {error_info['index']}):")
        if error_info["indicators"]["gender"]:
            print(f"  Found Gender Indicators: {error_info['indicators']['gender']}")
        if error_info["indicators"]["race_and_origin"]:
            print(
                f"  Found Race/Origin Indicators: {error_info['indicators']['race_and_origin']}"
            )
        # print(f"  Resume Preview: {error_info['resume_preview']}")
# %%
