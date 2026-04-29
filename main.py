import os
import urllib.request
from src.ocr_pipeline import run_ocr_pipeline
from src.nlp_processing import process_student_answer

def setup_sample_image():
    """
    Downloads a sample handwritten text image if one doesn't exist.
    """
    sample_dir = "data/sample_images"
    sample_path = os.path.join(sample_dir, "sample1.png")
    
    if not os.path.exists(sample_path):
        print("[*] Downloading sample handwritten image...")
        # Using a publicly available sample handwritten image
        url = "https://raw.githubusercontent.com/tesseract-ocr/test/master/testing/phototest.tif" 
        try:
            urllib.request.urlretrieve(url, sample_path)
            print(f"[*] Saved sample image to {sample_path}")
        except Exception as e:
            print(f"[!] Error downloading sample: {e}")
            print("[!] Please place a handwritten image named 'sample1.png' in data/sample_images/")
            
    return sample_path

def main():
    print("=== Automated Student Assignment Assessment ===")
    print("Phase 1 & 2: OCR Extraction and NLP Cleaning\n")
    
    # Get a sample image
    image_path = setup_sample_image()
    
    if not os.path.exists(image_path):
         return
         
    # --- PHASE 1: OCR ---
    print("\n--- PHASE 1: OCR ---")
    extracted_text, ocr_conf = run_ocr_pipeline(image_path)
    
    print("\n=== Raw Extracted Text ===")
    print("-" * 30)
    print(f"Confidence: {ocr_conf*100:.1f}%")
    print(extracted_text)
    print("-" * 30)
    
    # --- PHASE 2: NLP Clean ---
    print("\n--- PHASE 2: TEXT CLEANING & NLP ---")
    nlp_results = process_student_answer(extracted_text)
    
    print("\n=== Cleaned Output ===")
    print(f"1. Regex Cleaned: {nlp_results['regex_cleaned_text']}")
    print(f"2. Final Tokens: {nlp_results['tokens']}")
    print(f"3. Final Processed String: {nlp_results['final_processed_string']}")
    print("-" * 30)
    
    print("\n[+] Phase 2 Complete. Text is ready for Semantic Similarity scoring (Phase 3).")
    
    # --- PHASE 3: SEMANTIC SIMILARITY ---
    print("\n--- PHASE 3: SEMANTIC SIMILARITY ---")
    import json
    from src.similarity import compute_similarity
    
    # Load model answers
    with open("data/model_answers.json", "r") as f:
        model_answers = json.load(f)
    
    # In a real app, you would look up the correct model answer based on the assignment ID.
    model_answer = model_answers["sample1"]
    
    student_answer = nlp_results['final_processed_string']
    
    print("\n=== Similarity Scoring ===")
    print(f"Model Answer: {model_answer}")
    print(f"Student Answer (Cleaned): {student_answer}")
    
    # Compute similarity score 
    score = compute_similarity(student_answer, model_answer)
    
    print("-" * 30)
    print(f"Semantic Similarity Score: {score:.4f} (out of 1.0)")
    print("-" * 30)
    
    # --- PHASE 4: SCORING ---
    print("\n--- PHASE 4: SCORING SYSTEM ---")
    from src.scoring import calculate_grade
    from src.feedback import generate_feedback
    
    # Assuming the question is worth 10 marks
    max_marks_for_question = 10
    grading_results = calculate_grade(score, max_marks=max_marks_for_question)
    
    print(f"\nEvaluating Assignment (Max {grading_results['max_marks']} marks):")
    print(f"-> Grade: {grading_results['grade_category']}")
    print(f"-> Final Score: {grading_results['awarded_marks']} / {grading_results['max_marks']}")
    print("\n[+] Phase 4 Complete. Scoring engine finished.")
    
    # --- PHASE 6: AGENTIC VERIFICATION ---
    print("\n--- PHASE 6: AGENT VERIFICATION ---")
    from src.agent_verifier import rescue_grade_if_needed
    
    # Let the Agent verify if the grading is Average/Poor
    grading_results = rescue_grade_if_needed(
        grading_results, 
        student_raw_text=nlp_results['regex_cleaned_text'], 
        model_answer=model_answer
    )
    
    print("\n[+] Phase 6 Complete. Agent verification ended.")
    
    # --- PHASE 5: FEEDBACK GENERATION ---
    print("\n--- PHASE 5: FEEDBACK GENERATOR ---")
    feedback_string = generate_feedback(
        student_raw_text=extracted_text, 
        model_raw_text=model_answer, 
        grade_category=grading_results['grade_category']
    )
    
    print("\n=== Final Assessment Report ===")
    print(f"Score: {grading_results['awarded_marks']}/{grading_results['max_marks']} ({grading_results['grade_category']})")
    if 'agent_note' in grading_results:
        print(f"Agent Note: {grading_results['agent_note']}")
    print(f"Feedback: {feedback_string}")
    print("===============================\n")
    print("[+] All Phases Complete.")

if __name__ == "__main__":
    main()
