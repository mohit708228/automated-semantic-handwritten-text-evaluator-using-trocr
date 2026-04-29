from src.ocr_pipeline import run_ocr_pipeline
from src.nlp_processing import process_student_answer
from src.similarity import compute_similarity
from src.scoring import calculate_grade
from src.feedback import generate_feedback
from src.agent_verifier import rescue_grade_if_needed

import nltk

def run_full_assessment(image_path, model_answer_text, max_marks=10):
    """
    Orchestrates the entire 6-phase pipeline end-to-end.
    Returns a unified dictionary with all analysis results.
    """
    results = {}
    
    # PHASE 1: OCR
    raw_ocr, ocr_conf = run_ocr_pipeline(image_path)
    results['raw_ocr'] = raw_ocr
    results['ocr_confidence'] = ocr_conf
    
    # Character-level answer deviation: measures how different the OCR output
    # is from the model answer at the character level (normalized edit distance).
    # NOTE: This is NOT true CER (which requires ground-truth transcription of the
    # handwritten image). It quantifies answer deviation, not OCR accuracy.
    ref_len = max(len(model_answer_text), 1)
    char_deviation = nltk.edit_distance(model_answer_text.lower(), raw_ocr.lower()) / ref_len
    results['answer_deviation'] = min(char_deviation, 1.0)  # clamp to [0, 1]
    
    # PHASE 2: NLP Cleaning
    nlp_results = process_student_answer(raw_ocr)
    results['nlp_cleaned'] = nlp_results['final_processed_string']
    
    # PHASE 3: Semantic Similarity
    sim_score = compute_similarity(nlp_results['final_processed_string'], model_answer_text)
    results['similarity_score'] = sim_score
    
    # PHASE 4: Initial Grading
    grading_metrics = calculate_grade(sim_score, max_marks=max_marks)
    
    # PHASE 6: Agent Verification (rescuing low scores)
    final_grades = rescue_grade_if_needed(
        grading_metrics, 
        student_raw_text=nlp_results['regex_cleaned_text'], 
        model_answer=model_answer_text
    )
    results['grading'] = final_grades
    
    # PHASE 5: Feedback Generation
    feedback_str = generate_feedback(
        student_raw_text=raw_ocr,
        model_raw_text=model_answer_text,
        grade_category=final_grades['grade_category']
    )
    results['feedback'] = feedback_str
    
    return results
