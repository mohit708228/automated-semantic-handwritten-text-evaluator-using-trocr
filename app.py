import streamlit as st
import os
from PIL import Image

# Import our modular pipeline tools
from src.pdf_utils import extract_text_from_pdf
from src.pipeline_runner import run_full_assessment

# Set up page configurations
st.set_page_config(page_title="Automated Assessor", page_icon="📝", layout="centered")

st.title("📝 Automated Handwritten Assignment Assessor")
st.markdown("Upload a student's handwritten assignment (Image) and the corresponding grading key (PDF) to automatically grade the assignment using Computer Vision and NLP.")

# --- SIDEBAR / OPTIONS ---
max_score = st.sidebar.number_input("Maximum Assignment Score", min_value=1, value=10, step=1)

# --- UPLOADERS ---
st.subheader("1. Upload Grading Key")
pdf_file = st.file_uploader("Upload Model Answer (PDF format)", type=["pdf"])

st.subheader("2. Upload Student Submission")
img_file = st.file_uploader("Upload Handwriting (PNG, JPG format)", type=["png", "jpg", "jpeg"])

if pdf_file and img_file:
    # Preview Image
    st.image(Image.open(img_file), caption="Student Submission Preview", use_column_width=True)
    
    if st.button("Evaluate Assignment", type="primary"):
        with st.spinner("Processing Assignment... (Extracting OCR & Running NLP Models)"):
            
            # --- 1. Save uploaded image temporarily ---
            temp_img_dir = "data/sample_images"
            if not os.path.exists(temp_img_dir):
                os.makedirs(temp_img_dir)
            
            temp_img_path = os.path.join(temp_img_dir, "temp_upload_img." + img_file.name.split('.')[-1])
            with open(temp_img_path, "wb") as f:
                f.write(img_file.getbuffer())
                
            # --- 2. Save uploaded PDF temporarily ---
            temp_pdf_path = os.path.join(temp_img_dir, "temp_upload_key.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
                
            # --- 3. Extract Text from Grading Key ---
            try:
                model_answer_text = extract_text_from_pdf(temp_pdf_path)
            except Exception as e:
                st.error("Could not parse PDF file.")
                st.stop()
                
            if not model_answer_text:
                st.error("No text found in the PDF document.")
                st.stop()
                
            # --- 4. Run Core Pipeline ---
            results = run_full_assessment(temp_img_path, model_answer_text, max_marks=max_score)
            
            # --- 5. Display Results Dashboard ---
            st.success("Evaluation Complete!")
            
            # Use columns for layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Awarded Score", value=f"{results['grading']['awarded_marks']} / {max_score}")
            with col2:
                st.metric(label="Similarity Index", value=f"{results['similarity_score']:.2f} / 1.00")
                
            col3, col4 = st.columns(2)
            with col3:
                st.metric(label="OCR Confidence", value=f"{(results.get('ocr_confidence', 0))*100:.1f}%")
            with col4:
                st.metric(label="Character Error Rate (CER)", value=f"{(results.get('ocr_cer', 0))*100:.1f}%")
                
            st.markdown("### Feedback Summary")
            st.info(results['feedback'])
            
            if 'agent_note' in results['grading']:
                st.warning(f"🤖 **AI Agent Context Rescue:** {results['grading']['agent_note']}")
                
            # Give an expandable view of the data processing steps
            with st.expander("View Processing Details (Logs)"):
                st.markdown("**1. Raw OCR Extracted:**")
                st.write(results['raw_ocr'])
                
                st.markdown("**2. Regex Cleaned Token Output:**")
                st.write(results['nlp_cleaned'])
                
                st.markdown("**3. Model Answer Extraction (from PDF):**")
                st.write(model_answer_text)
                
            # Clean up temporary files
            try:
                os.remove(temp_img_path)
                os.remove(temp_pdf_path)
            except:
                pass
