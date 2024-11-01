from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pinecone  # Updated import
import tempfile
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from fastapi.responses import FileResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pinecone.init(api_key="pcsk_292Lko_55sw5PAf8MGQ5iVVLTutAN7bGf8bcsZ3wQcXLudLNrLtEUcSpG2DajvtyxcJnf9")
index_name = "policy-compliance"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        pod_type="p1"  # Replace ServerlessSpec with 'p1' or another supported pod type
    )
index = pinecone.Index(index_name)

# Initialize Sentence-BERT model and QA pipeline
model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Compliance questions
compliance_questions = [
    "Does the policy specify types of data collected?",
    "Does the policy mention data retention periods?",
    "Does the policy address data sharing with third parties?",
    "Does the policy include user rights (e.g., access, deletion)?",
    "Is there a statement about cookies or tracking technologies?"
]

# Helper function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to answer predefined compliance questions
def analyze_compliance(policy_text):
    policy_text_sections = policy_text.split("\n\n")
    answers = []

    for question in compliance_questions:
        # Query each question and get answers from top matches
        query_vector = model.encode(question).tolist()
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        unique_answers = set()
        
        for match in results['matches']:
            section_id = match['id'].split('_')[-1]
            section_text = policy_text_sections[int(section_id)]
            answer = qa_pipeline(question=question, context=section_text).get('answer', None)
            if answer and answer not in unique_answers:
                unique_answers.add(answer)
                answers.append({"question": question, "answer": answer})
                break  # Take the first relevant answer if available
        if not unique_answers:
            answers.append({"question": question, "answer": "No relevant information found."})
    
    return answers

# Save report as PDF
def save_report_as_pdf(answers, filename="compliance_report.pdf"):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 40, "Compliance Report")
    c.setFont("Helvetica", 12)
    y_position = height - 80

    for answer_data in answers:
        question = answer_data["question"]
        answer = answer_data["answer"]
        c.drawString(40, y_position, f"{question}")
        y_position -= 20
        c.drawString(60, y_position, f"Answer: {answer}")
        y_position -= 40
        if y_position < 40:  # Add new page if close to bottom
            c.showPage()
            y_position = height - 40

    c.save()
    return file_path

@app.get("/")
async def root():
    return {"message": "Welcome to the Compliance Analysis API"}

@app.post("/analyze_policy")
async def analyze_policy(file: UploadFile = File(...)):
    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text and analyze compliance
    policy_text = extract_text_from_pdf(file_path)
    compliance_report = analyze_compliance(policy_text)
    
    # Save report as PDF
    pdf_path = save_report_as_pdf(compliance_report)

    return {"answers": compliance_report, "pdf_path": pdf_path}
