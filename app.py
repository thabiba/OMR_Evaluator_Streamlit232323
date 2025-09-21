import streamlit as st
import os
import cv2
import json
import sqlite3
import numpy as np

# Setup folders
os.makedirs("uploaded", exist_ok=True)
os.makedirs("answer_keys", exist_ok=True)

# Bubble detection
def detect_bubbles(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 250 < area < 600:
            x, y, w, h = cv2.boundingRect(cnt)
            bubbles.append((x, y, w, h))
    bubbles = sorted(bubbles, key=lambda b: (b[0], b[1]))
    return bubbles

def is_marked(img, bubble):
    x, y, w, h = bubble
    roi = img[y:y+h, x:x+w]
    filled_pixels = cv2.countNonZero(roi)
    total_pixels = roi.size
    fill_ratio = filled_pixels / total_pixels
    return fill_ratio > 0.5

def map_bubbles_to_questions(bubbles):
    question_map = {}
    group_size = 5
    for i in range(0, len(bubbles), group_size):
        q_no = (i // group_size) + 1
        options = ['A', 'B', 'C', 'D', 'E']
        vertical_group = sorted(bubbles[i:i+group_size], key=lambda b: b[1])
        question_map[q_no] = dict(zip(options, vertical_group))
    return question_map

def get_marked_answers(img, question_map):
    marked = {}
    for q_no, options in question_map.items():
        for opt, bubble in options.items():
            if is_marked(img, bubble):
                marked[q_no] = opt
                break
    return marked

def load_answer_key(set_name):
    path = f"answer_keys/{set_name}.json"
    with open(path, "r") as f:
        return json.load(f)

def evaluate_answers(marked_answers, answer_key):
    score = 0
    subject_scores = {}
    for subject, answers in answer_key.items():
        correct = 0
        for q_no, correct_opt in answers.items():
            try:
                q_no_int = int(q_no)
                if q_no_int in marked_answers and marked_answers[q_no_int].upper() == correct_opt.upper():
                    correct += 1
            except Exception as e:
                st.warning(f"⚠ Error comparing Q{q_no}: {e}")
        subject_scores[subject] = correct
        score += correct
    return score, subject_scores

def save_result(student_id, score):
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS results (
                        student_id TEXT,
                        score INTEGER)''')
    cursor.execute("INSERT INTO results VALUES (?, ?)", (student_id, score))
    conn.commit()
    conn.close()

# Streamlit UI
st.title("📄 OMR Evaluation System")

student_name = st.text_input("Enter Student Name")
uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpg", "jpeg", "png"])

if uploaded_file and student_name:
    file_path = os.path.join("uploaded", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = cv2.imread(file_path, 0)
    bubbles = detect_bubbles(file_path)
    question_map = map_bubbles_to_questions(bubbles)
    marked_answers = get_marked_answers(img, question_map)
    answer_key = load_answer_key("set_a")
    score, subject_scores = evaluate_answers(marked_answers, answer_key)

    save_result(student_name, score)

    st.success(f"✅ Result for {student_name}")
    st.write(f"*Total Score:* {score}")
    st.write("*Subject-wise Scores:*")
    for subject, marks in subject_scores.items():
        st.write(f"- {subject}: {marks}")