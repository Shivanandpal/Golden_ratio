import streamlit as st
import cv2
import mediapipe
import numpy as np
from PIL import Image
print(mediapipe.__file__)

# Golden Ratio
GOLDEN_RATIO = 1.618

# Correct mediapipe import
mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def golden_score(r):
    diff = abs(r - GOLDEN_RATIO)
    return max(0, (1 - diff/GOLDEN_RATIO))*100


st.title("Golden Ratio Face Analyzer")

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            lm = face_landmarks.landmark

            chin = lm[152]
            forehead = lm[10]
            left_cheek = lm[234]
            right_cheek = lm[454]
            left_eye = lm[33]
            right_eye = lm[263]
            nose_top = lm[168]
            nose_bottom = lm[2]
            mouth_top = lm[13]
            mouth_bottom = lm[14]

            face_height = distance(forehead, chin, w, h)
            face_width = distance(left_cheek, right_cheek, w, h)
            eye_distance = distance(left_eye, right_eye, w, h)
            nose_length = distance(nose_top, nose_bottom, w, h)
            mouth_height = distance(mouth_top, mouth_bottom, w, h)

            ratio1 = face_height / face_width
            ratio2 = face_width / eye_distance
            ratio3 = nose_length / mouth_height

            score1 = golden_score(ratio1)
            score2 = golden_score(ratio2)
            score3 = golden_score(ratio3)

            final_score = (score1 + score2 + score3)/3

            for point in lm:
                x,y = int(point.x*w), int(point.y*h)
                cv2.circle(img,(x,y),1,(0,255,0),-1)

        passport_img = cv2.resize(img, (300, 400))  # width, height
        st.image(passport_img)

        st.subheader("Golden Ratio Results")

        st.write("Face Height / Face Width:", round(ratio1,3))
        st.write("Face Width / Eye Distance:", round(ratio2,3))
        st.write("Nose Length / Mouth Height:", round(ratio3,3))

        st.subheader(f"Golden Ratio Score: {round(final_score,2)} %")

        if final_score > 70:
            st.success("Face closely follows Golden Ratio")
        else:
            st.warning("Face does not follow Golden Ratio")

    else:
        st.error("No face detected")