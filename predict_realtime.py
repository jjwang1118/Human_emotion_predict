import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import cv2

# 載入模型
try:
    model = load_model('my_model_02.h5')
    print("load model successful")
except Exception as e:
    print("load model fail")
    print(e)

# 情緒對應表
emotion_map = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# 初始化 Haar cascade 臉部偵測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 灰階處理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        # 安全截取 ROI 區域
        h_img, w_img = gray.shape
        x1 = max(0, x - 20)
        y1 = max(0, y - 20)
        x2 = min(w_img, x + w + 20)
        y2 = min(h_img, y + h + 20)

        roi_gray = gray[y1:y2, x1:x2]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))  # (1, 48, 48, 1)

        preds = model.predict(roi)
        emotion = emotion_map[np.argmax(preds)]

        # 畫出框與情緒標籤
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # 顯示預測細節
        print("ROI shape:", roi.shape)
        print("ROI max value:", roi.max(), "min:", roi.min())
        emotion_probs = preds[0]
        for i, prob in enumerate(emotion_probs):
            print(f"{emotion_map[i]}: {prob:.2f}")

    cv2.imshow("Emotion Detection", frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
