import cv2
import mediapipe as mp
import math

# Inisialisasi Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Menginisialisasi kamera
cap = cv2.VideoCapture(0)

def count_raised_fingers(hand_landmarks):
    fingers_status = []
    
    # Landmark yang akan diperiksa (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    # Ambil koordinat untuk setiap fingertip dan MCP untuk thumb
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # Interphalangeal joint
    
    # Periksa apakah jempol terangkat
    fingers_status.append(thumb_tip.y < thumb_ip.y)
    
    # Periksa apakah jari lainnya terangkat
    for tip in finger_tips[1:]:
        finger_tip = hand_landmarks.landmark[tip]
        finger_pip = hand_landmarks.landmark[tip - 2]  # PIP adalah dua posisi sebelum TIP
        fingers_status.append(finger_tip.y < finger_pip.y)
    
    return fingers_status

def recognize_gesture(fingers_status, hand_landmarks):
    count = sum(fingers_status)
    
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    # Hitung jarak antara ujung jari jempol dan telunjuk
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    distance_index_thump = math.sqrt((thumb_tip.x - index_finger_mcp.x)**2 + (thumb_tip.y - index_finger_mcp.y)**2)
    
    if distance < 0.05 and count == 2:  # threshold untuk mendeteksi apakah ujung jari berdempetan
        return "cintaaa"
    
    elif count == 5 and distance_index_thump < 0.05:
        return "Empat"
    elif count == 4 and distance_index_thump < 0.05:
        return "Tiga"
    elif count == 3 and distance_index_thump < 0.05:
        return "Dua"
    elif count == 2 and distance_index_thump < 0.05:
        return "Satu"
    elif count == 5:
        return "Lima"
    else:
        return "Tonjok"

# Menggunakan model Mediapipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Gagal mengambil gambar dari kamera.")
            break
        
        # Membalikkan gambar secara horizontal untuk mirror view
        image = cv2.flip(image, 1)
        
        # Mengkonversi warna gambar dari BGR ke RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Proses deteksi tangan
        results = hands.process(image_rgb)
        
        # Gambar titik dan garis pada tangan yang terdeteksi
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Hitung jumlah jari yang terangkat
                fingers_status = count_raised_fingers(hand_landmarks)
                gesture = recognize_gesture(fingers_status, hand_landmarks)
                
                # Tampilkan hasil deteksi
                cv2.putText(image, gesture, (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
        # Tampilkan gambar
        cv2.imshow('Hand Tracking', image)
        
        # Keluar dari loop saat menekan tombol 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Melepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
