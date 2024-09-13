import cv2
from ultralytics import YOLO
import time
import threading
import random
import torch

# YOLOv8 모델 불러오기 (분류 모델)
model = YOLO("best_jmk.pt")

# 전역 변수
frame = None
lock = threading.Lock()  # 스레드 안전을 위한 락
countdown_done = False  # 카운트다운 완료 상태를 추적하는 변수
capture_running = True  # 동영상 캡처 상태를 추적하는 변수
info_displayed = False  # 정보 출력 여부를 추적하는 변수

def video_capture():
    global frame, capture_running
    cap = cv2.VideoCapture(0)
    while capture_running:
        ret, f = cap.read()
        if not ret:
            break
        with lock:
            frame = f
    cap.release()

def process_frame():
    global frame, countdown_done, info_displayed
    countdown_start_time = None
    countdown_number = None
    CA = ""
    
    while True:
        with lock:
            if frame is None:
                continue

            # YOLO 모델을 사용하여 예측 수행
            results = model(frame)
            # 예측 결과 가져오기 (리스트에서 첫 번째 결과)
            if results and len(results) > 0:
                result = results[0]  # 첫 번째 결과

                # 결과 객체의 모든 속성 및 메서드 출력
                if not countdown_done and not info_displayed:
                    # 예측 확률을 가져오기
                    if hasattr(result, 'probs') and result.probs is not None:
                        # Access the tensor from the Probs object
                        probs_tensor = result.probs.data
                        print("Probs tensor content:", probs_tensor)

                        # 텐서의 최대값과 인덱스 추출
                        max_value, max_index = torch.max(probs_tensor, dim=0)
                        max_label = model.names[max_index.item()]  # 해당 클래스 이름
                        print(f"Highest Probability Label: {max_label} ({max_value.item():.2f})")

                # 카운트다운 타이머
            current_time = time.time()
            if countdown_start_time is None:
                countdown_start_time = current_time

            elapsed_time = current_time - countdown_start_time

            if elapsed_time <= 1:
                countdown_number = 3
            elif elapsed_time <= 2:
                countdown_number = 2
            elif elapsed_time <= 3:
                countdown_number = 1
            else:
                countdown_number = None
                
                # Class labels: {0: 'paper', 1: 'rock', 2: 'scissors'}
                if not countdown_done:
                    # 카운트다운이 끝난 후 컴퓨터의 랜덤 응답을 결정합니다.
                    computer_Answer = random.randrange(0, 3)
                    if computer_Answer == 0:
                        CA = "COMPUTER - rock"
                    elif computer_Answer == 1:
                        CA = "COMPUTER - scissors"
                    else:
                        CA = "COMPUTER - paper"
                    
                    countdown_done = True  # 카운트다운이 완료되었음을 표시
                    info_displayed = True  # 3초 후 정보 출력을 중단하도록 설정

            # 카운트다운 숫자 또는 결과 메시지 표시
            if countdown_number is not None:
                cv2.putText(frame, str(countdown_number), (frame.shape[1] // 2 - 50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif countdown_done:
                # 카운트다운이 끝난 후 컴퓨터의 응답을 표시
                cv2.putText(frame, CA, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, max_label, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 결과 창 띄우기
            cv2.imshow("Gesture Detection", frame)

            # 'q'를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 화면 종료
    cv2.destroyAllWindows()

# 스레드 생성
video_thread = threading.Thread(target=video_capture)
process_thread = threading.Thread(target=process_frame)

# 스레드 시작
video_thread.start()
process_thread.start()

# 'q'를 눌러서 종료할 때까지 기다리기
process_thread.join()

# 동영상 캡처 상태를 업데이트하여 캡처 스레드를 종료
capture_running = False
video_thread.join()
