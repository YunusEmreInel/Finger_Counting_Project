import cv2               # OpenCV library
import mediapipe as mp   # MediaPipe library

cap = cv2.VideoCapture(0)               # Start the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Set the width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set the height to 720 pixels

mpHand = mp.solutions.hands           # Initialize the MediaPipe hands module
hands = mpHand.Hands(max_num_hands=4) # Create a hands detection object with Hands class
mpDraw = mp.solutions.drawing_utils   # Drawing utility functions

tipIds = [4, 8, 12, 16, 20]      # Finger tip landmark IDs

while True:                      # Start an infinite loop
    success, img = cap.read()                       # Read the image from the camera
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert the image from BGR to RGB
    
    results = hands.process(imgRGB)                 # Hands detection process
    totalFingers = 0                                # Total number of open fingers
    
    if results.multi_hand_landmarks:                  # If hands are detected
        for handLms in results.multi_hand_landmarks:  # For each detected hand
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)   # Draw hand connections

            lmList = []  # Initialize the landmark list
            for id, lm in enumerate(handLms.landmark):  # For each landmark
                h, w, _ = img.shape                     # Get image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)   # Calculate landmark coordinates
                lmList.append([id, cx, cy])             # Add landmark ID and coordinates to the list

            if len(lmList) != 0:  # If the landmark list is not empty
                fingers = []  # List to hold finger states

                # Determine if it is the right hand or left hand
                isRightHand = lmList[tipIds[0]][1] > lmList[tipIds[4] - 2][1]  # Check if the thumb is to the right of the little finger

                # Thumb state
                if isRightHand:
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                        fingers.append(1)  # Thumb is open
                    else:
                        fingers.append(0)  # Thumb is closed
                else:  # Left hand
                    if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                        fingers.append(1)  # Thumb is open
                    else:
                        fingers.append(0)  # Thumb is closed

                # Other 4 fingers
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # Check finger positions
                        fingers.append(1)  # Finger is open
                    else:
                        fingers.append(0)  # Finger is closed

                print(fingers)  # Print finger states (for debugging)
    
                totalFingers += fingers.count(1)  # Calculate the total number of open fingers
    
    cv2.putText(img, str(totalFingers), (30, 125), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0), 8)  # Display the number of open fingers on the screen
    cv2.imshow("img", img)  # Show the image on the screen

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit when 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
