import cv2
video_path = "assets/stems.mp4"
cap = cv2.VideoCapture(video_path)

tracker = cv2.TrackerCSRT_create()
ret, frame = cap.read()
frame = cv2.resize(frame, (600, 900))
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 900))

    if not ret:
        break

    # Update the tracker for the current frame
    success, bbox = tracker.update(frame)

    if success:
        # If tracking is successful, get the bounding box coordinates
        x, y, w, h = [int(v) for v in bbox]
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the center point of the bounding box
        center_x, center_y = x + w // 2, y + h // 2

        # Print or store the coordinates
        print(f"Center coordinates: ({center_x}, {center_y})")
    else:
        # If tracking fails, display a message
        cv2.putText(frame, "Tracking failed", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()