import cv2
import numpy as np

# Initialize variables
roiPts = []
inputMode = False
roiBox = None
pause = False
zoom_factor = 1.2
frame_count = 1000
current_frame = 0

# Mouse callback function
def select_roi(event, x, y, flags, param):
    global roiPts, inputMode, roiBox, pause

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(param, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("Frame", param)

def zoom_generator(zoom_in_ratio, zoom_out_ratio, total_frames, zoom_speed):
    frame = 0
    while True:
        if frame < total_frames * zoom_in_ratio:
            # Zoom in phase
            zoom_level = 1 + 2 * (frame / (total_frames * zoom_in_ratio * zoom_speed))
        elif frame < total_frames * (zoom_in_ratio + zoom_out_ratio):
            # Zoom out phase
            zoom_level = 3 - 2 * ((frame - total_frames * zoom_in_ratio) / (total_frames * zoom_out_ratio * zoom_speed))
        else:
            # Reset the frame counter
            frame = 0
            zoom_level = 1

        frame += 1
        yield zoom_level
zoom_gen = zoom_generator(zoom_in_ratio=1, zoom_out_ratio=1, total_frames=100, zoom_speed=20)
def zoom(frame, roiBox, zoom_level):
    x, y, w, h = [int(v) for v in roiBox]

    # Calculate the center of the original ROI
    center_x, center_y = x + w // 2, y + h // 2

    # Calculate the size of the cropped region
    new_width = int(frame.shape[1] / zoom_level)
    new_height = int(frame.shape[0] / zoom_level)

    # Calculate the top-left corner of the cropped region
    x_start = max(0, center_x - new_width // 2)
    y_start = max(0, center_y - new_height // 2)

    # Make sure the cropped region is within the frame
    if x_start + new_width > frame.shape[1]:
        x_start = frame.shape[1] - new_width
    if y_start + new_height > frame.shape[0]:
        y_start = frame.shape[0] - new_height

    # Crop the frame
    cropped = frame[y_start:y_start+new_height, x_start:x_start+new_width]

    # Resize the cropped frame to the original frame's size
    zoomed = cv2.resize(cropped, (frame.shape[1], frame.shape[0]))

    return zoomed
# Create a tracker
tracker = cv2.TrackerCSRT_create()

# Open the video
cap = cv2.VideoCapture('assets/stems.mp4')

# Set the mouse callback function
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_roi)

while True:
    if not pause:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (600, 900))  # You can adjust the size as needed

        # If the ROI has been computed
        if roiBox is not None:
            # Update the tracker
            success, roiBox = tracker.update(frame)

            # Draw the bounding box
            if success:
                x, y, w, h = [int(v) for v in roiBox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Zoom in or out on the ROI
                zoom_level = next(zoom_gen)
                print(zoom_level)
                frame = zoom(frame, roiBox, zoom_level)

            else:
                cv2.putText(frame, "Tracking failed", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Show the frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20) & 0xFF

    # If 'i' is pressed, enter input mode to select the ROI
    if key == ord("i") and len(roiPts) == 4:
        roiPts = []
        inputMode = False
        pause = False
    if key == ord("i") and len(roiPts) < 4:
        inputMode = True
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 900))  # You can adjust the size as needed

        cv2.imshow("Frame", frame)

        orig = frame.copy()
        pause = True

        while len(roiPts) < 4:
            cv2.waitKey(0)

        # Determine the top-left and bottom-right points
        roiPts = np.array(roiPts)
        s = roiPts.sum(axis=1)
        tl = roiPts[np.argmin(s)]
        br = roiPts[np.argmax(s)]

        # Grab the ROI for the bounding box and initialize the
        # tracker
        roi = orig[tl[1]:br[1], tl[0]:br[0]]
        roiBox = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
        tracker.init(frame, roiBox)
        pause = False

    # If 'q' is pressed, stop the loop
    elif key == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()