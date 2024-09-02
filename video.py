import cv2
import numpy as np

# Initialize variables
roiPts = []
inputMode = False
roiBox = None
pause = False
zoom_factor = 2
frame_count = 100
current_frame = 0

# Mouse callback function
def select_roi(event, x, y, flags, param):
    global roiPts, inputMode, roiBox, pause

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(param, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("Frame", param)


def zoom(frame, roiBox, zoom_factor, frame_count, current_frame):
    x, y, w, h = [int(v) for v in roiBox]

    # Calculate the amount of zoom for this frame
    zoom_level = 1 + (zoom_factor - 1) * (current_frame / frame_count)

    # Calculate the size of the zoomed ROI
    w_zoom = int(w * zoom_level)
    h_zoom = int(h * zoom_level)

    # Calculate the top-left corner of the zoomed ROI
    x_zoom = max(0, x - (w_zoom - w) // 2)
    y_zoom = max(0, y - (h_zoom - h) // 2)

    # Make sure the zoomed ROI is within the frame
    if x_zoom + w_zoom > frame.shape[1]:
        w_zoom = frame.shape[1] - x_zoom
    if y_zoom + h_zoom > frame.shape[0]:
        h_zoom = frame.shape[0] - y_zoom

    # Crop and resize the frame
    cropped = frame[y_zoom:y_zoom+h_zoom, x_zoom:x_zoom+w_zoom]
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
            # Zoom in on the ROI
            if current_frame < frame_count:
                frame = zoom(frame, roiBox, zoom_factor, frame_count, current_frame)
                current_frame += 1

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