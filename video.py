import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import simpledialog
import pygame
from moviepy.editor import VideoFileClip, AudioFileClip

# Initialize variables

lerp_step =0 
lerp_frames = 20
roiPts = []
roiKeyDict = {}
frame_count = 0
inputMode = False
roiBox = None
pause = False
zoom_factor = 1.2
frame_count = 1000
current_frame = 0
oldROIBox = None 
video_path = None
# Function to show a popup window and get the file name from the user
def get_video_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = simpledialog.askstring("Input", "Please enter the video file name:")
    root.destroy()
    return video_path

# Get the video path from the userpygame.mixer.init()

zoomDict = {}
current_zoom = (False, 1, 0)   # (zoom_in, zoom_level, zoom_timing)
zoom_index = 0
oldZoomValue = 1
current_pas_zoom = 0
soundFilePath = 'vids/output.wav'
inputFilePath = 'vids/guitarMountainFuckTrim.mp4'
pygame.mixer.music.load(soundFilePath)
pygame.mixer.music.play(-1)

def lerp(a, b, t):
    return a * (1 - t) + b * t
# Mouse callback function
def select_roi(event, x, y, flags, param):
    global roiPts, inputMode, roiBox, pause

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(param, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("Frame", param)

def zoom_generator(zoom_in_ratio, zoom_out_ratio, total_frames, zoom_speed, brutal_variation_speed=1):
    frame = 0
    while True:
        if frame < total_frames * zoom_in_ratio:
            # Zoom in phase
            zoom_level = 1 + 1.5 * (frame / (total_frames * zoom_in_ratio * zoom_speed))
        elif frame < total_frames * (zoom_in_ratio + zoom_out_ratio):
            # Zoom out phase
            zoom_level = 2.5 - 1.5 * ((frame - total_frames * zoom_in_ratio) / (total_frames * zoom_out_ratio * zoom_speed))
        else:
            # Reset the frame counter
            frame = 0
            zoom_level = 1

        # Apply brutal variation if needed
        zoom_level *= brutal_variation_speed

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
def get_zoom():
    global zoom_index, oldZoomValue, current_pas_zoom
    if zoom_index < current_zoom[2]:
        zoom_index += 1
        return oldZoomValue + current_pas_zoom * zoom_index
    else:
        return current_zoom[1]
    
# Create a tracker
tracker = cv2.TrackerCSRT_create()

# Get the video path from the user
video_path = get_video_path()
# Open the video
cap = cv2.VideoCapture(inputFilePath)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame size: {frame_width} x {frame_height}")
print(inputFilePath)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Frame rate: {fps}")
def render_video(roiKeyDict, video_path, output_path):
    global roiBox, lerp_step, lerp_frames, current_frame, oldROIBox, current_zoom, oldZoomValue, current_pas_zoom, zoom_index
    print("Rendering video...")
    cap = cv2.VideoCapture(video_path)
    # Define the codec and create a VideoWriter object
    current_frame = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (506, 900))

    roiBox = None
    oldROIBox = None
    lerp_step = 0
    lerp_frames = 20
    current_frame = 0
    while True:
        ret, frame = cap.read()
        current_frame += 1
        if not ret:
            break
        # Resize the frame
        frame = cv2.resize(frame, (506, 900))  # You can adjust the size as needed
        if zoomDict.get(current_frame) is not None:
            oldZoomValue = current_zoom[1]
            current_zoom = zoomDict[current_frame]
            zoom_index = 0 
            current_pas_zoom = (current_zoom[1] - oldZoomValue) / current_zoom[2]

        # If the ROI has been computed
        if roiKeyDict.get(current_frame) is not None:
            oldROIBox = roiBox
            roiBox = roiKeyDict[current_frame]
            tracker.init(frame, roiBox)
            success, roiBox = tracker.update(frame)
            lerp_step = 0

        if roiBox:    # Zoom in or out on the ROI
            success, roiBox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in roiBox]
                if lerp_step < lerp_frames and oldROIBox is not None:
                    x = int(lerp(oldROIBox[0], x, lerp_step / lerp_frames))
                    y = int(lerp(oldROIBox[1], y, lerp_step / lerp_frames))
                    w = int(lerp(oldROIBox[2], w, lerp_step / lerp_frames))
                    h = int(lerp(oldROIBox[3], h, lerp_step / lerp_frames))
                    roiBox = (x, y, w, h)
                    lerp_step += 1

                zoom_level = get_zoom() #next(zoom_gen)
                frame = zoom(frame, roiBox, zoom_level)

        # Write the frame to the output file
        out.write(frame)

        print("Frame:", current_frame)
    cap.release()
    out.release()
    print("Video rendered successfully")
    # Load your video
    video = VideoFileClip(output_path)

    # Load your audio
    audio = AudioFileClip(soundFilePath)

    # Add the audio to the video
    final_video = video.set_audio(audio)

    # Write the result to a file
    final_video.write_videofile('final_output.mp4', codec='libx264')# Set the mouse callback function
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_roi)

while True:
    if not pause:
        # Read a new frame
        ret, frame = cap.read()
        current_frame += 1
        if not ret:
             # If the video has ended, reset the counter and pause the video
            cap = cv2.VideoCapture('assets/'+video_path)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (506, 900))  # You can adjust the size as needed

            current_frame = 1
            pause = True
            pygame.mixer.music.pause()

            continue

        # Resize the frame
        frame = cv2.resize(frame, (506, 900))  # You can adjust the size as needed
        if(roiKeyDict.get(current_frame) is not None):
            oldROIBox = roiBox
            print("updated ROI" + str(current_frame))
            roiBox = roiKeyDict[current_frame]
            tracker.init(frame, roiBox)
            success, roiBox = tracker.update(frame)
            lerp_step=0
                
        if zoomDict.get(current_frame) is not None:
            oldZoomValue = current_zoom[1]
            current_zoom = zoomDict[current_frame]
            zoom_index = 0 
            current_pas_zoom = (current_zoom[1] - oldZoomValue) / current_zoom[2]


        if roiBox is not None:
            # Update the tracker
            success, roiBox = tracker.update(frame)
            # Draw the bounding box
            if success:
                x, y, w, h = [int(v) for v in roiBox]
                if lerp_step < lerp_frames and oldROIBox is not None:
                    x = int(lerp(oldROIBox[0], x, lerp_step / lerp_frames))
                    y = int(lerp(oldROIBox[1], y, lerp_step / lerp_frames))
                    w = int(lerp(oldROIBox[2], w, lerp_step / lerp_frames))
                    h = int(lerp(oldROIBox[3], h, lerp_step / lerp_frames))
                    roiBox = (x, y, w, h)
                    lerp_step += 1
            
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Zoom in or out on the ROI
                zoom_level = get_zoom()#next(zoom_gen)
                frame = zoom(frame, roiBox, zoom_level)

            else:
                cv2.putText(frame, "Tracking failed", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Show the frame
    cv2.imshow("Frame", frame)
    if roiBox is None:
        key = cv2.waitKey(25) & 0xFF
    else:
        key = cv2.waitKey(8) & 0xFF
    if key == ord("d"):
        print(roiKeyDict)
        max_key = max(k for k in roiKeyDict if k < current_frame)
        print("last key deleted")        
        # Delete the entry with the largest key less than current_frame
        del roiKeyDict[max_key]
    # If 'i' is pressed, enter input mode to select the ROI
    if key == ord(" ") :
        print("back to zero")
        roiBox = None
        pygame.mixer.music.load(soundFilePath)
        pygame.mixer.music.play(-1)
        current_frame = 0
        pause = False
        cap = cv2.VideoCapture('assets/'+video_path)
        
    if key == ord("i") and len(roiPts) == 4:
        roiPts = []
        inputMode = False
        pause = False
        pygame.mixer.music.unpause()


    if key == ord("s"):
        roiDict_int = {int(k): (int(v[0]), int(v[1]), int(v[2]), int(v[3])) for k, v in roiKeyDict.items()}
        with open('roiDict.json', 'w') as file:
            json.dump(roiDict_int, file)
        with open('zoomDict.json', 'w') as file:
            json.dump(zoomDict, file)
        print("Data saved to roiDict.json")

    if key == ord("z"):
        pygame.mixer.music.pause()
        print("Zoom in or out? Enter 't' for zoom in, 'f' for zoom out:")
        zoom_in = input().lower() == 't'
        print("Enter the final zoom level (1-3):")
        zoom_level = float(input())
        zoom_level = max(1, min(3, zoom_level))  # Ensure zoom level is between 1 and 3
        print("Enter the timing of the zoom effect (0-300):")
        zoom_timing = int(input())
        zoom_timing = max(0, min(300, zoom_timing))
        zoomDict[current_frame] = (zoom_in, zoom_level, zoom_timing)
        oldZoomValue = current_zoom[1]
        current_zoom = zoomDict[current_frame]
        zoom_index = 0 
        current_pas_zoom = (current_zoom[1] - oldZoomValue) / current_zoom[2]
        zoomDict[current_frame] = (zoom_in, zoom_level, zoom_timing)
        current_zoom = zoomDict[current_frame]
        pygame.mixer.music.unpause()

    if key == ord("l"):
        with open('roiDict.json', 'r') as file:
            roiKeyDict = json.load(file)
            roiKeyDict = {int(k): v for k, v in roiKeyDict.items()}
        with open('zoomDict.json', 'r') as file:
            zoomDict = json.load(file)
            zoomDict = {int(k): v for k, v in zoomDict.items()}
            print(roiKeyDict)

    if key == ord("i") and len(roiPts) < 4:
        pygame.mixer.music.pause()
        inputMode = True
        ret, frame = cap.read()
        frame = cv2.resize(frame, (506, 900))  # You can adjust the size as needed

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
        roiKeyDict[current_frame] = roiBox
        tracker.init(frame, roiBox)
        pause = False
        pygame.mixer.music.unpause()

    if key == ord("p"):
        pause = not pause
        if pause:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()
        if key == ord("q"):
            break
    if key == ord('f'):
        pygame.mixer.music.pause()
        render_video(roiKeyDict, video_path, 'output.mp4')
    # If 'q' is pressed, stop the loop
    elif key == ord("q"):
        break

# ADD AUTO ADJUSTING FRAME SIZE
pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()