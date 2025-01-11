import cv2


def crop_video_with_preview(input_video_path, output_video_path, roi):
    """
    Crops the video to the specified region of interest (ROI) with real-time preview.

    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path to save the cropped output video.
        roi (tuple): Region of interest as (x, y, width, height).
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (roi[2], roi[3]))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        x, y, w, h = roi
        cropped_frame = frame[y:y + h, x:x + w]

        # Display the original and cropped frames side-by-side
        combined_frame = cv2.hconcat([frame, cv2.resize(cropped_frame, (frame_width, frame_height))])
        cv2.imshow("Original (Left) vs Cropped (Right)", combined_frame)

        # Write the cropped frame to the output video
        out.write(cropped_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Cropped video saved to:", output_video_path)


# Define the region of interest (adjust based on your video)
# Example ROI: focus on the bottom half of the frame
input_video_path = "miamifirstbasic.avi"  # Replace with your video file
output_video_path = "cropped_video.avi"
roi = (0, 300, 640, 300)  # Example ROI (x, y, width, height)

# Crop the video with preview
crop_video_with_preview(input_video_path, output_video_path, roi)
