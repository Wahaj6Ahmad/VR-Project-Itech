import cv2  
import numpy as np  


# Set ArUco dictionary and parameters  
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  
parameters = cv2.aruco.DetectorParameters_create()  

# Define A4 paper dimensions (in meters)  
a4_width = 0.071
a4_height = 0.156

# Set marker length (in meters)  
marker_length = 0.05
  
# load the camera calibration data
data = np.load('camera_calibration.npz')

# Camera matrix and distortion coefficients from camera calibration  
camera_matrix = data['mtx']  
dist_coeffs = data['dist']

# Open webcam  
cap = cv2.VideoCapture(0)  
  
while True:  
    # Capture frame from webcam  
    ret, input_image = cap.read()  
    if not ret:  
        break

#     import cv2  
# import numpy as np  
  
# # Initialize webcam capture  
# cap = cv2.VideoCapture(0)  
  
# while True:  
#     # Capture frame from webcam  
#     ret, input_image = cap.read()  
#     if not ret:  
#         break  
  
    # # Create a transparent layer (same size as input_image)  
    # transparent_layer = np.zeros_like(input_image, dtype=np.uint8)  
  
    # # Customize your transparent layer here, e.g., add text, shapes, etc.  
    # # Example: Add text to the top-left corner  
    # font = cv2.FONT_HERSHEY_SIMPLEX  
    # text = "Hello, World!"  
    # color = (255, 0, 0)  # Blue color  
    # thickness = 2  
    # cv2.putText(transparent_layer, text, (10, 30), font, 1, color, thickness, cv2.LINE_AA)  
  
    # # Overlay the transparent layer on the input_image  
    # input_image = cv2.addWeighted(input_image, 1, transparent_layer, 0.7, 0)  
  

  
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  

    # Detect ArUco markers  
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Load the image to project  
    image_to_project = cv2.imread("./uploaded_images/image.png", cv2.IMREAD_UNCHANGED)  
    print(image_to_project.shape)
  
    if ids is not None:  
        # Estimate marker poses  
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)  
  
        for i, rvec in enumerate(rvecs):  
            # Draw the detected marker  
            cv2.aruco.drawDetectedMarkers(input_image, corners, ids)  
  
            # Define the 3D coordinates of the A4 paper vertices relative to the marker center  
            half_width = a4_width / 2  
            half_height = a4_height / 2  
            a4_points = np.float32([  
                [-half_width, -half_height, 0], [half_width, -half_height, 0], [half_width, half_height, 0], [-half_width, half_height, 0]  
            ])  
  
            # Project the A4 paper vertices onto the input image  
            a4_points_proj, _ = cv2.projectPoints(a4_points, rvec, tvecs[i], camera_matrix, dist_coeffs)  

            # Compute the homography between the image_to_project and the input image  
            image_points_proj = np.int32(a4_points_proj.reshape(-1, 2))  
            h, _ = cv2.findHomography(np.float32([[0, 0], [image_to_project.shape[1], 0], [image_to_project.shape[1], image_to_project.shape[0]], [0, image_to_project.shape[0]]]), image_points_proj)  

            # Warp the image_to_project using the computed homography and blend it with the input image  
            warped_image = cv2.warpPerspective(image_to_project, h, (input_image.shape[1], input_image.shape[0]))  
            mask = cv2.warpPerspective(np.ones_like(image_to_project) * 255, h, (input_image.shape[1], input_image.shape[0])) > 0  
            
            # Add alpha channel to webcam
            (h, w) = input_image.shape[:2]
            image_to_project = np.dstack([image_to_project, np.ones((h, w), dtype="uint8") * 0])

            
            input_image[mask[..., 0]] = cv2.addWeighted(input_image[mask[..., 0]], 0, warped_image[mask[..., 0]], 1, 0)  
  
    # Display the result  
    cv2.imshow('Result', input_image)  
  
    # Break the loop if 'q' is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  



# Release the webcam and close the windows  
cap.release()  
cv2.destroyAllWindows()  