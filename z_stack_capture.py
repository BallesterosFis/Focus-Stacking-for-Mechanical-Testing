import openflexure_microscope_client as ofm_client
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import requests
import time
import io
from PIL import Image
import msvcrt

def capture_full_resolution_image(base_uri, params: dict = None):
    """
    Captures a full-resolution image from the microscope by interacting directly with the API.
    This bypasses the client's grab_image() for higher quality stills.

    Args:
        base_uri (str): The base API URL of the microscope (e.g., "http://host:port/api/v2").
        params (dict, optional): Additional parameters to pass to the capture API endpoint.

    Returns:
        numpy.ndarray: The captured image as a NumPy array.
    """
    # Set up the payload for the capture request. 'use_video_port: False' ensures a full-resolution still capture.
    payload = {
        "use_video_port": False,
        "bayer": False,
    }
    if params:
        payload.update(params)

    # Initiate the capture process by calling the API endpoint
    r = requests.post(
        base_uri + "/actions/camera/capture",
        json=payload,
        headers={'Accept': 'application/json'},
        timeout=60
    )
    r.raise_for_status()  # Raise an exception for bad HTTP status codes
    # Get the URL to poll for the action's status and result
    action_href = json.loads(r.content.decode('utf-8'))['href']

    # Poll the action URL until the capture is complete and the image is available
    while True:
        r = requests.get(action_href)
        status = json.loads(r.content.decode('utf-8'))
        if status.get('output') is not None:
            # Once the capture is done, download the image from the provided URL
            download_url = status['output']['links']['download']['href']
            r = requests.get(download_url)
            # Open the image bytes and convert to a NumPy array
            return np.array(Image.open(io.BytesIO(r.content)))
        time.sleep(1)  # Wait before polling again to avoid overwhelming the server


def preview_live(microscope):
    """
    Displays a live video feed from the microscope in a matplotlib window.
    The preview runs until the user presses ENTER in the terminal or closes the window.
    """
    plt.ion()  # Turn on interactive mode for live updating
    fig, ax = plt.subplots(1, 1)
    img_display = None  # Placeholder for the image plot object

    print("Live stream active. Press ENTER in the terminal or close the window to start the scan.")

    while plt.fignum_exists(fig.number):
        # Grab the most recent video frame from the microscope
        img = microscope.grab_image()
        img_array = np.array(img)

        # Update the plot with the new frame
        if img_display is None:
            img_display = ax.imshow(img_array)
        else:
            img_display.set_data(img_array)

        ax.set_title("Live Preview")
        plt.pause(0.1)  # Pause to allow the plot to update

        # Check for key press (ENTER) to exit the preview
        if msvcrt.kbhit() and msvcrt.getch() == b'\r':
            break

    plt.close(fig)  # Clean up the plot window


def z_stack_scan(microscope, base_uri, folder_name, num_images, step_size):
    """
    Performs a Z-stack scan by moving the microscope stage and capturing an image at each Z position.

    Args:
        microscope: The OpenFlexure microscope client object.
        base_uri (str): The base API URL for high-res image capture.
        folder_name (str): Name of the folder to save images and log file.
        num_images (int): Number of images to capture in the stack.
        step_size (float): Distance to move the Z axis between captures.
    """
    # Create the directory for saving results, named after the sample/folder_name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Images will be saved to: {save_path}")

    # Calculate the target Z positions relative to the current position
    start_z = microscope.position["z"]
    z_positions = [start_z + i * step_size for i in range(num_images)]

    # Create a log file to record the exact Z position for each image index
    log_file = os.path.join(save_path, f"{folder_name}_z_positions.txt")
    with open(log_file, "w") as f:
        f.write("Index\tZ (actuator units)\n")

    # Initialize the plot for live display during the scan
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    img_display = None

    # Main scan loop: move to Z, capture image, save, log, and update display
    for idx, z in enumerate(z_positions):
        # Move the microscope stage to the target Z position
        pos = microscope.position.copy()
        pos["z"] = z
        microscope.move(pos)
        # Small pause to allow the stage to settle and avoid motion blur
        time.sleep(0.2)

        # Capture a high-resolution image at the current Z position
        img_array = capture_full_resolution_image(base_uri)

        # Save the image as a JPEG file with an index-padded filename
        filename = os.path.join(save_path, f"z_{idx:03d}.jpeg")
        Image.fromarray(img_array).save(filename, format="TIFF", compression="none")

        # Record the Z position for this image in the log file
        with open(log_file, "a") as f:
            f.write(f"{idx}\t{z}\n")

        print(f"[{idx+1}/{num_images}] Saved: {filename} | Z = {z:.2f}")

        # Update the live display with the newly captured image
        if img_display is None:
            img_display = ax.imshow(img_array)
        else:
            img_display.set_data(img_array)
        ax.set_title(f"Z Scan {idx+1}/{num_images} | Z = {z:.2f}")
        plt.pause(0.1)  # Pause to update the plot

    plt.ioff()
    plt.show()  # Keep the final image displayed until the window is closed
    print(f"Z-stack scan complete.\nPosition log saved to: {log_file}")


def main():
    """Main function to connect to the microscope and execute the Z-stack scan."""
    # Discover and connect to the first available microscope
    microscope = ofm_client.find_first_microscope()
    base_uri = f"http://{microscope.host}:{microscope.port}/api/v2"
    print(f"Connected to microscope at {microscope.host}")

    # Get user input for scan parameters
    folder_name = input("Enter the name for the save folder: ")
    num_images = int(input("Enter the number of images to capture: "))
    step_size = float(input("Enter the Z step size: "))

    # Show a live preview to allow the user to frame the sample
    preview_live(microscope)

    # Execute the Z-stack scan
    z_stack_scan(microscope, base_uri, folder_name, num_images, step_size)


if __name__ == "__main__":
    main()
