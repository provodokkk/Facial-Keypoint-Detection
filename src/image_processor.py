import os
import cv2
import csv
import shutil
import numpy as np

from typing import List, Tuple, Sequence
from tkinter import *
from tkinter.font import Font

from PIL import Image, ImageTk, ImageDraw

from config import IMG_RESIZE, keypoints


IMG_PATH: str = '../img.jpg'
CSV_PATH: str = '../input/test.csv'
SAVE_PATH: str = '../processed_images'
VALIDATION_IMG_PATH: str = '../validation_images'
SCALER: int = 4


def get_file_name_without_extension(full_file_name: str) -> str:
    """Extracts and returns the file name without its extension."""
    return os.path.splitext(full_file_name)[0]


class FaceProcessor:
    """
    A class for processing images with people's faces.

    Attributes:
        save_path: The path where processed images are saved.
        face_cascade_path: Path to the Haar cascade file for face detection.
        face_cascade: The Haar cascade classifier loaded from face_cascade_path.
    """
    def __init__(self, save_path: str = SAVE_PATH):
        self.save_path = save_path
        self.face_cascade_path = '../haarcascade_frontalface_alt2.xml'
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

    def get_cropped_faces(self, img_path: str) -> List[np.ndarray]:
        """
        Detects faces in an image, processes them, and saves the processed images.

        Args:
            img_path: The path to the image file from which faces will be cropped.

        Returns:
            A list of flattened lists representing the grayscale cropped faces.
        """

        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)

        detected_faces = self.detect_faces(img)

        resized_faces_1_channel = self.get_processed_faces(img, detected_faces, three_channels=False)
        resized_faces_3_channels = self.get_processed_faces(img, detected_faces, three_channels=True)

        self.save_faces(img_name, resized_faces_3_channels)

        return [face.flatten().tolist() for face in resized_faces_1_channel]

    def detect_faces(self, img: np.ndarray) -> Sequence[Sequence[int]]:
        """
        Detects faces in an image using Haar cascade classifier.

        Args:
            img: The image in which to detect faces.

        Returns:
            A list of coordinates of the detected faces.
        """

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 4)

        return detected_faces

    def get_processed_faces(self, img: np.ndarray, detected_faces: Sequence[Sequence[int]],
                            three_channels: bool = False) -> List[np.ndarray]:
        """
        Processes detected faces in an image, resizing them.

        Args:
            img: The original image.
            detected_faces: Coordinates of the detected faces.
            three_channels: Flag to return processed faces in 3-channel BGR format. Defaults to False.

        Returns:
            A list of processed face images.
        """

        # To make the picture gray, but the keypoints are colored
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if three_channels:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cropped_faces = [img[y:y + h, x:x + w] for (x, y, w, h) in detected_faces]
        resized_faces = [cv2.resize(face, (96, 96)) for face in cropped_faces]

        return resized_faces

    def save_faces(self, img_name: str, images) -> None:
        """
        Saves processed face images to the specified save path.

        Args:
            img_name: The name of the original image file.
            images: A list of face images to save.
        """

        base_img_name = get_file_name_without_extension(img_name)
        os.makedirs(self.save_path, exist_ok=True)

        for i, image in enumerate(images):
            new_img_path = f'{self.save_path}/{base_img_name}_{i}.jpg'
            cv2.imwrite(new_img_path, image)


class ImageAnnotator:
    """A GUI tool for annotating images with keypoints."""
    def __init__(self, image_file: str):
        self.win = Tk()
        self.canvas = Canvas(self.win, width=IMG_RESIZE * SCALER, height=IMG_RESIZE * SCALER)
        self.canvas.pack()

        # Load and resize the image for display
        img_path = os.path.join(SAVE_PATH, image_file)
        self.img = Image.open(img_path)
        self.img_resized = self.img.resize((IMG_RESIZE * SCALER, IMG_RESIZE * SCALER), Image.NEAREST)

        self.img_tk = ImageTk.PhotoImage(self.img_resized)
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)

        # A label for instructions
        self.kp_index = 0
        font = Font(family='San Francisco', size=16, weight='bold')
        self.status_label = Label(self.win, text=keypoints[self.kp_index], font=font, fg='#D5331E')
        self.status_label.pack()

        self.image_file = image_file
        self.keypoint_coordinates = []
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def on_canvas_click(self, event) -> None:
        """Handles mouse click events on the canvas to annotate the image."""

        # Draw a red point on the image at the clicked location
        img_x, img_y = event.x, event.y
        self.draw_point(img_x, img_y)

        # Update the canvas to reflect the drawing
        self.update_canvas()

        # Save the clicked point
        self.save_point(img_x, img_y)

        # Update label and check if all keypoints have been annotated
        self.kp_index += 1
        if self.kp_index < len(keypoints):
            self.status_label.config(text=keypoints[self.kp_index])
        else:
            # Save the image with keypoints when all points are annotated
            self.save_image()
            self.win.destroy()

    def draw_point(self, x, y, radius: float = 5, color: str = '#67ff58') -> None:
        """Draws a point on the image."""
        draw = ImageDraw.Draw(self.img_resized)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def save_point(self, x, y) -> None:
        """Saves the coordinates of an annotated point."""
        x, y = self.convert_coordinates(x, y)
        self.keypoint_coordinates.extend([x, y])

    def convert_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """Converts the coordinates from canvas scale to original image scale."""
        x = float(x) / SCALER
        y = float(y) / SCALER
        return (x if x < 96.0 else 96.0), (y if y < 96.0 else 96.0)

    def update_canvas(self) -> None:
        """Updates the canvas to reflect the current state of the annotated image."""
        self.img_tk = ImageTk.PhotoImage(self.img_resized)
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)

    def get_image_data(self) -> Tuple[List[float], Image.Image]:
        """Returns the annotated keypoints and the image."""
        return self.keypoint_coordinates, self.img

    def save_image(self) -> None:
        """Saves the annotated image with keypoints."""
        img_name = get_file_name_without_extension(self.image_file)
        annotated_img_path = os.path.join(VALIDATION_IMG_PATH, f'{img_name}_with_keypoints.jpg')
        self.img_resized.save(annotated_img_path)

    def run(self) -> None:
        """Runs the main loop of the GUI."""
        self.win.mainloop()


def get_all_images_in_folder(folder_path: str) -> List[str]:
    """Returns a list of filenames of JPG images in the folder."""
    return [f for f in os.listdir(folder_path) if f.endswith('.jpg')]


def reset_folder(folder_path: str = SAVE_PATH) -> None:
    """
    Deletes the specified folder and all its contents, then recreates an empty folder with the same path.

    Args:
        folder_path: The path to the folder to reset.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Folder {folder_path} has been deleted.")
    except FileNotFoundError:
        print(f"Folder {folder_path} does not exist, creating it.")
    except Exception as e:
        print(f"Failed to delete {folder_path}. Reason: {e}")
        return

    try:
        os.makedirs(folder_path)
        print(f"Folder {folder_path} has been recreated.")
    except Exception as e:
        print(f"Failed to recreate {folder_path}. Reason: {e}")


def append_data_to_csv(csv_path: str, coords: list[float], img: np.ndarray) -> None:
    """
    Appends a list of coordinates and a flattened image array to the end of a specified CSV file.

    Args:
        csv_path: The file path to the CSV file.
        coords: A list of coordinates to be appended.
        img: A NumPy array representing the image.
    """

    img_str = ' '.join(map(str, img))
    space_separated_img = img_str.replace(',', ' ')
    row_data = coords + [space_separated_img]

    with open(csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row_data)


def main() -> None:
    reset_folder(SAVE_PATH)

    face_processor = FaceProcessor()

    cropped_faces = face_processor.get_cropped_faces(IMG_PATH)
    image_files = get_all_images_in_folder(SAVE_PATH)

    for image_file, face in zip(image_files, cropped_faces):
        annotator = ImageAnnotator(image_file)
        annotator.run()

        keypoint_coordinates, img = annotator.get_image_data()
        append_data_to_csv(CSV_PATH, keypoint_coordinates, face)


if __name__ == '__main__':
    main()
