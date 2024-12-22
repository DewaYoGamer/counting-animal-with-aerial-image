import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Global variables
original_img = None
distorted_img = None
corrected_img = None
edges = None
thresholded = None

# Fungsi untuk membuka file gambar
def open_image():
    file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        global original_img
        original_img = cv2.imread(file_path)
        show_image(original_img, "Gambar Asli")  # Tampilkan gambar asli
        # Tombol Lanjutkan aktif setelah gambar dipilih
        btn_continue.config(state=tk.NORMAL, command=lambda: continue_to_next_step('distorted'))
        btn_continue.pack(side="bottom", pady=10)  # Pastikan tombol Lanjutkan muncul

# Fungsi untuk menampilkan gambar pada UI
def show_image(img, process_name):
    img_resized = resize_image(img, 600, 400)  # Sesuaikan ukuran gambar
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Menampilkan gambar pada label
    label_img.config(image=img_tk)
    label_img.image = img_tk
    label_process_name.config(text=process_name)  # Menampilkan nama proses di atas gambar

# Fungsi untuk mengubah ukuran gambar agar sesuai dengan resolusi UI
def resize_image(img, target_width, target_height):
    # Ubah ukuran gambar agar sesuai dengan ukuran target (600x400 untuk gambar)
    height, width = img.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    return cv2.resize(img, (new_width, new_height))

# Fungsi untuk menambahkan distorsi panjang fokus
def add_focal_length_distortion(img):
    global distorted_img
    # Matriks kamera (disimulasikan)
    h, w = img.shape[:2]
    fx = 800  # panjang fokus pada sumbu x (focal length)
    fy = 800  # panjang fokus pada sumbu y
    cx = w / 2  # pusat gambar
    cy = h / 2  # pusat gambar
    
    # Distorsi radial
    k1 = -0.3  # koefisien distorsi radial (barrel distortion)
    k2 = 0.1   # koefisien distorsi radial kedua

    # Matriks kamera dan distorsi
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)  # Hanya distorsi radial
    
    # Menambahkan distorsi pada gambar
    distorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
    
    return distorted_img

# Fungsi untuk koreksi distorsi menggunakan matriks dan koefisien distorsi
def correct_focal_length_distortion(img):
    global corrected_img
    # Matriks kamera (disimulasikan)
    h, w = img.shape[:2]
    fx = 800  # panjang fokus pada sumbu x (focal length)
    fy = 800  # panjang fokus pada sumbu y
    cx = w / 2  # pusat gambar
    cy = h / 2  # pusat gambar
    
    # Distorsi radial
    k1 = -0.3  # koefisien distorsi radial (barrel distortion)
    k2 = 0.1   # koefisien distorsi radial kedua

    # Matriks kamera dan distorsi
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)  # Hanya distorsi radial
    
    # Mengoreksi distorsi pada gambar
    corrected_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
    
    return corrected_img


def preprocess_image(img):
    # Konversi ke ruang warna HSV
    hsv = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2HSV)
    
    # Segmentasi warna (tentukan rentang HSV yang sesuai untuk objek pekat)
    lower_bound = np.array([0, 0, 0])  # Batas bawah HSV (pekat)
    upper_bound = np.array([180, 255, 100])  # Batas atas HSV (pekat)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Aplikasikan mask ke gambar asli
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization untuk meningkatkan kontras
    equalized = cv2.equalizeHist(gray)
    
    # Median Blur untuk mengurangi noise
    blurred = cv2.medianBlur(equalized, 5)
    
    return blurred

# Fungsi untuk deteksi tepi
def edge_detection():
    global edges
    
    # Preprocessing dengan fungsi yang baru
    preprocessed = preprocess_image(corrected_img)
    
    edges = cv2.Canny(preprocessed, 50, 150)
    
    show_image(edges, "Edge Detection")  # Tampilkan hasil deteksi tepi
    btn_continue.config(command=lambda: continue_to_next_step('thresholded'))
    
# Fungsi untuk thresholding
def thresholding():
    global thresholded
    # Gunakan thresholding Otsu
    _, thresholded = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operation untuk menghilangkan noise
    kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    if thresholded is None or thresholded.size == 0:
        print("Thresholding failed. Ensure that edges is not empty.")
        return

    # Pastikan thresholded adalah gambar biner (grayscale)
    if len(thresholded.shape) == 3:  # Jika gambar thresholded masih dalam format warna (CV_8UC3)
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale

    # Temukan kontur
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Buat gambar biner baru untuk menyimpan kontur yang lebih besar
    filtered_thresholded = np.zeros_like(thresholded)
    
    # Filter kontur berdasarkan area
    min_area = 83  # Area minimum untuk kontur yang akan dipertahankan
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_thresholded, [contour], -1, 255, thickness=cv2.FILLED)
    
    thresholded = filtered_thresholded

    show_image(thresholded, "Thresholded Image")  # Tampilkan gambar yang telah diberi threshold
    btn_continue.config(command=lambda: continue_to_next_step('contoured'))

# Fungsi untuk menggambar kontur pada gambar thresholded dan menghitung jumlah hewan
def contour_drawing():
    global thresholded, corrected_img
    
    # Pastikan thresholded adalah gambar biner (grayscale)
    if len(thresholded.shape) == 3:  # Jika gambar thresholded masih dalam format warna (CV_8UC3)
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
    
    # Cek jika thresholded sudah menjadi biner (grayscale)
    if thresholded is None or thresholded.size == 0:
        print("Error: Thresholded image is empty or not valid.")
        return
    
    # Temukan kontur
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    min_area = 500  # Area minimum
    max_area = 5000  # Area maksimum
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Tambahkan filter kebulatan jika perlu
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.5 < circularity < 1.2:  # Rentang kebulatan
                    filtered_contours.append(contour)
    
    # Gambar kontur pada gambar corrected_img
    contoured_img = corrected_img.copy()
    cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)
    
    # Hitung jumlah hewan (kontur)
    num_animals = len(contours)
    print(f"Jumlah hewan yang terdeteksi: {num_animals}")
    
    show_image(contoured_img, "Contoured Image")  # Tampilkan gambar dengan kontur
    
    # Hapus tombol Lanjutkan setelah menghitung jumlah hewan
    btn_continue.pack_forget()

def continue_to_next_step(stage):
    global corrected_img, edges, thresholded

    if stage == 'distorted':
        # Tambahkan distorsi panjang fokus
        distorted_img = add_focal_length_distortion(original_img)
        show_image(distorted_img, "Distorted Image")  # Tampilkan gambar yang terdistorsi
        btn_continue.config(command=lambda: continue_to_next_step('corrected'))

    elif stage == 'corrected':
        # Koreksi distorsi pada gambar yang terdistorsi
        corrected_img = correct_focal_length_distortion(original_img)
        show_image(corrected_img, "Corrected Image")  # Tampilkan gambar yang sudah dikoreksi
        btn_continue.config(command=lambda: continue_to_next_step('edge_detection'))

    elif stage == 'edge_detection':
        # Deteksi tepi
        edge_detection()

    elif stage == 'thresholded':
        # Thresholding
        thresholding()

    elif stage == 'contoured':
        # Gambar kontur pada gambar thresholded
        contour_drawing()

# Membuat UI dengan Tkinter
root = tk.Tk()
root.title("Focal Length Distortion and Correction")
root.geometry("1000x600")  # Mengatur ukuran jendela Tkinter menjadi 1000x600

# Label untuk menampilkan gambar
label_img = tk.Label(root)
label_img.pack(padx=20, pady=20)

label_process_name = tk.Label(root, text="Pilih Gambar", font=("Arial", 12))
label_process_name.pack(pady=10)

# Tombol untuk membuka gambar
btn_open = tk.Button(root, text="Pilih Gambar", command=open_image)
btn_open.pack(side="bottom", pady=10)

# Tombol untuk melanjutkan ke tahap selanjutnya
btn_continue = tk.Button(root, text="Lanjutkan", command=None, state=tk.DISABLED)
btn_continue.pack(side="bottom", pady=10)  # Tampilkan tombol Lanjutkan pada awalnya

# Menjalankan aplikasi UI
root.mainloop()