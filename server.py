import os
import numpy as np
import cv2
import laspy
import pyvista as pv
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from scipy.interpolate import griddata
import skimage.measure

app = Flask(__name__)
@app.after_request
def add_embed_headers(resp):
    resp.headers['Content-Security-Policy'] = "frame-ancestors https://sites.google.com https://*.google.com"
    resp.headers.pop('X-Frame-Options', None)
    return resp
OUTPUT_SIZE = (800, 600)
ROAD_THICKNESS = 5
RIVER_COLOR = (255, 0, 0)
STREAM_COLOR = (127, 127, 127)
RAVINE_COLOR = (0, 75, 150)
TREE_COLORS = {'coniferous': (50, 50, 50), 'deciduous': (100, 100, 100), 'bush': (150, 150, 150)}

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

def generate_realistic_lidar_from_image(image_path, output_path, use_inversion=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
    
    img_resized = cv2.resize(img, OUTPUT_SIZE)
    
    img_enhanced = cv2.equalizeHist(img_resized)
    
    if use_inversion:
        img_enhanced = 255 - img_enhanced
    
    x = np.linspace(0, 1000, OUTPUT_SIZE[0])
    y = np.linspace(0, 1000, OUTPUT_SIZE[1])
    x, y = np.meshgrid(x, y)
    
    z = img_enhanced * 0.3  
    
    points = np.column_stack((
        x.flatten() + np.random.normal(0, 2, x.size),
        y.flatten() + np.random.normal(0, 2, y.size),
        z.flatten() + np.random.normal(0, 1, z.size)
    ))

    # Сохраняем как псевдо-LiDAR файл
    header = laspy.LasHeader(version="1.2", point_format=3)
    header.scales = [0.1, 0.1, 0.1]
    
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(output_path, laz_backend=laspy.compression.LazBackend.Lazrs)

def generate_completely_artificial_lidar(output_path):
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    x, y = np.meshgrid(x, y)
    
    z = 50 * (np.sin(x/50) * np.cos(y/40) + np.exp(-((x-500)**2 + (y-700)**2)/100**2)*30)
    
    points = np.column_stack((
        x.flatten() + np.random.normal(0, 2, x.size),
        y.flatten() + np.random.normal(0, 2, y.size),
        z.flatten() + np.random.normal(0, 1, z.size)
    ))

    header = laspy.LasHeader(version="1.2", point_format=3)
    header.scales = [0.1, 0.1, 0.1]
    
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(output_path, laz_backend=laspy.compression.LazBackend.Lazrs)

def draw_coniferous_tree(img, center, size):
    cv2.line(img, (center[0], center[1]), (center[0], center[1]-size), TREE_COLORS['coniferous'], 1)
    for i in range(3):
        angle = 45 + i*30
        dx = int(size*0.3 * np.cos(np.radians(angle)))
        dy = int(size*0.3 * np.sin(np.radians(angle)))
        cv2.line(img, (center[0]-dx, center[1]-dy), (center[0]+dx, center[1]-dy), TREE_COLORS['coniferous'], 1)

def draw_deciduous_tree(img, center, size):
    radius = size//2
    cv2.circle(img, center, radius, TREE_COLORS['deciduous'], 1)
    for angle in range(0, 360, 60):
        dx = int(radius*0.75 * np.cos(np.radians(angle)))
        dy = int(radius*0.75 * np.sin(np.radians(angle)))
        cv2.line(img, center, (center[0]+dx, center[1]+dy), TREE_COLORS['deciduous'], 1)

def draw_bush(img, center, size):
    points = [
        (center[0], center[1]-size//2),
        (center[0]-size//2, center[1]+size//2),
        (center[0]+size//2, center[1]+size//2),
        center
    ]
    for pt in points:
        cv2.circle(img, pt, 2, TREE_COLORS['bush'], -1)

def create_height_map(lidar_path):
    las = laspy.read(lidar_path)
    points = np.vstack((las.x, las.y, las.z)).T

    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    normalized_z = (points[:, 2] - z_min) / (z_max - z_min)

    grid_x, grid_y = np.mgrid[
        np.min(points[:, 0]):np.max(points[:, 0]):OUTPUT_SIZE[0]*1j,
        np.min(points[:, 1]):np.max(points[:, 1]):OUTPUT_SIZE[1]*1j
    ]

    grid_z = griddata((points[:, 0], points[:, 1]), normalized_z, (grid_x, grid_y), method='cubic', fill_value=0)
    height_map = (grid_z * 255).astype(np.uint8)
    height_map = cv2.flip(height_map, 0)

    result = np.ones((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8) * 255

    for level in range(30, 255, 20):
        contours = skimage.measure.find_contours(height_map, level)
        for contour in contours:
            contour = np.round(contour).astype(np.int32)
            cv2.polylines(result, [contour[:, [1, 0]]], isClosed=False, color=(0, 0, 0), thickness=1)

    grad = cv2.Laplacian(height_map, cv2.CV_64F)
    ravine_mask = (grad < -10)
    result[ravine_mask] = (0, 0, 0)

    cv2.line(result, (100, 500), (700, 500), (0, 0, 0), 2)

    for center in [(150, 150), (250, 200), (350, 300)]:
        cv2.circle(result, center, 5, (0, 0, 0), -1)

    return result

def rotate_points(points):
    
    points = np.dot(points, np.array([[0, 0, 1],
                                      [0, 1, 0],
                                      [-1, 0, 0]]))
    
    points = np.dot(points, np.array([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]]))
    
    points = np.dot(points, np.array([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, -1, 0]]))
    
    points[:, 0] = -points[:, 0]
    return points

def create_3d_model(lidar_path):
    las = laspy.read(lidar_path)
    points = np.vstack((las.x, las.y, las.z)).T

    z_median = np.median(points[:, 2])
    z_std = np.std(points[:, 2])
    mask = np.abs(points[:, 2] - z_median) < 2 * z_std
    points = points[mask]
    points[:, 2] = (points[:, 2] - z_median) * 0.1 
    points = rotate_points(points)

    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()

    model_path = os.path.join("output", "model.obj")
    surf.save(model_path)
    return model_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Изображение обязательно"}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({"error": "Имя файла пустое"}), 400

        filename = secure_filename(image.filename)
        image_path = os.path.join("uploads", filename)
        image.save(image_path)

        lidar_path = os.path.join("uploads", "lidar_data.laz")
        if 'lidar' in request.files and request.files['lidar'].filename != '':
            request.files['lidar'].save(lidar_path)
        else:
            generate_realistic_lidar_from_image(image_path, lidar_path, use_inversion=True)
            
        height_map = create_height_map(lidar_path)
        map_2d_path = os.path.join("output", "height_map.png")
        cv2.imwrite(map_2d_path, height_map)

        model_path = create_3d_model(lidar_path)

        return jsonify({
            "status": "success",
            "2d_map": "/output/height_map.png",
            "3d_model": "/output/model.obj"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/output/<filename>')
def send_file(filename):
    return send_from_directory("output", filename)

@app.route('/output/model.obj')
def send_3d_model():
    return send_from_directory('output', 'model.obj', as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)