

from mmseg.apis import init_model, inference_model, show_result_pyplot
import rasterio
from PIL import Image
import cv2
import os
import numpy as np
import glob
import shutil
from fastapi import FastAPI
import uvicorn
from osgeo import gdal, osr, ogr
from rasterio.windows import Window

app = FastAPI()

def tif_to_png(input_file, output_png_file):

    src_ds = gdal.Open(input_file)
    if src_ds is None:
        print('Не удалось открыть исходный файл:', input_file)
        exit(1)

    bands = src_ds.RasterCount
    # Получаем информацию о геотрансформации (геокодировании)
    gt = src_ds.GetGeoTransform()
    data = src_ds.ReadAsArray()
    # data = (data * 65535).astype(np.uint16)
    scaled_data = (data * 255).astype(np.uint8)
    scaled_data = 255 - scaled_data
    # scaled_data = np.fliplr(scaled_data)
    image = Image.fromarray(scaled_data)
    image.save(output_png_file)

    return output_png_file


def apply_sliding_window(image_path, window_size, stride_x, stride_y, output_folder):
    # Загружаем изображение
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    output_filename = os.path.basename(image_path)[:-4]
    
    # Проходим по изображению с помощью скользящего окна
    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            # Проверяем, не выходит ли окно за пределы изображения
            if x + window_size[0] > width:
                x = width - window_size[0]
            if y + window_size[1] > height:
                y = height - window_size[1]

            # Вырезаем кусок изображения
            window = img[y:y+window_size[1], x:x+window_size[0]]
            crop_filename = f'{output_folder}/{output_filename}_{x}_{y}.png'
            cv2.imwrite(crop_filename, window)

            # Выходим из цикла, если достигли нижнего правого угла изображения
            if y + window_size[1] >= height and x + window_size[0] >= width:
                break

    
    return output_folder

def predict(crop_images_path, out_predicted_folder):

    config_path = 'configs/mask2former_swin-t_8xb2-90k_cityscapes-512x1024_water.py'
    checkpoint_path = 'configs/iter_90000.pth'

    model = init_model(config_path, checkpoint_path)

    filenames = glob.glob(f'{crop_images_path}/*.png')
    for filename in filenames:
        print(filename)
        img_path = filename
        result = inference_model(model, img_path)
        np.save(f'{out_predicted_folder}/{os.path.basename(filename)[:-3]}npy', result.pred_sem_seg.data.cpu().numpy(), allow_pickle=True, fix_imports=True)
        # vis_iamge = show_result_pyplot(model, img_path, result,  show=False,  out_file=f'data/landsat_test_images/images_320_320/{dir}/predictions_{model_name}/{os.path.basename(filename)}', save_dir='./') #draw_pred=False, draw_gt=False,  

    return out_predicted_folder



def combine_mask_images(input_folder, output_path, src_folder):
    # Получаем список всех файлов в папке input_folder
    mask_files = os.listdir(input_folder)

    
    # H, W = first_mask.shape
    # print(output_path)
    src_img = cv2.imread(f'{src_folder}/{os.path.basename(output_path)[:-4]}.png')
    H, W = src_img.shape[:2]
    # Создаем пустую маску, куда будем объединять другие маски
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Проходимся по всем файлам масок
    for mask_file in mask_files:
        # Проверяем, что файл - маска
        if mask_file.endswith('.npy'):
            # Загружаем маску
            mask = np.load(f'{input_folder}/{mask_file}')
            mask = mask[0]
            # Извлекаем координаты x и y из названия файла
            x, y = map(int, mask_file[:-4].split('_')[-2:])
            

            try:
            # Добавляем изображение в общую матрицу по соответствующим координатам
                combined_mask[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
            except:
                if x+mask.shape[1] > combined_mask.shape[1] and y + mask.shape[0] > combined_mask.shape[0]:
                    combined_mask[combined_mask.shape[0]-mask.shape[0]:combined_mask.shape[0], combined_mask.shape[1]-mask.shape[1]:combined_mask.shape[1]] = mask
                    
                elif x + mask.shape[0] > combined_mask.shape[1]:
                    combined_mask[y:y+mask.shape[0], combined_mask.shape[1]-mask.shape[1]:combined_mask.shape[1]] = mask 
                else:
                    combined_mask[combined_mask.shape[0]-mask.shape[0]:combined_mask.shape[0], x:x+mask.shape[1]] = mask


    np.save(output_path, combined_mask, allow_pickle=True, fix_imports=True)
    return output_path

def save_tif(input_path, tiff_path, output_filename):
    data = np.load(input_path)
    with rasterio.open(tiff_path) as src:
        # Чтение массива данных
        data_VV = src.read(1)
        transform = src.transform 
    # Указываем параметры для создания GeoTIFF файла
    
    count = 1  # Количество каналов
    height, width = data.shape  # Размеры массива
    dtype = data.dtype  # Тип данных массива

    # Открываем файл для записи с использованием контекстного менеджера
    with rasterio.open(output_filename, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype, transform=transform) as dst:
        # Записываем массив в файл
        dst.write(data, indexes=1)
    
    return output_filename



def aggregate_to_km_resolution(src_path, dst_path):
    with rasterio.open(src_path) as src:
        # Получаем метаданные исходного файла
        meta = src.meta.copy()
        scale_factor = 100  # масштабный фактор для изменения разрешения (100 пикселей = 1 км)

        # Меняем размер изображения и пересчитываем разрешение
        meta.update({
            'height': src.height // scale_factor,
            'width': src.width // scale_factor,
            'transform': src.transform * src.transform.scale(scale_factor, scale_factor)
        })

        # Оставляем количество каналов (band count) таким же, как в исходном изображении
        meta.update(count=1)

        with rasterio.open(dst_path, 'w', **meta) as dst:
            # Итерируем по новым ячейкам 1x1 км
            for i in range(0, src.height, scale_factor):
                for j in range(0, src.width, scale_factor):
                    # Вычисляем фактические размеры окна, особенно на краях изображения
                    win_height = min(scale_factor, src.height - i)
                    win_width = min(scale_factor, src.width - j)

                    # Извлекаем блок данных (win_height x win_width)
                    window = Window(j, i, win_width, win_height)
                    block = src.read(1, window=window)

                    # Подсчет количества пикселей с значением 1
                    num_ones = np.sum(block == 1)

                    # Вычисление площади "1" (в процентах от общей площади блока)
                    total_pixels = win_height * win_width
                    area_percentage = num_ones * 100 * 0.0002
                    # area_percentage = (num_ones / total_pixels) * 100


                    # Создаем массив правильной формы для записи
                    aggregated_block = np.full((1, 1), area_percentage, dtype=src.meta['dtype'])

                    # Записываем результат в новый файл (индекс канала = 1)
                    if (i // scale_factor) < dst.height and (j // scale_factor) < dst.width:
                        dst.write(aggregated_block, window=Window(j // scale_factor, i // scale_factor, 1, 1), indexes=1)
    
    return dst_path

def tiff_to_xyz(filepath):
    ds = gdal.Open(filepath)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    # Get geotransform and calculate the coordinates
    geotransform = ds.GetGeoTransform()
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    # Create arrays for X and Y coordinates
    rows, cols = data.shape
    x = np.arange(cols) * pixel_width + origin_x
    y = np.arange(rows) * pixel_height + origin_y
    # Create a meshgrid for coordinates
    xx, yy = np.meshgrid(x, y)
    # Flatten the arrays
    xyz = np.column_stack((xx.flatten(), yy.flatten(), data.flatten()))
    # Remove NaN and Zero values (if any)
    xyz = xyz[~np.isnan(xyz).any(axis=1)][xyz[:, 2]>0]

    return xyz

def raster_to_vector(input_raster, output_geojson):
    """
    Преобразует одноканальный растровый файл в векторный формат GeoJSON с полигонализацией классов.
    
    Parameters:
    input_raster (str): Путь к входному растровому файлу (GeoTIFF).
    output_geojson (str): Путь к выходному векторному файлу (GeoJSON).
    """
    # Открываем растровый файл
    src_ds = gdal.Open(input_raster)
    if src_ds is None:
        raise FileNotFoundError(f"Не удалось открыть растровый файл: {input_raster}")
    
    band = src_ds.GetRasterBand(1)  # Предполагается, что у вас один канал
    
    # Создаём векторный файл GeoJSON
    driver = ogr.GetDriverByName("GeoJSON")
    out_ds = driver.CreateDataSource(output_geojson)
    if out_ds is None:
        raise RuntimeError(f"Не удалось создать векторный файл: {output_geojson}")
    
    # Создаём слой для полигонов
    out_layer = out_ds.CreateLayer("polygonized", geom_type=ogr.wkbPolygon)
    
    # Добавляем поле для значений класса
    field = ogr.FieldDefn("class", ogr.OFTInteger)
    out_layer.CreateField(field)
    
    # Полигонализация растра
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    
    # Закрываем файлы
    src_ds = None
    out_ds = None
    
    print(f"Полигональный векторный файл создан: {output_geojson}")
    return output_geojson

from datetime import datetime
import re

def extract_timestamp_from_filename(filename):
    """
    Извлекает дату и время из названия файла в формате S1A_YYYYMMDDTHHMMSS_1km.tif и преобразует в объект datetime.
    
    Parameters:
    filename (str): Название файла.
    
    Returns:
    datetime: Объект datetime с извлечённой датой и временем.
    """
    # Используем регулярное выражение для извлечения части с датой и временем
    match = re.search(r'_(\d{8}T\d{6})_', filename)
    if match:
        timestamp_str = match.group(1)  # Извлекаем подстроку с датой и временем, например, "20231231T143759"
        # Преобразуем строку в объект datetime
        timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError("Не удалось найти дату и время в названии файла")

def delete_files_and_folders(folder_path, file_extensions=None):
    """
    Удаляет все файлы и подпапки в указанной папке. 
    Если указаны расширения, удаляются только файлы с этими расширениями, остальные файлы и папки остаются.
    
    :param folder_path: Путь к папке, из которой удаляются файлы и подпапки.
    :param file_extensions: Расширение или список расширений для удаления. Если None — удаляются все файлы и подпапки.
    """
    # Удаляем все файлы и папки, если расширения не указаны
    if file_extensions is None:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f'Файл {item_path} был удален.')
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f'Папка {item_path} была удалена.')
            except Exception as e:
                print(f'Не удалось удалить {item_path}. Ошибка: {e}')
    else:
        # Если передано одно расширение, превращаем его в список
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        # Перебираем файлы и подпапки
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                # Если это файл с нужным расширением, удаляем его
                if os.path.isfile(item_path) and any(item.endswith(ext) for ext in file_extensions):
                    os.remove(item_path)
                    print(f'Файл {item_path} был удален.')
                # Если это подпапка, удаляем ее вместе с содержимым
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f'Папка {item_path} была удалена.')
            except Exception as e:
                print(f'Не удалось удалить {item_path}. Ошибка: {e}')

def process(src_image_name, SRC_IMAGES_FOLDER = 'src_images', CROP_IMAGES_FOLDER = 'crop_images', PREDICTED_CROP_IMAGES_FOLDER = 'predicted_crop_images', PREDICTED_IMAGES_FOLDER = 'predicted_images'):

    # Convert rastr 2 rgb.
    os.makedirs(SRC_IMAGES_FOLDER, exist_ok=True)
    output_png_path = os.path.join(SRC_IMAGES_FOLDER, f'{os.path.basename(src_image_name)[:-4]}.png')
    result_src_image = tif_to_png(src_image_name, output_png_path)

    # Crop src image to fragments(320x320)
    window_size = (1250, 650)
    stride_x = 1250
    stride_y = 650
    output_crop_images_folder = os.path.basename(result_src_image)[:-4]
    os.makedirs(CROP_IMAGES_FOLDER, exist_ok=True)
    output_crop_images_folder = os.path.join(CROP_IMAGES_FOLDER, output_crop_images_folder)
    os.makedirs(output_crop_images_folder, exist_ok=True)
    result_crop_images = apply_sliding_window(result_src_image, window_size, stride_x, stride_y, output_crop_images_folder)
    
    # Predict images.
    os.makedirs(PREDICTED_CROP_IMAGES_FOLDER, exist_ok=True)
    output_predicted_images_folder = os.path.join(PREDICTED_CROP_IMAGES_FOLDER, os.path.basename(result_crop_images))
    os.makedirs(output_predicted_images_folder, exist_ok=True)
    result_predict = predict(result_crop_images, output_predicted_images_folder)


    # Combine predicted images.
    os.makedirs(PREDICTED_IMAGES_FOLDER, exist_ok=True)
    output_combine_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.npy')
    result_combine = combine_mask_images(result_predict, output_combine_path, SRC_IMAGES_FOLDER)

    # Convert rgb 2 rast.
    output_tif_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.tif')
    result_tif_file = save_tif(result_combine, src_image_name, output_tif_path)

    # Rasterization to 1 km.
    output_tif_path_1km = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}_1km.tif')
    result_rasterization_1km = aggregate_to_km_resolution(result_tif_file, output_tif_path_1km)
    geojson_1km = raster_to_vector(result_rasterization_1km, f"{result_rasterization_1km[:-4]}.geojson")
    time_1km = extract_timestamp_from_filename(result_rasterization_1km)
    xyz = tiff_to_xyz(result_rasterization_1km)
    result_xyz  = list([list(x) for x in xyz])

    # Clear folders
    delete_files_and_folders(SRC_IMAGES_FOLDER)
    delete_files_and_folders(CROP_IMAGES_FOLDER)
    delete_files_and_folders(PREDICTED_CROP_IMAGES_FOLDER)
    delete_files_and_folders(PREDICTED_IMAGES_FOLDER, '.npy')


    return {'xyz': result_xyz, 'time':time_1km, 'geojson': geojson_1km}

@app.post("/segment")
def read_root(file_name: str):
    result_cl_file = process(file_name)
    return result_cl_file



# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
# uvicorn main:app --host 172.20.107.6 --port 5544
