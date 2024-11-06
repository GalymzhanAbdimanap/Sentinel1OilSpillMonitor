# Oil Spill Detection API

This API allows users to perform segmentation on satellite imagery for detecting oil spills using deep learning models. It processes one-channel TIFF images, crops them into smaller fragments, applies predictions, and combines the results into a single output.

## Endpoints

### POST /segment

This endpoint accepts a request to process an image for oil spill detection.

#### Request Parameters

- **file_name** (string): The name of the source TIFF image file to be processed. The image should be located in the default folder `src_images`.

#### Response

- **xyz** (array): A list of coordinates and values extracted from the processed image.

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/segment" -H "Content-Type: application/json" -d '{"file_name": "your_image.tif"}'
```

#### Example Response

```json
{
  "xyz": [[x1, y1, value1], [x2, y2, value2], ...]
}
```

## Image Processing Workflow

1. **Convert Raster to RGB**: 
   - The input multi-channel TIFF image is processed to extract relevant channels and saved as an RGB PNG file.

2. **Crop Image**:
   - The RGB image is cropped into overlapping fragments of size 320x320 pixels.

3. **Predict Segmentation**:
   - Each cropped image is fed into a pre-trained segmentation model to predict the mask for oil spills.

4. **Combine Predicted Masks**:
   - The predicted masks are combined back into a single mask based on their coordinates in the original image.

5. **Save Output as TIFF**:
   - The combined mask is saved as a GeoTIFF file.
  
6. **Rasterization to 1km**:
   
8. **Get geoposition of oil spill**:

9. **Cleanup**:
   - Temporary files and folders are deleted to keep the workspace clean.

## Setup

### Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Rasterio
- GDAL
- OpenCV
- Pillow
- NumPy
- mmsegmentation (including necessary models and configurations)

### Installation

Install the required packages using pip:

```bash
conda create -n env python=3.8
conda activate env
```

```bash
conda install -c conda-forge gdal
```

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```


```bash
pip install -r requirements.txt
```

```bash
pip install "mmsegmentation>=1.0.0"
```

Download weights from [drive](https://drive.google.com/file/d/1NkMV5pGZ6yasBzVr3guKCvm1KjAbmdvd/view?usp=drive_link) and place them in the appropriate configuration directory.

### Running the API

To run the API server, use the following command:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Replace `main` with the name of your Python file containing the FastAPI app if it's different.

## Directory Structure

- `src_images/`: Folder for source images.
- `crop_images/`: Temporary folder for cropped images.
- `predicted_crop_images/`: Temporary folder for predicted cropped images.
- `predicted_images/`: Folder for saving the final predicted TIFF images.

## License

This project is licensed under the MIT License.
