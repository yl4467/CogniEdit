from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import os
import numpy as np
import json
import cv2
from multiprocessing import Pool, cpu_count
# Paths

def process_image(img_id, coco, dataDir, output_dir):
    img_data = coco.loadImgs(img_id)[0]
    try:
        image_path = f'{dataDir}/train2017/{img_data["file_name"]}'
        image = io.imread(image_path)
    except:
        return None  # Skip failed images

    original_save_path = f'{output_dir}/original/{img_data["file_name"]}'
    masked_save_path = f'{output_dir}/masked/{img_data["file_name"]}'
    
    # Save original image
    original_image = cv2.imread(image_path)
    cv2.imwrite(original_save_path, original_image)
    
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)
    
    try:
        largest_ann = max(annotations, key=lambda x: x['area'])
    except:
        if len(annotations) == 0:
            return None
        else:
            largest_ann = annotations[0]
    
    # Process annotations
    ann = largest_ann
    coco.showAnns([ann])
    bbox = ann['bbox']
    position = get_object_position(bbox, img_data['width'], img_data['height'])
    
    # Save masked image
    output_path = f'{output_dir}/masked/{img_data["file_name"]}'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=180)
    plt.close()
    
    cat_name = coco.loadCats(ann['category_id'])[0]['name']
    return {
        'image_id': img_id,
        'image_file': img_data['file_name'],
        'object_name': cat_name,
        'original_path': original_save_path,
        'masked_path': masked_save_path,
        'bbox': ann['bbox'],
        'area': ann['area'],
        'position': position,
    }

def process_chunk(args):
    chunk, chunk_idx, coco, dataDir, output_dir = args
    metadata = []
    processed_count = 0
    
    for img_id in chunk:
        result = process_image(img_id, coco, dataDir, output_dir)
        if result is not None:
            metadata.append(result)
            processed_count += 1
            print(f'Chunk {chunk_idx}: Processed {processed_count}/{len(chunk)} - {result["image_file"]}')
    
    # Save metadata for this chunk
    output_file = f'{output_dir}/metadata_{chunk_idx}.json'
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return len(metadata)

def get_object_position(bbox, image_width, image_height):
    """
    Classify object position based on bbox [x,y,w,h] and image dimensions.
    Returns: List of position labels (e.g., ['left', 'top'])
    """
    x, y, w, h = bbox
    center_x = x + w/2
    center_y = y + h/2
    
    positions = []
    
    # Horizontal position
    if center_x < image_width * 0.33:
        positions.append("left")
    elif center_x > image_width * 0.66:
        positions.append("right")
    else:
        positions.append("center_horizontal")
    
    # Vertical position
    if center_y < image_height * 0.33:
        positions.append("top")
    elif center_y > image_height * 0.66:
        positions.append("bottom")
    else:
        positions.append("center_vertical")
    
    # Special case: centered
    if "center_horizontal" in positions and "center_vertical" in positions:
        positions = ["center"]
    
    return positions

dataDir = '/home/'
annFile = f'{dataDir}/annotations/instances_train2017.json'
output_dir = '/Pos_Data'  # Folder to save masked images
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/original', exist_ok=True)
os.makedirs(f'{output_dir}/masked', exist_ok=True)
# Initialize COCO API
coco = COCO(annFile)


cat_ids = coco.getCatIds()  # All category IDs
img_ids = coco.getImgIds()  # All image IDs

chunk_size = 20000
chunks = [img_ids[i:i + chunk_size] for i in range(0, len(img_ids), chunk_size)]

# Prepare arguments for each chunk
args_list = [(chunk, idx + 1, coco, dataDir, output_dir) for idx, chunk in enumerate(chunks)]

# Use multiprocessing (one process per chunk)
num_processes = min(cpu_count(), len(chunks))
with Pool(num_processes) as pool:
    results = pool.map(process_chunk, args_list)

print(f"Processing complete. Total images processed: {sum(results)}")


'''
metadata = []  # Stores object info for JSON
count=0
count_1=0
for img_id in img_ids:  # Process first 10 images (adjust as needed)
    img_data = coco.loadImgs(img_id)[0]
    try:
        image_path = f'{dataDir}/train2017/{img_data["file_name"]}'
        image = io.imread(image_path)
        count_1 = count_1 + 1
    except:
        count=count+1
        continue
    

    original_save_path = f'{output_dir}/original/{img_data["file_name"]}'
    masked_save_path = f'{output_dir}/masked/{img_data["file_name"]}'
    
    # Save original image
    original_image = cv2.imread(image_path)
    cv2.imwrite(original_save_path, original_image)
    
    overlay = image.copy()
    output = image.copy()
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)
    
    try:
        largest_ann = max(annotations, key=lambda x: x['area'])
    except:
        if len(annotations)==0:
            continue
        else:
            largest_ann = annotations[0]
    

    # Extract metadata
    #for ann in annotations[:1]:
    ann = largest_ann
    #print(ann)
    coco.showAnns([ann])
    bbox = ann['bbox']
    position = get_object_position(bbox, img_data['width'], img_data['height'])
    #print(f"Object {coco.loadCats(ann['category_id'])[0]['name']} is at: {position}")
    # Save the figure (matches coco.showAnns style)
    output_path = f'{output_dir}/masked/{img_data["file_name"]}'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=180)

    cat_name = coco.loadCats(ann['category_id'])[0]['name']
    metadata.append({
        'image_id': img_id,
        'image_file': img_data['file_name'],
        'object_name': cat_name,
        'original_path': original_save_path,
        'masked_path': masked_save_path,
        'bbox': ann['bbox'],
        'area': ann['area'],
        'position': position,
    })
    print(count_1,'/',len(img_ids))
    print(f"Processed: {img_data['file_name']}")

# Save metadata to JSON
with open(f'{output_dir}/metadata_1.json', 'w') as f:
    json.dump(metadata, f, indent=4)
    
print(count)  
   
'''

'''
# Get all category IDs
catIds = coco.getCatIds()
categories = coco.loadCats(catIds)
print("COCO Categories:", [cat['name'] for cat in categories])

# Load an image with a specific category (e.g., 'person')
imgIds = coco.getImgIds(catIds=coco.getCatIds(['person']))
img = coco.loadImgs(imgIds[0])[0]

print(img)

# Load and display the image
I = io.imread(f'{dataDir}/val2017/{img["file_name"]}')
plt.imshow(I)
plt.axis('off')

# Load annotations
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)

coco.showAnns(anns)  # Draws bounding boxes & masks
plt.savefig('tmp.png')
'''