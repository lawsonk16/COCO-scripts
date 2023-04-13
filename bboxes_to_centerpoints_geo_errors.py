import json
import os
from tqdm import tqdm
import numpy as np
import random
import argparse


def anns_on_image(im_id, contents):
    '''
    IN: 
        - im_id: int id for 'id' in 'images' of coco json
        - json_path: path to coco gt json
    OUT:
        - on_image: list of annotations on the given image
    '''
    
    # Pull out annotations
    anns = contents['annotations']
    
    # Create list of anns on this image
    on_image = []
    for a in anns:
        if a['image_id'] == im_id:
            on_image.append(a)
    
    return on_image

def average_bboxes_from_centerpoints(anns_path, avg_img_gsd = None):
    '''
    PURPOSE: After finding average object sizes, and using bounding boxes to add
             centerpoints (all to a coco annotation file), replace bounding boxes 
             using image GSD and average object sizes to grow the centerpoints
    IN:
     - anns_path: str, path to coco annotations file
     - avg_img_gsd: float or int, optional, average image size in dataset, 
                    which will be used as a default if an image doesn't have 
                    a noted GSD. 
    OUT:
     - new_anns_path: str, path to new annotation file
    '''
    
    # open annotation file
    with open(anns_path, 'r') as f:
        content = json.load(f)

    # if necessary, get average gsd
    if avg_img_gsd == None:
        avg_img_gsd = get_average_image_gsd(anns_path)
    
    # pull out key sections of file
    anns = content['annotations']

    # adjust bounding boxes based on centerpoints and object sizes
    new_annotations = []
    for a in tqdm(anns, desc = 'Creating Square Bboxes'):
        new_a = a.copy()
        [x,y] = a['object_center']
        obj_size = get_obj_size_from_id(a['category_id'], content)
        im_gsd = get_im_gsd_from_id(a['image_id'], content)
        if im_gsd != None:
            ob_h_w = int(obj_size/im_gsd)
        else:
            try:
                ob_h_w = int(obj_size/avg_img_gsd)
            except:
                print(f'object size is {obj_size} and the average gsd is {avg_img_gsd}')
        square_bbox = [x - (ob_h_w/2), y - (ob_h_w/2), ob_h_w, ob_h_w]
        new_a['bbox'] = square_bbox
        new_annotations.append(new_a)

    content['annotations'] = new_annotations

    new_anns_path = anns_path.split('.')[0] + '_square.json'

    if os.path.exists(new_anns_path):
        os.remove(new_anns_path)
    
    with open(new_anns_path, 'w') as f:
        json.dump(content, f)

    return new_anns_path

def estimate_category_size(anns_path, write_out = False, matched_files = []):
    '''
    PURPOSE: Get average sizes in meters for each object category in a 
             coco dataset and optionally add them to the file, with the option
             to add the values to multiple files
    IN:
     - anns_path: str, path to coco annotation file
     - write_out: boolean, whether or not to write the values into the coco file
     - matched_files : list of strs, paths to other files to write our average 
                       object sizes to
    OUT:
     - estimates: dict, contains information about each category keyed to its id
    '''

    # open annotation file
    with open(anns_path, 'r') as f:
        content = json.load(f)
    
    # pull out key sections of file
    cats = content['categories']
    anns = content['annotations']

    # create dictionary for storing key info about objct sizes
    estimates = {}
    for c in cats:
        estimates[c['id']] = {'name': c['name'], 'sizes': []}

    # add the size of each indivdual object to the list for that category
    for a in tqdm(anns, desc = "Estimating Category Sizes"):
        bbox = a['bbox']

        # get largest side of object
        size = max(bbox[2:3])
        im_gsd = get_im_gsd_from_id(a['image_id'], content)
        if im_gsd != None:
          size_m = size*im_gsd
          estimates[a['category_id']]['sizes'].append(size_m)
    
    # add average sizes using size lists
    for k,v in estimates.items():
        avg = np.mean(v['sizes'])
        estimates[k]['average'] = avg

    new_cats = []
    if write_out:

        for c in cats:
            new_c = c.copy()
            new_c['average_size'] = estimates[c['id']]['average']
            new_cats.append(new_c)
        content['categories'] = new_cats

        os.remove(anns_path)

        with open(anns_path, 'w') as f:
            json.dump(content, f)
        
        for fp in matched_files:
            with open(fp, 'r') as f:
                f_contents = json.load(f)
            f_contents['categories'] = new_cats
            os.remove(fp)
            with open(fp, 'w') as f:
                json.dump(f_contents, f)
    return estimates


def get_average_image_gsd(anns_path):
    '''
    PURPOSE: Find the average GSD of the images in a coco ground truth file
    IN:
     - anns_path: str, path to coco annotation file
    OUT:
     - avg_img_gsd: float, average gsd of images in dataset
    '''
    with open(anns_path, 'r') as f:
        content = json.load(f)
    images = content['images']

    gsd_vals = []

    for i in tqdm(images, desc = 'Finding Average Image GSD'):
        if 'acquisition_data' in i.keys():
            gsd_val = i['acquisition_data']['GSD'][0]
            if gsd_val != None:
                gsd_vals.append(gsd_val)

    avg_img_gsd = np.average(gsd_vals)

    return avg_img_gsd

def get_im_gsd_from_id(im_id, gt_content):
    '''
    PURPOSE: Get the GSD of an image based on its id in a coco file
    IN:
     - im_id: int, image id for the image in question
     - gt_content: the content from a coco ground truth file
    OUT:
     - pt: either the image's GSD or None if it isn'available
    '''
    images = gt_content['images']

    for i in images:
        if i['id'] == im_id:
            try:
                return i['acquisition_data']['GSD'][0]
            except:
                return None
        

def get_obj_size_from_id(cat_id, gt_content):
    '''
    PURPOSE: Get the average size of an object from a coco file, after 
             estimate_category_size has been run on that annotation set
    IN:
     - cat_id: the integer id of a coco category
     - gt_content: the content from a coco ground truth file
    OUT:
     - pt: Either the object's average size (float) or None if it isn't recorded
    '''
    cats = gt_content['categories']

    for c in cats:
        if c['id'] == cat_id:
            return c['average_size']
    return None



def get_category_id_from_name(cat_name, gt_content):
    '''
    IN: 
      - cat_name: str, category name from coco json
      - gt_content: json contents of coco gt file
    OUT: cat_id: int, category id for that named category
    '''

    for c in gt_content['categories']:
        if c['name'] == cat_name:
            return c['id']
    return None

def make_cat_ids_match(src_anns, match_anns):
    '''
    IN: 
      - src_anns: str, path to the annotations whose category ids will provide
                  the mapping
      - match_anns: str, path to annotations whose categories will be remapped
    OUT: None, the categories will be remapped in place
    PURPOSE: Given two sets of coco annotations whose categories match, 
    ensure that the ids of each category are the same by forcing 
    match_anns categories to match src_anns categories
    '''
    # Open the annotations
    with open(src_anns, 'r') as f:
        src_gt = json.load(f)
    with open(match_anns, 'r') as f:
        match_gt = json.load(f)
    
    # Get the lists of categories
    src_cats = src_gt['categories']
    match_cats = match_gt['categories']

    # Create a mapping from one set of ids to the other
    cat_map = {}
    for c in match_cats:
        cat_map[c['id']] = get_category_id_from_name(c['name'], src_gt)

    # Remap the annotations in match_anns
    new_annotations = []
    for a in match_gt['annotations']:
        new_a = a.copy()
        new_a['category_id'] = cat_map[a['category_id']]
        new_annotations.append(new_a)
    
    match_gt['annotations'] = new_annotations
    match_gt['categories'] = src_cats
    
    # Save out a new file
    os.remove(match_anns)
    with open(match_anns, 'w') as f:
        json.dump(match_gt, f)

    return 

def convert_anns_centerpoint_meters(anns_path, avg_img_gsd, shift_meters = 5, percentage_shift = 100, random_amount = False):
    '''
    PURPOSE: Convert an annotation file with image-oriented bounding boxes to 
             center point annotations instead
    IN:
     - anns_path
     - avg_img_gsd
     - shift_meters = 5
     - percentage_shift = 100
    OUT:
     - new_anns_path: str, path to new annotations
    '''

    # open the annotation file
    with open(anns_path, 'r') as f:
        ann_contents = json.load(f)
        
    # Pull out key section
    images = ann_contents['images']
    
    shift_decimal = (percentage_shift/100)
    split_point = int(len(images)*shift_decimal)
    images_shuffle = random.sample(images, len(images))
    images_shift = images_shuffle[:split_point]
    images_regular = images_shuffle[split_point:]
        
    new_anns = []
    for i in tqdm(images_shift, desc = f'Shifting points on {percentage_shift}% of the images'):
        im_gsd = get_im_gsd_from_id(i['id'], ann_contents)
        if im_gsd:
            if random_amount:
                shift = random.choice(range(1, int(shift_meters/im_gsd)+1))
            else:
                shift = shift_meters/im_gsd  
        else:
            if random_amount:
                shift = random.choice(range(1, int(shift_meters/avg_img_gsd)+1))
            else:
                shift = shift_meters/avg_img_gsd
            
        
        # choose a shift direction for the annotations on this image
        # define options
        vert_opts = ['up', 'down', 'centered']
        hori_opts = ['left', 'right', 'centered']
        
         # randomly select a movement
        v_d = random.choice(vert_opts)
        h_d = random.choice(hori_opts)
        
        anns = anns_on_image(i['id'], ann_contents)
        for a in anns:
        
            new_a = a.copy()
            x1, y1, w, h = a['bbox']
            x_c = x1 + int(w/2)
            y_c = y1 + int(h/2)
            
            # move the point:
            if v_d == 'up':
                y_c += shift
            elif v_d == 'down':
                y_c -= shift
        
            if h_d == 'right':
                x_c += shift
            elif h_d == 'left':
                x_c -= shift
          
            # make sure there are no negatives
            if x_c < 0:
               x_c = 0
            if y_c < 0:
               y_c = 0

            new_a['object_center'] = [x_c, y_c]
            new_anns.append(new_a)
            
    for i in images_regular:
        anns = anns_on_image(i['id'], ann_contents)
        
        for a in anns:
            new_a = a.copy()
            x1, y1, w, h = a['bbox']
            x_c = x1 + int(w/2)
            y_c = y1 + int(h/2)
            new_a['object_center'] = [x_c, y_c]
            new_anns.append(new_a)

    ann_contents['annotations'] = new_anns

    # create and save new annotation file
    new_anns_path = anns_path.split('.')[0] + f'_cp_{shift_meters}_meters_{percentage_shift}_percent.json'

    if os.path.exists(new_anns_path):
        os.remove(new_anns_path)
    
    with open(new_anns_path, 'w') as f:
        json.dump(ann_contents, f)

    return new_anns_path




if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-train_fp", "--train_fp", help = "File path to geococo train annotations")
    parser.add_argument("-shift_meters", "--shift_meters", help = "Int, the number of meters you would like annotations to be shifted, on an image-by-image basis, in meters")
    parser.add_argument("-shift_percent", "--shift_percent", help = "[0-100]The percentage of images you would like to shift by the value in shift_meters", required = False, default = 100)
    parser.add_argument("-avg_gsd", "--avg_gsd", help = "Average image GSD you would like to use", required = False)
    
    # Read arguments from command line
    args = parser.parse_args()
    
    print("shift_percentage", args.shift_percent)
    
    # add size estimates in meters to the object categories
    estimates = estimate_category_size(args.train_fp, True)
    print('Estimated category sizes:')
    for k in estimates.keys():
          name = estimates[k]['name']
          avg = round(estimates[k]['average'], 1)
          print(f'{name}: {avg} meters')
    
    
    shift_m =int(args.shift_meters)
        
    if args.avg_gsd:
        # convert bounding boxes to square boxes around centerpoints based on gsd and 
        # average object size
        # add centerpoints to the annotations
        train_c_cp = convert_anns_centerpoint_meters(args.train_fp, args.avg_gsd, shift_meters = shift_m, percentage_shift = int(args.shift_percent), random_amount = True)
        train_anns_sq = average_bboxes_from_centerpoints(train_c_cp, avg_img_gsd = float(args.avg_gsd))
    else:
        
        # get the average image gsd value
        avg_img_gsd = get_average_image_gsd(args.train_fp)
        print(f'Average Image GSD: {avg_img_gsd}')
        train_c_cp = convert_anns_centerpoint_meters(args.train_fp, avg_img_gsd, shift_meters = shift_m, percentage_shift = int(args.shift_percent), random_amount = True)
        # convert bounding boxes to square boxes around centerpoints based on gsd and 
        # average object size
        train_anns_sq = average_bboxes_from_centerpoints(train_c_cp, avg_img_gsd = avg_img_gsd)
    
    
    
    
    
   
