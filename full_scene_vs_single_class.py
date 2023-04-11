import os
import shutil
import json
from tqdm import tqdm
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

def single_cat_dataset(cat_id, coco_gt_fp, image_fp, new_exp_dir = False):
  '''
  Creates a new coco experiment folder with only the annotations and images 
  relevant to a specific category/class. If no new directory is passed,
  one will be generated.
  '''
  with open(coco_gt_fp, 'r') as f:
      content = json.load(f)
  
  anns = content['annotations']
  ims = content['images']
  cats = content['categories']
  
  ### pull out annotations only of the chosen class
  new_anns = []
  for a in tqdm(anns, desc='Processing Annotations'):
    if a['category_id'] == cat_id:
      new_anns.append(a)
  
  ### Exit the process if there aren't 
  if len(new_anns) < 1:
      print('There are no annotations of this type in the dataset. Try another category.')
      return
  
  content['annotations'] = new_anns

  ### update the categories section
  for c in cats:
    if c['id'] == cat_id:
        cat_name = c['name'].replace(' ', '-')
        new_cats = [c]
  content['categories'] = new_cats

  ### create the updated experimental folder
  if not new_exp_dir:
      new_exp_dir = '/'.join(coco_gt_fp.split('/')[:-2]) + '/'+ f'{cat_name}_' + coco_gt_fp.split('/')[-2] + '/'
  # If the folder exists, delete it and everything in it
  if not os.path.exists(new_exp_dir):
      os.mkdir(new_exp_dir)
  new_gt_fp = new_exp_dir + coco_gt_fp.split('/')[-1]
  new_image_fp = new_exp_dir + 'images/'
  # ensure annotation file and image directory don't already exist
  if os.path.exists(new_gt_fp):
    os.remove(new_gt_fp)
  if os.path.exists(new_image_fp):
    shutil.rmtree(new_image_fp)
  os.mkdir(new_image_fp)

  ### ensure only images with annotations remain in the dataset
  new_ims = []
  for i in tqdm(ims, desc = 'Processing Images'):
      anns_on_im = anns_on_image(i['id'], content)
      if len(anns_on_im) > 0:
          new_ims.append(i)
          im_name = i['file_name']
          src = image_fp + im_name
          dst = new_image_fp + im_name
          shutil.copy2(src, dst)
  content['images'] = new_ims

  with open(new_gt_fp, 'w') as f:
    json.dump(content, f, indent = 3)
  
  return new_gt_fp, new_image_fp

def main(cat_id, ann_fp, img_fp):
    


  print('Generating Single Class Dataset')
  anns_1c, ims_1c = single_cat_dataset(cat_id, ann_fp, img_fp)

  ### open the existing files ###
  with open(ann_fp, 'r') as f:
    content = json.load(f)

  with open(anns_1c, 'r') as f:
    content_1c = json.load(f)

  # shuffle the images used in the single class experiment
  ims_options = content_1c['images']
  random.shuffle(ims_options)

  # the number of annotations we would like to have in our full scene dataset
  target_anns = len(content_1c['annotations'])
  
  ### Make the new experimental directory
  exp_dir_mc = '/'.join(anns_1c.split('/')[:-2]) + '/'+ 'Full-Scene_' + anns_1c.split('/')[-2] + '/'
  if not os.path.exists(exp_dir_mc):
    os.mkdir(exp_dir_mc)

  ims_mc_fp = exp_dir_mc + 'images/'
  gt_mc_fp = exp_dir_mc + anns_1c.split('/')[-1]
  if os.path.exists(gt_mc_fp):
    os.remove(gt_mc_fp)
  if os.path.exists(ims_mc_fp):
    shutil.rmtree(ims_mc_fp)
  os.mkdir(ims_mc_fp)

  ### Create multiclass annotation and image contents ###
  # loop at random through images with at least one of the target class and get 
  # all the catgeories on them
  anns_mc = []
  ims_mc = []
  im_index = 0
  print('Generating Comparable Full Scene Dataset')
  while len(anns_mc) < target_anns:

      add_im = ims_options[im_index]
      im_src = ims_1c + add_im['file_name']
      im_dst = ims_mc_fp + add_im['file_name']
      shutil.copy2(im_src, im_dst)
      ims_mc.append(add_im)
      im_id = add_im['id']
      anns_add = anns_on_image(im_id, content)
      anns_mc.extend(anns_add)
      im_index += 1

  print(f'\nAnnotations in the single class dataset: {target_anns}')
  print('Annotations in the full scene comparison dataset: ',len(anns_mc))
  print('Images in the single class dataset: ', len(ims_options))
  print('Images in the full scene comparison dataset: ',len(ims_mc))

  content['annotations'] = anns_mc
  content['images'] = ims_mc

  with open(gt_mc_fp, 'w') as f:
    json.dump(content, f, indent=3)
  return 

if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-cat_id", "--cat_id", help = "int, COCO category id of the class you would like to focus on for this experiment")
    parser.add_argument("-ann_fp", "--ann_fp", help = "str, File path to coco annotations")
    parser.add_argument("-img_fp", "--img_fp", help = "str, File path to images for the annotations", required = False)
    
    # Read arguments from command line
    args = parser.parse_args()
    
    main(cat_id = int(args.cat_id), ann_fp = args.ann_fp, img_fp = args.img_fp)