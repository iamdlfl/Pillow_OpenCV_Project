import zipfile
import itertools

from PIL import Image, ImageDraw, ImageFont
from kraken import pageseg
import pytesseract
import cv2 as cv
import numpy as np

FONT = ImageFont.truetype('readonly/fanwood-webfont.ttf', size=30)
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('readonly/haarcascade_eye.xml')

def cvt_color(img_list):
    """Use CV to convert img to an image array
    
    :param img_list:
    :return: No return, modifies dictionary of image information in place
    """
    for img in img_list:
        img['gray'] = cv.cvtColor(img['cv'], cv.COLOR_BGR2GRAY)


def adaptive_threshold(img_list):
    """Uses CV adaptiveThreshold() to implement Gaussian Thresholding
    
    :param img_list:
    :return: No return, modifies dictionary of image info
    """
    for img in img_list:
        img['thresh'] = cv.adaptiveThreshold(img['gray'],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 4)
        
        
def get_bounding_boxes(image_array, cascade, size=(80,80), scale=1.3, minN=3):
    """Get bounding boxes for a cascade in a picture
    
    :param image_array: An image array
    :param cascade: A haar cascade
    :param size: A tuple for the minimum size
    :param scale: The scaling to be used
    :param minN: Minimum neighbors
    :return bounding_boxes: An array of [x, y, w, h] values for boxes in which faces are found
    """
    boxes = cascade.detectMultiScale(image_array, scaleFactor=scale, minSize=size, maxSize=(300,300), minNeighbors=minN)
    return boxes


def find_faces(img, 
               cascade1=face_cascade, cascade2=eye_cascade, 
               size1=(50,50), size2=(10,10), 
               scale1=1.31, scale2=1.18, 
               minN1=3, minN2=0):
    """Finds rectangle bounding boxes for faces and eyes and then compares them to increase accuracy
    
    :param img: An img dictionary 
    :param cascade: A haar cascade
    :param size: A tuple for the minimum size
    :param scale: The scaling to be used in get_bounding_boxes()
    :param minN: Minimum neighbors
    :return faces_found: An array of faces found
    """
    faces = get_bounding_boxes(img['gray'], cascade1, size1, scale1, minN1)
    eyes = get_bounding_boxes(img['gray'], cascade2, size2, scale2, minN2)
    face_list = []
    for x, y, w, h in faces:
        x1 = x+w
        y1 = y+h
        for testx, testy, w2, h2 in eyes:
            if x<=testx<=x1 and y<=testy<=y1:
                face_list.append([x, y, w, h])
            else:
                continue
    face_list_s = sorted(face_list)
    new_list = list(x for x,_ in itertools.groupby(face_list_s))
    faces_found = np.asarray(new_list)
    return faces_found


def draw_faces(img, boxes):
    """Draws rectangles around faces in a PIL Image and returns it: for ease in visualizing and testing
    
    :param img: An image dictionary
    :param boxes: Bounding boxes for faces
    :return pil_image: A PIL Image with boxes drawn around the faces
    """
    pil_image = img['image'].copy()
    drawing = ImageDraw.Draw(pil_image)
    for x, y, w, h in boxes:
        drawing.rectangle((x, y, x+w, y+h), outline='white', width=3)
    return pil_image
        

def get_faces(img, boxes):
    """Create a list of cropped images of faces
    
    :param img: An img dictionary 
    :param boxes: A set of bounding boxes to crop out faces with
    :return image_list: A list of PIL.Image objects containing faces 
    """
    image_list = []
    pil_image = img['image'].copy()
    for x, y, w, h in boxes:
        image_list.append(pil_image.crop((x, y, x+w, y+h)))
    return image_list


def make_thumbnails(img_list):
    """Makes images in a list into thumbnails
    
    :param img_list: A list of PIL images
    :return: None - modifies the img_list in place
    """
    for img in img_list:
        img.thumbnail((100, 100))

        
def get_text(img_list):
    """Gets text of each image and adds it as a key to the image info dictionary
    
    :param img_list: A list of image info dictionaries
    :return None: Changes the dictionaries in place
    """
    for image in img_list:
        print(image['title'])
        text = pytesseract.image_to_string(image['gray'])
        image['text'] = text
        
        
def phrase_in(img, search_phrase):
    """Simply tests if a phrase is in the text of a page
    
    :param img: An image info dictionary
    :param search_phrase: A phrase to search for
    :return boolean: True if found, false if not
    """
    if str(search_phrase) in img['text']:
        return True
    else:
        return False
        
    
def search_page(img, search_phrase, faces):
    """Searches a list of images for a phrase and enacts the get_faces function on each image that has the phrase
    
    :param img: An image dictionary to search
    :param search_phrase: A phrase to search for in the 'text' key of each image
    :param faces: Bounding boxes for faces
    :return face_list, results: Returns a tuple with the list of face pictures and a string containing it's name
    """
    if str(search_phrase) in img['text']:
        face_list = get_faces(img, faces)
        results = f'Results found in file {img["title"]}'
        return face_list, results
    
    
def make_contact_sheet(img_list, results):
    """Makes a contact sheet for a list of faces and a result string
    
    :param img_list: A list of faces to be placed into a contact sheet
    :param results: A string containing the title of the original image
    :return contact_sheet: A PIL image containing faces and text
    """
    height = int(len(img_list)/5)
    if len(img_list) % 5 != 0:
        height += 1
    if len(img_list) == 0:
        height += 1
    contact_sheet = Image.new('RGB', (500, (height*100)+50))
    drawing = ImageDraw.Draw(contact_sheet)
    drawing.rectangle([0, 0, 500, 50], outline='white', fill='white')
    drawing.text((5, 5), results, font=FONT, fill='black')
    if len(img_list) == 0:
        drawing.rectangle([0, 50, 500, 150], outline='white', fill='white')
        drawing.text((5, 55), text='But there were no faces in that file!', font=FONT, fill='black')
    x = 0
    y = 50
    for img in img_list:
        contact_sheet.paste(img, (x, y))
        if x + 100 == contact_sheet.width:
            x = 0 
            y += 100
        else:
            x += 100
    return contact_sheet
    
    
def concatenate_images(img1, img2):
    """Concatenates two images together
    
    :param img1: First image
    :param img2: Second image
    :return result: The first image placed on top of the second image
    """
    result = Image.new('RGB', (img1.width, img1.height + img2.height))
    result.paste(img1, (0, 0))
    result.paste(img2, (0, img1.height))
    return result
    

def do_project(img_list, phrase_to_search):
    """Combines many of the above functions to search text, get faces and return a contact sheet
    
    :param img_list: A list of images to be searched
    :param phrase_to_search: A search-phrase
    :return contact_sheet: Returns the fully concatenated contact_sheet
    """
    # Make a copy of a CV gray image
    print("Converting images")
    cvt_color(img_list)
    # Get the text on each page
    print("Retrieving text")
    get_text(img_list)
    print("Getting faces")
    for img in img_list:
        print(img['title'])
        faces = find_faces(img)
        if phrase_in(img, phrase_to_search):
            images, results = search_page(img, phrase_to_search, faces)
            make_thumbnails(images)
            if img == img_list[0]:
                contact_sheet = make_contact_sheet(images, results)
            else:
                temp_sheet = make_contact_sheet(images, results)
                contact_sheet = concatenate_images(contact_sheet, temp_sheet)   
    return contact_sheet
    
# Get images open and start the dictionaries
img_list = []
with zipfile.ZipFile('readonly/small_img.zip') as myzip:
    print("Getting small_img.zip files")
    for info_file in myzip.namelist():
        print(info_file)
        img_list.append({'title': info_file, 
                         'image': Image.open(myzip.extract(info_file)), 
                         'cv': cv.imread(myzip.extract(info_file))})
        
big_img_list = []
with zipfile.ZipFile('readonly/images.zip') as myzip2:
    print("Getting images.zip files")
    for info_file in myzip2.namelist():
        print(info_file)
        big_img_list.append({'title': info_file, 
                             'image': Image.open(myzip2.extract(info_file)), 
                             'cv': cv.imread(myzip2.extract(info_file))})

        
# Search the text and get faces
print("Searching for 'Chris' in the small_img.zip")
chris = do_project(img_list, 'Chris')
display(chris)
print("Searching for 'Mark' in the images.zip")
mark = do_project(big_img_list, 'Mark')
display(mark)



    
