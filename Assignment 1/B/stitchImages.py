import sys
from PIL import Image
import os

def buildLayout(number_of_imgs_per_row, number_of_imgs_per_col):
    # Create the layout
    images = [Image.open(f"plots/{x}") for x in os.listdir('plots')]
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (max_width*number_of_imgs_per_row, max_height*number_of_imgs_per_col))

    x_offset = 0
    y_offset = 0
    images_in_row = 0
    for im in images:
        print(x_offset, y_offset)
        new_im.paste(im, (x_offset, y_offset))
        images_in_row += 1
        x_offset += im.size[0]
        if images_in_row == number_of_imgs_per_row:
            images_in_row += 0
            y_offset += max_height
            x_offset = 0
        
    
    new_im.save('stitched.png')

images = [Image.open(f"plots/{x}") for x in os.listdir('plots')]
buildLayout(int(len(images)/2), 2)


# new_im = Image.new('RGB', (total_width, max_height))

# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]

# new_im.save('test.jpg')