from PIL import Image
import os

split = ['train', 'val']
lr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
# regularization_scaling_factors = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
totalfolds = [2,3,4,5]
reg = ['No']
# Create the frames
def make_gif(split, lr, totalfolds, regularization, save_name):
    frames = []
    imgs = [filename for filename in os.listdir('plots') if filename.startswith(f"split_{split}_RMSE_lr={lr}_totalFolds_{totalfolds}") and f"reg={regularization}" in filename]

    for i in imgs:
        new_frame = Image.open(f"plots/{i}")
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save(f'gifs/{save_name}.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)

for i in split:
    for j in lr:
        for k in totalfolds:
            for l in reg:
                make_gif(i, j, k, l, f"split_{i}_RMSE_lr={j}_totalFolds_{k}_reg={l}_regScaling=No")

