import re

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    caption = caption.strip()
    return caption

data = {}

with open("/kaggle/input/datasets/adityajn105/flickr30k/captions.txt", 
          "r", encoding="utf-8") as f:
    
    next(f)  # skip header if exists
    
    for line in f:
        
        line = line.strip()
        image, caption = line.split(",", 1)
        
        caption = clean_caption(caption)
        caption = "<start> " + caption + " <end>"

        
        if image not in data:
            data[image] = []
            
        
        data[image].append(caption)

print(len(data))        # number of images
print(data["1000092795.jpg"])
