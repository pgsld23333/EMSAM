import numpy as np
from torchvision import transforms

def expand_bbox(bbox, H, W, rand_val=100):
    [x1, y1, x2, y2] = bbox
    
    x1 = max(0, x1 - np.random.randint(rand_val))
    y1 = max(0, y1 - np.random.randint(rand_val))
    x2 = min(W-2, x2 + np.random.randint(rand_val))
    y2 = min(H-2, y2 + np.random.randint(rand_val))
    
    return [x1, y1, x2, y2]


# def expand_bbox_normal(bbox, H, W, rand_val=None):
#     [x1_, y1_, x2_, y2_] = bbox
#     dx1_mean = -0.06340208893992781
#     dx1_std = 0.15141646484136576

#     dx2_mean = 0.061553240191869744
#     dx2_std = 0.14384385685129084
    
#     dy1_mean = -0.13626701710380448
#     dy1_std = 0.15301446844414784
    
#     dy2_mean = 0.11543334456707802
#     dy2_std = 0.19530838453439883
#     h, w = y2_ - y1_, x2_ - x1_
#     while True:
#         dx1 = np.random.normal(dx1_mean, dx1_std)
#         dx2 = np.random.normal(dx2_mean, dx2_std)
#         dy1 = np.random.normal(dy1_mean, dy1_std)
#         dy2 = np.random.normal(dy2_mean, dy2_std)
#         # print(dx1, dx2, dy1, dy2)
#         x1 = max(0, int(x1_ + dx1 * w + 0.5))
#         y1 = max(0, int(y1_ + dy1 * h + 0.5))
#         x2 = min(W-2, int(x2_ + dx2 * w + 0.5))
#         y2 = min(H-2, int(y2_ + dy2 * h + 0.5))

#         if x2 - x1 > 0.1 * w and y2 - y1 > 0.1 * h:
#             break

#     return [x1, y1, x2, y2]

def pad_image(image, mask=None, bbox=None):
    H, W, _ = image.shape
    # assert H < W, "Height should be less than width"
    if H < W:
        pad_len = (W - H) // 2
        top_pad = np.zeros((pad_len, W, 3), dtype=np.uint8)
        bottom_pad = np.zeros(((W - H) - pad_len, W, 3), dtype=np.uint8)
        pad_image = np.concatenate([top_pad, image, bottom_pad], axis=0)

        if mask is not None:
            top_pad = np.zeros((pad_len, W), dtype=np.uint8)
            bottom_pad = np.zeros((pad_len, W), dtype=np.uint8)
            pad_mask = np.concatenate([top_pad, mask, bottom_pad], axis=0)
        else:
            pad_mask = None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            y1 += pad_len
            y2 += pad_len
            pad_bbox = [x1, y1, x2, y2]
        else:
            pad_bbox = None
        assert pad_image.shape[0] == pad_image.shape[1] and pad_image.shape[0] == W, "Padding error."
        assert (pad_mask is None) or (pad_mask.shape[0] == pad_mask.shape[1] and pad_mask.shape[0] == W), "Padding error."
    elif H > W:
        pad_len = (H - W) // 2
        left_pad = np.zeros((H, pad_len, 3), dtype=np.uint8)
        right_pad = np.zeros((H, (H - W) - pad_len, 3), dtype=np.uint8)
        pad_image = np.concatenate([left_pad, image, right_pad], axis=1)

        if mask is not None:
            left_pad = np.zeros((H, pad_len), dtype=np.uint8)
            right_pad = np.zeros((H, pad_len), dtype=np.uint8)
            pad_mask = np.concatenate([left_pad, mask, right_pad], axis=1)
        else:
            pad_mask = None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 += pad_len
            x2 += pad_len
            pad_bbox = [x1, y1, x2, y2]
        else:
            pad_bbox = None
        assert pad_image.shape[0] == pad_image.shape[1] and pad_image.shape[0] == H, "Padding error."
        assert (pad_mask is None) or (pad_mask.shape[0] == pad_mask.shape[1] and pad_mask.shape[0] == H), "Padding error."
    else:
        pad_image = image
        pad_mask = mask
        pad_bbox = bbox
    return pad_image, pad_mask, pad_bbox

def direction_augmentation(image, mask, bbox, H, W):    
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
        bbox = [W - bbox[2], bbox[1], W - bbox[0], bbox[3]]
    
    if np.random.rand() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
        bbox = [bbox[0], H - bbox[3], bbox[2], H - bbox[1]]
    
    if np.random.rand() < 0.5:
        image = np.rot90(image)
        mask = np.rot90(mask)
        x1, y1, x2, y2 = bbox
        x1_ = y1
        y1_ = W - x1
        x2_ = y2
        y2_ = W - x2
        bbox = [min(x1_, x2_), min(y1_, y2_), max(x1_, x2_), max(y1_, y2_)]

    if np.random.rand() < 0.5:
        image = np.rot90(image, k=3)
        mask = np.rot90(mask, k=3)
        # x = h - y, y = x
        x1, y1, x2, y2 = bbox
        x1_ = H - y1
        y1_ = x1
        x2_ = H - y2
        y2_ = x2
        bbox = [min(x1_, x2_), min(y1_, y2_), max(x1_, x2_), max(y1_, y2_)]
    
    return image, mask, bbox

def random_crop(image, mask, bbox, scale=(0.1, 1.0)):
    H, W, _ = image.shape
    assert H == W, "Height and width should be same."
    x1, y1, x2, y2 = bbox
    bbox_long_edge = max(x2 - x1, y2 - y1)
    assert int(W * np.sqrt(scale[1])) > bbox_long_edge+2, "Scale is too small."
    crop_W = np.random.randint(
        max(int(W * np.sqrt(scale[0])), bbox_long_edge+2),
        max(int(W * np.sqrt(scale[1])), bbox_long_edge+2)
    )
    try:
        crop_x1 = np.random.randint(max(0, x2+1 - crop_W), min(x1+1, W - crop_W))
        crop_y1 = np.random.randint(max(0, y2+1 - crop_W), min(y1+1, H - crop_W))
    except:
        import pdb; pdb.set_trace()
    crop_x2 = crop_x1 + crop_W
    crop_y2 = crop_y1 + crop_W    
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_bbox = [x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]
    if cropped_bbox[0] < 0 or cropped_bbox[1] < 0 or cropped_bbox[2] >= crop_W or cropped_bbox[3] >= crop_W:
        print("Error: ", cropped_bbox)
        import pdb; pdb.set_trace()
    return cropped_image, cropped_mask, cropped_bbox
