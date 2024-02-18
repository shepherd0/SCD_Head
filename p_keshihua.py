import json
import shutil
import cv2

def select(json_path, outpath, image_path):
    json_file = open(json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = image_path + "/" + images[i]["file_name"]
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            if annos[j]["image_id"] == im_id:
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
                img_name = outpath + "/" + images[i]["file_name"]
                cv2.imwrite(img_name, img)
                # continue
        print(i)

if __name__ == "__main__":
    json_path = "cervicalData/annotations/instances_val2017.json"#放标注json的地址
    out_path = "runs/detect/json"#结果放的地址
    image_path = "cervicalData/images/val2017"#原图的地址
    select(json_path, out_path, image_path)
