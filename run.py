import paddle
import csv
from paddleocr import PaddleOCR, draw_ocr

ocr=PaddleOCR(use_angle_cls=True, lang='en',
              rec_model_dir="model/rec/PP-OCRv3_ver1/inference/en_PP-OCRv3_rec",
              det_model_dir="model/det/db_mv3_inference_ver1/inference/db_mv3_inference",
              use_gpu=True)



img_path="data/Type1_ver2/output/test7.jpg"
result = ocr.ocr(img_path, cls=True)

for idx in range(len(result)):
    res = result[idx]
    with open('1.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      for line in res:
          line_str = str(line)
          writer.writerow([idx + 1, line_str])
          print(line)

from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result_7.jpg')
print(txts)

mean_value = sum(scores) / len(scores)
print(mean_value)