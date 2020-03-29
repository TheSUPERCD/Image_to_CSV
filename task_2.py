import keras_ocr
import cv2
import os

# ============= Use this autocorrector when decoding to plain text files, not tables or stuff using languages other than English =============

from autocorrect import Speller
spell = Speller()


images = [
    cv2.imread('.\\images\\' + filename, 1) for filename in os.listdir('.\\images\\')
    # cv2.imread('excel_test.png')
]

pipeline = keras_ocr.pipeline.Pipeline()
prediction_group = pipeline.recognize(images)

# with open('result_excel_test.txt', 'rb') as f:
#     prediction_group = pickle.load(f)
# for elm in prediction_group:
#     elm.sort(key=lambda x: ((x[1][1,1] + x[1][2,1])/2))

# print(prediction_group[0])


# for (label, box) in prediction_group[0]:
#     cv2.rectangle(images[0], tuple(box[0]), tuple(box[2]), (255,0,0), 1)
# cv2.imwrite('res.png', images[0])


y_tolerance = 20
x_tolerance = 7


for img_prediction, filename in zip(prediction_group, os.listdir('.\\images\\')):
    result = []
    temp_y = (img_prediction[0][1][1,1] + img_prediction[0][1][2,1])/2
    temp_res = []
    for prediction in img_prediction:
        if abs(temp_y - 0.5*(prediction[1][1,1] + prediction[1][2,1])) <= y_tolerance:
            temp_res.append(prediction)
        else:
            result.append(temp_res)
            temp_res = [prediction]
            temp_y = (prediction[1][1,1] + prediction[1][2,1])/2


    for item in result:
        item.sort(key=lambda x: x[1][0,0])


    with open('.\\exports_csv\\' + filename.split('.')[0] + '.csv','w+') as f:
        for line in result:
            temp_x_initial = line[0][1][0,0]
            temp_x_final = line[0][1][1,0]
            f.write(line[0][0])
            for word in line[1:]:
                if abs(temp_x_final - word[1][0,0]) <= 2:
                    f.write('')
                    temp_x_initial = word[1][0,0]
                    temp_x_final = word[1][1,0]
                elif abs(temp_x_final - word[1][0,0]) <= x_tolerance or abs(temp_x_initial - word[1][0,0]) <= x_tolerance:
                    f.write(' ')
                    temp_x_initial = word[1][0,0]
                    temp_x_final = word[1][1,0]
                else:
                    temp_x_initial = word[1][0,0]
                    temp_x_final = word[1][1,0]
                    f.write(',')
                f.write(word[0])
            f.write('\n')
