import os
import sys

sys.path.append('./')

import face_recognition  # noqa


if __name__ == "__main__":
    """计算距离评测"""
    img_labels = ['chenglong', 'dongxuan', 'guanzhilin', 'gulinazha', 'gutianle', 'huge', 'jindong', 'jingtian', 'lilianjie', 'liming', 'linjunjie', 'liudehua', 'sunli', 'tongliya', 'yangmi', 'zhangmin', 'zhangxueyou', 'zhoujielun', 'zhourunfa', 'zhouxingchi']
    # 已有图像
    known_img_path = './data/test_data/mingxing/train/'
    known_imgs = list()
    known_img_label_dict = dict()  # 图片序号对应标签
    known_img_id = 0
    for img_label in img_labels:
        temp_img_path = os.path.join(known_img_path, img_label)
        for file in os.listdir(temp_img_path):
            temp_img_file_name = os.path.join(temp_img_path, file)
            known_imgs.append(temp_img_file_name)
            known_img_label_dict[known_img_id] = img_label
            known_img_id += 1
    print('known_imgs len:{},example:{}'.format(len(known_imgs), known_imgs[:5]))
    # 测试图像读取
    test_img_path = './data/test_data/mingxing/test/'
    test_imgs = list()
    for img_label in img_labels:
        temp_img_path = os.path.join(test_img_path, img_label + '.jpg')
        test_imgs.append(temp_img_path)
    print('test_imgs len:{},test_imgs:{}'.format(len(test_imgs), test_imgs[:5]))
    # 已有图像编码
    known_encodings = list()
    for known_img in known_imgs:
        image_to_test = face_recognition.load_image_file(known_img)
        image_to_test_encodings = face_recognition.face_encodings(image_to_test)
        if len(image_to_test_encodings) < 1:
            print('error img:{}'.format(known_img))
        else:
            image_to_test_encoding = image_to_test_encodings[0]
            known_encodings.append(image_to_test_encoding)
    print('已有图像编码完成')
    # 测试图像编码
    test_encodings = list()
    for known_img in test_imgs:
        image_to_test = face_recognition.load_image_file(known_img)
        image_to_test_encodings = face_recognition.face_encodings(image_to_test)
        if len(image_to_test_encodings) < 1:
            print('error img:{}'.format(known_img))
        else:
            image_to_test_encoding = image_to_test_encodings[0]
            test_encodings.append(image_to_test_encoding)
    print('测试图像编码完成')
    # 进行评测
    right_count = 0
    sum_count = 0
    for test_encoding, img_label in zip(test_encodings, img_labels):
        sum_count += 1
        face_distances = face_recognition.face_distance(known_encodings, test_encoding)
        face_distances = face_distances.tolist()
        min_distance = min(face_distances)
        min_index = face_distances.index(min_distance)
        pred_label = known_img_label_dict.get(min_index, 'unk')
        if pred_label == img_label:
            right_count += 1
        print('pred_label:{},\t img_label:{},\t min_distance:{}'.format(pred_label, img_label, min_distance))
    print('{}/{}'.format(right_count, sum_count))
    """
    pred_label:chenglong,	 img_label:chenglong,	 min_distance:0.3906323227022819
    pred_label:dongxuan,	 img_label:dongxuan,	 min_distance:0.29246836193302517
    pred_label:guanzhilin,	 img_label:guanzhilin,	 min_distance:0.27012643856720475
    pred_label:gulinazha,	 img_label:gulinazha,	 min_distance:0.31939417724474395
    pred_label:gutianle,	 img_label:gutianle,	 min_distance:0.30113090183423596
    pred_label:huge,	 img_label:huge,	 min_distance:0.3967538160678809
    pred_label:jindong,	 img_label:jindong,	 min_distance:0.15628897392048907
    pred_label:jingtian,	 img_label:jingtian,	 min_distance:0.3071836831226583
    pred_label:lilianjie,	 img_label:lilianjie,	 min_distance:0.29883914091351726
    pred_label:liming,	 img_label:liming,	 min_distance:0.3287387411170835
    pred_label:linjunjie,	 img_label:linjunjie,	 min_distance:0.32151545256047426
    pred_label:liudehua,	 img_label:liudehua,	 min_distance:0.29985123431121696
    pred_label:sunli,	 img_label:sunli,	 min_distance:0.3460626216485016
    pred_label:jingtian,	 img_label:tongliya,	 min_distance:0.45687686001451794
    pred_label:yangmi,	 img_label:yangmi,	 min_distance:0.3464611332028798
    pred_label:zhangmin,	 img_label:zhangmin,	 min_distance:0.33978723139952904
    pred_label:zhangxueyou,	 img_label:zhangxueyou,	 min_distance:0.2648244638109685
    pred_label:zhoujielun,	 img_label:zhoujielun,	 min_distance:0.3692935296298237
    pred_label:zhourunfa,	 img_label:zhourunfa,	 min_distance:0.3003104639719296
    pred_label:zhouxingchi,	 img_label:zhouxingchi,	 min_distance:0.31288879075045833
    19/20
    """
