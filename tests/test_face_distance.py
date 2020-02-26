import sys

sys.path.append('./')

import face_recognition  # noqa


if __name__ == "__main__":
    """计算距离"""
    known_chenglong_image = face_recognition.load_image_file("./data/test_data/mingxing/train/chenglong/4d017614bbd293dc.jpg")
    known_huge_image = face_recognition.load_image_file("./data/test_data/mingxing/train/huge/456ff070bfb7381f.jpg")

    # Get the face encodings for the known images
    chenglong_face_encoding = face_recognition.face_encodings(known_chenglong_image)[0]
    huge_face_encoding = face_recognition.face_encodings(known_huge_image)[0]

    known_encodings = [
        chenglong_face_encoding,
        huge_face_encoding
    ]

    # Load a test image and get encondings for it
    image_to_test = face_recognition.load_image_file("./data/test_data/mingxing/test/chenglong.jpg")
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

    for i, face_distance in enumerate(face_distances):
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
        print()
