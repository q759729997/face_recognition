import sys
from click.testing import CliRunner

sys.path.append('./')

from face_recognition import face_detection_cli  # noqa


if __name__ == "__main__":
    runner = CliRunner()
    image_file = './data/test_data/mingxing/test/chenglong.jpg'
    result = runner.invoke(face_detection_cli.main, args=[image_file])
    print('exit_code:{}'.format(result.exit_code))  # ./data/test_data/mingxing/test/chenglong.jpg,50,509,308,251
    print('output:{}'.format(result.output))  # exit_code:0
