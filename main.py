from __future__ import division

import sys, time, torch, random, argparse#, PIL
# from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from lib.datasets import GeneralDataset as Dataset
from lib.xvision import transforms, draw_image_by_points
from lib.models import obtain_model, remove_module_dict
from lib.config_utils import load_configure

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_model(model_path, face=(250, 150, 900, 1100)):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # print('The image is {:}'.format(args.image))
    print('The model is {:}'.format(model_path))
    snapshot = Path(model_path)
    assert snapshot.exists(), 'The model path {:} does not exist'
    print('The face bounding box is {:}'.format(face))
    assert len(face) == 4, 'Invalid face input : {:}'.format(face)
    snapshot = torch.load(snapshot)

    # General Data Argumentation
    mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    param = snapshot['args']
    # import pdb;
    # pdb.set_trace()
    eval_transform = transforms.Compose(
        [transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),
         transforms.ToTensor(), normalize])
    model_config = load_configure(param.model_config, None)
    dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
    dataset.reset(param.num_pts)

    net = obtain_model(model_config, param.num_pts + 1)
    net = net.cuda()
    # weights = remove_module_dict(snapshot['detector'])
    try:
        weights = remove_module_dict(snapshot['detector'])
    except:
        weights = remove_module_dict(snapshot['state_dict'])
    net.load_state_dict(weights)

    return net, dataset, face


def predict(net, dataset, face, image):
    print('Prepare input data')
    [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(image, face)
    inputs = image.unsqueeze(0).cuda()
    # network forward
    with torch.no_grad():
        batch_heatmaps, batch_locs, batch_scos = net(inputs)
    # obtain the locations on the image in the orignial size
    cpu = torch.device('cpu')
    np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(
        cpu).numpy(), cropped_size.numpy()
    locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

    scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

    locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                       cropped_size[3]
    prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)

    return locations, prediction


def main():
    model_path = r"E:\code\supervision-by-registration\snapshots\cpm_vgg16-epoch-049-050.pth"
    net, dataset, face = get_model(model_path)

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    left_eye_idxs = np.arange(36, 42)
    right_eye_idxs = np.arange(42, 48)
    between_eyes_idx = np.array([27])
    all_eyes_idxs = np.concatenate((left_eye_idxs, right_eye_idxs, between_eyes_idx))

    lines = None
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        b, g, r = cv2.split(frame)
        frame_rgb = cv2.merge((r, g, b))

        locations, prediction = predict(net, dataset, face, frame_rgb)

        eyes_mean_location = locations[all_eyes_idxs].mean(axis=0)

        if lines is not None:
            lines.pop(0).remove()

        plt.imshow(frame_rgb)
        # plt.scatter(*locations.T)
        # [plt.text(x, y, str(i), color='w') for i, (x, y) in enumerate(locations)]
        lines = plt.plot(*eyes_mean_location, 'x')
        plt.title('Matplotlib')
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)


        # cv2.plot()

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
