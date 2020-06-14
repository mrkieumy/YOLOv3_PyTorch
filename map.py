import torch
from darknet import Darknet
import dataset
from torchvision import transforms
from utils import get_all_boxes, nms, read_data_cfg, load_class_names
from image import correct_yolo_boxes
import os
import tqdm
from my_eval import Evaluation_from_Valid
from LAMR_AP import meanAP_LogAverageMissRate
from output2JSON import convert_predict_to_JSON

def valid(datacfg, cfgfile, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    print('Validate with the list file: ',valid_images)
    name_list = options['names']
    prefix = 'results'
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(cfgfile)
    # m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    fps = [0]*m.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(m.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    if m.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(m.width, m.height)
    for _, (data, target, org_w, org_h) in enumerate(tqdm.tqdm(valid_loader)):
        data = data.cuda()
        output = m(data)
        batch_boxes = get_all_boxes(output, shape, conf_thresh, m.num_classes, only_objectness=0, validation=True)
        
        for i in range(len(batch_boxes)):
            lineId += 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = float(org_w[i]), float(org_h[i])
            boxes = batch_boxes[i]
            correct_yolo_boxes(boxes, width, height, m.width, m.height)
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = int(box[6+2*j])
                    prob = det_conf * cls_conf
                    fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

    for i in range(m.num_classes):
        fps[i].close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) >=1:
        datacfg = 'data/kaist.data'
        cfgfile = 'cfg/yolov3_kaist.cfg'
        weightfile = 'weights/kaist_thermal_detector.weights'
        outfile = 'det_test_'

        if len(sys.argv) == 2:
            weightfile = sys.argv[1]
        elif len(sys.argv) == 3:
            weightfile = sys.argv[1]
            cfgfile = sys.argv[2]
        elif len(sys.argv) == 4:
            weightfile = sys.argv[1]
            cfgfile = sys.argv[2]
            datacfg = sys.argv[3]

        print('validation with datacfg = %s, cfgfile = %s, weighfile = %s \n' % (datacfg, cfgfile, weightfile))
        valid(datacfg, cfgfile, weightfile, outfile)
        options = read_data_cfg(datacfg)
        test_file = options['valid']
        class_names = options['names']
        _map = Evaluation_from_Valid('results/'+outfile, test_file, class_names, output_dir='output')
        if datacfg == 'data/kaist.data':
            convert_predict_to_JSON()
            all_ap, day_ap, night_ap, all_mr, day_mr, night_mr = meanAP_LogAverageMissRate()
            print('Precision day & night: %.4f \t Precision daytime: %.4f \t Precision nighttime: %.4f \n'
                  'Miss rate day & night: %.4f \t Miss rate daytime: %.4f \t Miss rate nighttime: %.4f \n' % (
                all_ap / 100.0, day_ap / 100.0, night_ap / 100.0, all_mr / 100.0, day_mr / 100.0,
                night_mr / 100.0))

    else:
        print('Usage: python map.py [weightfile] [cfgfile] [datacfg] ')
        print('example: python map.py weights/kaist_thermal_detector.weight cfg/yolov3_kaist.cfg data/kaist.data')
