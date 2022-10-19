from timm.models import create_model
import torch
import torch.nn as nn


def test_mmcv():
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info

    model_name_list = ['mp_mobilenet_v2', 'mp_mobilenet_v2_075', 'mp_mobilenet_v2_050',
                       'ca_mobilenet_v2', 'ca_mobilenet_v2_075', 'ca_mobilenet_v2_050',
                       'mobilenetv2_100', 'mobilenetv2_075', 'mobilenetv2_050', ]

    for mn in model_name_list:
        model = create_model(
            mn,
            pretrained=False,
            num_classes=1000,
            drop_rate=0)

        flops_count, param_count = get_model_complexity_info(
            model, input_shape=(3, 224, 224), print_per_layer_stat=False)

        print(f"Model:{mn.ljust(20)} \t Flops: {flops_count} \t Param: {param_count}")
    
    print('='*20)


def test_ptflops():
    from ptflops import get_model_complexity_info

    model_name_list = ['mp_mobilenet_v2', 'mp_mobilenet_v2_075', 'mp_mobilenet_v2_050',
                       'ca_mobilenet_v2', 'ca_mobilenet_v2_075', 'ca_mobilenet_v2_050',
                       'mobilenetv2_100', 'mobilenetv2_075', 'mobilenetv2_050', ]

    for mn in model_name_list:
        model = create_model(
            mn,
            pretrained=False,
            num_classes=1000,
            drop_rate=0)

        flops_count, param_count = get_model_complexity_info(
            model, (3, 224, 224), print_per_layer_stat=False)

        print(f"Model:{mn.ljust(20)} \t Flops: {flops_count} \t Param: {param_count}")

    print('='*20)
    
if __name__ == '__main__':
    test_mmcv()
    test_ptflops()
