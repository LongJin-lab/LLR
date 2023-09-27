from .build import MODELS_REGISTRY
# from .alexnet import *
from .cbam_resnext import *
from .densenet import *
from .genet import *
# from .lenet import *
from .preresnet import *
from .resnext import *
from .senet import *
from .shake_shake import *
from .sknet import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)

from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from .wideresnet import (
    wideresnet28x10,
    wideresnet34x10,
)

from .vgg import (
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
)

from .resnet_small import (
resnet20_small,
resnet32_small,
resnet44_small,
resnet56_small
)
# from .resnet_small import cifar10_resnet20
# from .resnet_small import cifar10_resnet32
# from .resnet_small import cifar10_resnet44
# from .resnet_small import cifar10_resnet56

# from .resnet_small import cifar100_resnet20
# from .resnet_small import cifar100_resnet32
# from .resnet_small import cifar100_resnet44
# from .resnet_small import cifar100_resnet56


from .mobilenetv2 import mobilenetv2_x0_5
from .mobilenetv2 import mobilenetv2_x0_75
from .mobilenetv2 import mobilenetv2_x1_0
from .mobilenetv2 import mobilenetv2_x1_4

# from .mobilenetv2 import cifar100_mobilenetv2_x0_5
# from .mobilenetv2 import cifar100_mobilenetv2_x0_75
# from .mobilenetv2 import cifar100_mobilenetv2_x1_0
# from .mobilenetv2 import cifar100_mobilenetv2_x1_4

from .shufflenetv2 import shufflenetv2_x0_5
from .shufflenetv2 import shufflenetv2_x1_0
from .shufflenetv2 import shufflenetv2_x1_5
from .shufflenetv2 import shufflenetv2_x2_0

# from .shufflenetv2 import cifar100_shufflenetv2_x0_5
# from .shufflenetv2 import cifar100_shufflenetv2_x1_0
# from .shufflenetv2 import cifar100_shufflenetv2_x1_5
# from .shufflenetv2 import cifar100_shufflenetv2_x2_0

from .repvgg import repvgg_a0
from .repvgg import repvgg_a1
from .repvgg import repvgg_a2

# from .repvgg import cifar100_repvgg_a0
# from .repvgg import cifar100_repvgg_a1
# from .repvgg import cifar100_repvgg_a2

from .vit import vit_b16
from .vit import vit_b32
from .vit import vit_l16
from .vit import vit_l32
from .vit import vit_h14

# from .vit import cifar100_vit_b16
# from .vit import cifar100_vit_b32
# from .vit import cifar100_vit_l16
# from .vit import cifar100_vit_l32
# from .vit import cifar100_vit_h14
