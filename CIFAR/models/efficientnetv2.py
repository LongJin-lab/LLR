import torch
import torch.nn as nn


import collections

GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            fused=('f' in block_string),
            id_skip=('noskip' not in block_string))

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def get_efficientnetv2_params(model_name, num_classes):
    #################### EfficientNet V2 configs ####################
    v2_base_block = [  # The baseline config for v2 models.
        'r1_k3_s1_e1_i32_o16_f',
        'r2_k3_s2_e4_i16_o32_f',
        'r2_k3_s2_e4_i32_o48_f',
        'r3_k3_s2_e4_i48_o96_se0.25',
        'r5_k3_s1_e6_i96_o112_se0.25',
        'r8_k3_s2_e6_i112_o192_se0.25',
    ]
    v2_s_block = [  # about base * (width1.4, depth1.8)
        'r2_k3_s1_e1_i24_o24_f',
        'r4_k3_s2_e4_i24_o48_f',
        'r4_k3_s2_e4_i48_o64_f',
        'r6_k3_s2_e4_i64_o128_se0.25',
        'r9_k3_s1_e6_i128_o160_se0.25',
        'r15_k3_s2_e6_i160_o256_se0.25',
    ]
    v2_m_block = [  # about base * (width1.6, depth2.2)
        'r3_k3_s1_e1_i24_o24_f',
        'r5_k3_s2_e4_i24_o48_f',
        'r5_k3_s2_e4_i48_o80_f',
        'r7_k3_s2_e4_i80_o160_se0.25',
        'r14_k3_s1_e6_i160_o176_se0.25',
        'r18_k3_s2_e6_i176_o304_se0.25',
        'r5_k3_s1_e6_i304_o512_se0.25',
    ]
    v2_l_block = [  # about base * (width2.0, depth3.1)
        'r4_k3_s1_e1_i32_o32_f',
        'r7_k3_s2_e4_i32_o64_f',
        'r7_k3_s2_e4_i64_o96_f',
        'r10_k3_s2_e4_i96_o192_se0.25',
        'r19_k3_s1_e6_i192_o224_se0.25',
        'r25_k3_s2_e6_i224_o384_se0.25',
        'r7_k3_s1_e6_i384_o640_se0.25',
    ]
    v2_xl_block = [  # only for 21k pretraining.
        'r4_k3_s1_e1_i32_o32_f',
        'r8_k3_s2_e4_i32_o64_f',
        'r8_k3_s2_e4_i64_o96_f',
        'r16_k3_s2_e4_i96_o192_se0.25',
        'r24_k3_s1_e6_i192_o256_se0.25',
        'r32_k3_s2_e6_i256_o512_se0.25',
        'r8_k3_s1_e6_i512_o640_se0.25',
    ]

    efficientnetv2_params = {
        # (block, width, depth, train_size, eval_size, dropout, randaug, mixup, aug)
        'efficientnetv2-s':  # 83.9% @ 22M
            (v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug'),
        'efficientnetv2-m':  # 85.2% @ 54M
            (v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug'),
        'efficientnetv2-l':  # 85.7% @ 120M
            (v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug'),

        'efficientnetv2-xl':
            (v2_xl_block, 1.0, 1.0, 384, 512, 0.4, 20, 0.5, 'randaug'),

        # For fair comparison to EfficientNetV1, using the same scaling and autoaug.
        'efficientnetv2-b0':  # 78.7% @ 7M params
            (v2_base_block, 1.0, 1.0, 192, 224, 0.2, 0, 0, 'effnetv1_autoaug'),
        'efficientnetv2-b1':  # 79.8% @ 8M params
            (v2_base_block, 1.0, 1.1, 192, 240, 0.2, 0, 0, 'effnetv1_autoaug'),
        'efficientnetv2-b2':  # 80.5% @ 10M params
            (v2_base_block, 1.1, 1.2, 208, 260, 0.3, 0, 0, 'effnetv1_autoaug'),
        'efficientnetv2-b3':  # 82.1% @ 14M params
            (v2_base_block, 1.2, 1.4, 240, 300, 0.3, 0, 0, 'effnetv1_autoaug'),
    }

    assert model_name in list(efficientnetv2_params.keys()), "Wrong model name."
    all_params = efficientnetv2_params[model_name]

    blocks_args = BlockDecoder.decode(all_params[0])

    global_params = GlobalParams(
        width_coefficient=all_params[1],
        depth_coefficient=all_params[2],
        image_size=all_params[3],
        dropout_rate=all_params[5],
        num_classes=num_classes,

        batch_norm_momentum=None, #0.99,
        batch_norm_epsilon=None, #1e-3,
        drop_connect_rate=None, #drop_connect_rate,
        depth_divisor=None, #8,
        min_depth=None, #None,
        include_top=None, #include_top,
    )

    return blocks_args, global_params

# CBAM Module
# Not use, just for practice.
class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self._r = reduction
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)
        self._fc1 = nn.Linear(channels, channels // self._r, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self._fc2 = nn.Linear(channels // self._r, channels, bias=False)
        self._sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, _, _ = inputs.size()
        x1 = self._avg_pool(inputs).squeeze()
        x2 = self._max_pool(inputs).squeeze()

        x1 = self._fc2(self._relu(self._fc1(x1)))
        x2 = self._fc2(self._relu(self._fc1(x2)))
      
        y = self._sigmoid(x1 + x2)
        return inputs * y.view(b, c, 1, 1).expand_as(inputs)


# CBAM Module
# Not use, just for practice.
class SpartialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self._max_pool = nn.AdaptiveMaxPool2d((None, 1))
        padding = (kernel_size - 1) // 2
        self._conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self._sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, _, h, w = inputs.size()
        trans_inputs = inputs.transpose(1, 3)
        x1 = self._avg_pool(trans_inputs).transpose(1, 3)
        x2 = self._max_pool(trans_inputs).transpose(1, 3)
        x = torch.cat((x1, x2), dim=1)
        x = self._sigmoid(self._conv(x))
        return inputs * x.view(b, 1, h, w).expand_as(inputs)


class SEModule(nn.Module):
    def __init__(self, channels, ratio=1/16):
        super().__init__()
        self._r = ratio
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = int(channels * self._r)
        self._fc1 = nn.Linear(channels, hidden_channels, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self._fc2 = nn.Linear(hidden_channels, channels, bias=False)
        self._sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, _, _ = inputs.size()
        x = self._avg_pool(inputs).squeeze()
        x = self._relu(self._fc1(x))
        x = self._sigmoid(self._fc2(x))
        return inputs * x.view(b, c, 1, 1).expand_as(inputs)


class Conv2dAutoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        assert kernel_size % 2 == 1, "Only support odd kernel size."
        padding = (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)


class MBConvBlock(nn.Module):
    def __init__(self, block_arg):
        super().__init__()
        self._block_arg = block_arg
        # expand
        inc = self._block_arg.input_filters
        outc = inc * self._block_arg.expand_ratio
        if self._block_arg.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(inc, outc, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(outc)
        # dw
        self._dw_conv = Conv2dAutoPadding(outc, outc, self._block_arg.kernel_size, 
                                    self._block_arg.stride, groups=outc, bias=False)
        self._bn1 = nn.BatchNorm2d(outc)
        # squeeze and extract
        if self._block_arg.se_ratio:
            self._se = SEModule(outc, self._block_arg.se_ratio)
        # pw
        inc = outc
        outc = self._block_arg.output_filters
        self._pw_conv = nn.Conv2d(inc, outc, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(outc)
        # activation
        self._swish = nn.SiLU()

    def forward(self, inputs):
        x = inputs
        if self._block_arg.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._dw_conv(x)))
        if self._block_arg.se_ratio:
            x = self._se(x)
        x = self._bn2(self._pw_conv(x))  # pw conv: linear activation
        if self._block_arg.input_filters == self._block_arg.output_filters and self._block_arg.stride == 1:
            x = x + inputs
        return x


class FusedMBConvBlock(nn.Module):
    def __init__(self, block_arg):
        super().__init__()
        self._block_arg = block_arg
        # fused conv
        inc = self._block_arg.input_filters
        outc = inc * self._block_arg.expand_ratio
        self._fused_conv = Conv2dAutoPadding(inc, outc, self._block_arg.kernel_size, self._block_arg.stride, bias=False)
        self._bn = nn.BatchNorm2d(outc)
        # squeeze and extract
        if self._block_arg.se_ratio:
            self._se = SEModule(outc, self._block_arg.se_ratio)
        # pw
        inc = outc
        outc = self._block_arg.output_filters
        self._pw_conv = nn.Conv2d(inc, outc, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(outc)
        # activation
        self._swish = nn.SiLU()

    def forward(self, inputs):
        x = inputs
        x = self._swish(self._bn(self._fused_conv(inputs)))
        if self._block_arg.se_ratio:
            x = self._se(x)
        x = self._bn2(self._pw_conv(x))  # pw conv: linear activation
        if self._block_arg.input_filters == self._block_arg.output_filters and self._block_arg.stride == 1:
            x = x + inputs
        return x


class EfficientNetV2(nn.Module):
    def __init__(self, blocks_args, global_params):
        super().__init__()
        self._blocks_args = blocks_args
        self._global_params = global_params
        # stem
        inc = 3
        outc = blocks_args[0].input_filters
        self._stem_conv = Conv2dAutoPadding(inc, outc, 3, 2)
        self._bn0 = nn.BatchNorm2d(outc)
        # blocks
        self._blocks = nn.ModuleList([]) # BUG: [] -> nn.ModuleList([])
        for block_arg in self._blocks_args:
            block = FusedMBConvBlock(block_arg) if block_arg.fused == True else MBConvBlock(block_arg)
            self._blocks.append(block)
            if block_arg.num_repeat > 1:
                block_arg = block_arg._replace(input_filters=block_arg.output_filters, stride=1)
            for _ in range(block_arg.num_repeat - 1):
                block = FusedMBConvBlock(block_arg) if block_arg.fused == True else MBConvBlock(block_arg)
                self._blocks.append(block)
        # head
        inc = block_arg.output_filters
        outc = int(self._global_params.width_coefficient * 1280)
        self._head_conv = nn.Conv2d(inc, outc, 1, 1)
        self._bn1 = nn.BatchNorm2d(outc)
        # top
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate) # missing dropout
        self._fc = nn.Linear(outc, self._global_params.num_classes)
        # activation
        self._swish = nn.SiLU()  # hasattr?


    def forward(self, inputs):
        x = self._swish(self._bn0(self._stem_conv(inputs)))

        for i, block in enumerate(self._blocks): # BUG: missing enumerate
            x = block(x)
        
        x = self._swish(self._bn1(self._head_conv(x)))

        x = self._avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x



if __name__ == '__main__':
    blocks_args, global_params = get_efficientnetv2_params('efficientnetv2-b0', 1000)
    model = EfficientNetV2(blocks_args, global_params)
    image_size = global_params.image_size
    x = torch.randn(1, 3, image_size, image_size)
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)