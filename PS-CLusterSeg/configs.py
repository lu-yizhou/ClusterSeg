import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.img_size = (512, 512)

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = './ckpt/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    config.patch_size = 16
    config.head_channels = 512
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    config.is_deconv = True
    config.is_batchnorm = True
    config.is_ds = False

    # Resnet50
    config.patches.grid = (32, 32)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 6, 3)
    config.resnet.width_factor = 1
    config.skip_channels = [256, 128, 64, 32]
    config.n_skip = 4
    return config
