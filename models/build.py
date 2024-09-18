from .image_swin import Nuclei_swin
#from .nuclei_swin_nomask import Nuclei_swin

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'nuclei':
        model = Nuclei_swin(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                zoom_size=config.DATA.ZOOM_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                depths_y=config.MODEL.SWIN.DEPTHS_Y,
                                dim_y=config.MODEL.SWIN.EMBED_DIM_Y,
                                mix_depth=config.MODEL.SWIN.MIX_DEPTH,
                                mix_heads = config.MODEL.SWIN.MIX_HEADS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT
                                )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
