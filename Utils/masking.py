import torch
import math

from logging import getLogger
logger = getLogger()

def random_masking(input, ratio):
    B, N, C = input.shape
    len_keep = int(N * (1 - ratio))

    noise = torch.rand(B, N, device=input.device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    input_masked = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))

    mask = torch.ones(B, N, device=input.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return input_masked, mask, ids_restore

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

def visualise_mask(x, masked_indices, patches):
    h_per_patch = x.shape[1] // patches[0]
    w_per_patch = x.shape[2] // patches[1]
    mask = torch.zeros(x.shape)
    for i in masked_indices:
        for h_i in range(h_per_patch):
            for w_i in range(w_per_patch):
                mask[:, (i//patches[0])*h_per_patch + h_i, (i % patches[1])*w_per_patch + w_i] = 1
    x = x * mask
    return x

    if x is None:
        x = torch.ones

# adapted from https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/multiblock.py#L55
class MaskGenerator(object):

    def __init__(
        self,
        input_size=(16,16),
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        enc_aspect_ratio=(1.0, 1.0),
        pred_aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        device='cpu'
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size[0], input_size[1]
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.enc_aspect_ratio = enc_aspect_ratio
        self.pred_aspect_ratio = pred_aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self.device = device
    
    def to(self, device):
        self.device = device
        return self

    def _sample_block_size(self, scale, aspect_ratio_scale):
        _rand = torch.rand(2)
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand[0].item() * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand[1].item() * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32, device=self.device)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) >= self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze(1)
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32, device=self.device)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def sample_masks(self, batch_size):
        '''
        Create encoder and predictor masks
        # 1. sample pred block shape
        # 2. sample enc block shape
        # 3. sample npred pred block locations for each image
        # 4. sample a single enc block location for each image
        # 5. return enc mask and pred masks
        '''

        p_size = self._sample_block_size(
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.pred_aspect_ratio)
        e_size = self._sample_block_size(
            scale=self.enc_mask_scale,
            aspect_ratio_scale=self.enc_aspect_ratio)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(batch_size):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred