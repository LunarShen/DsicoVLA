from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset
import warnings
import logging

logger = logging.getLogger(__name__)

class MSRVTTDataset(RetrievalDataset):
    """MSRVTT dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(MSRVTTDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """
        csv_path = {'train': join(self.anno_path, 'MSRVTT_train.9k.csv'),
                    'val': join(self.anno_path, 'MSRVTT_JSFUSION_test.csv'),
                    'test': join(self.anno_path, 'MSRVTT_JSFUSION_test.csv'),
                    'train_test': join(self.anno_path, 'MSRVTT_train.9k.csv')}[subset]
        if exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError

        video_id_list = list(csv['video_id'].values)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        if subset == 'train':
            PseudoImageCaptions_path = self.config.PseudoImageCaptions_path
            logger.info(f"PseudoImageCaptions_path {PseudoImageCaptions_path}")
            data_v1 = json.load(open(PseudoImageCaptions_path, 'r'))
            need_add = set()
            
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            data = json.load(open(anno_path, 'r'))
            for itm in data['sentences']:
                if itm['video_id'] in video_id_list:
                    name = itm['video_id'] + '_' + itm['caption']
                    if name not in data_v1:
                        need_add.add(name)
                        continue
                    sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['caption'], None, None), data_v1[name])
                    video_dict[itm['video_id']] = join(self.video_path, "{}.mp4".format(itm['video_id']))
                    
            if len(need_add)>0:
                warnings.warn(f"{len(need_add)} need add.")
                logger.info(f"{len(need_add)} need add.")
            else:
                logger.info("success load ImageCaption!")
        
        elif subset == 'train_test':
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            data = json.load(open(anno_path, 'r'))
            used = []
            for itm in data['sentences']:
                if itm['video_id'] in video_id_list and itm['video_id'] not in used:
                    used.append(itm['video_id'])
                    sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['caption'], None, None))
                    video_dict[itm['video_id']] = join(self.video_path, "{}.mp4".format(itm['video_id']))
        else:
            for _, itm in csv.iterrows():
                sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['sentence'], None, None))
                video_dict[itm['video_id']] = join(self.video_path, "{}.mp4".format(itm['video_id']))

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict
