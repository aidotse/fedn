#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : validate.py
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 16.12.2020
# Last Modified Date: 16.12.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>
import logging
import sys
import json
from kerasmodel import create_seed_model

#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':

    logger = logging.getLogger('__name__')
    logger.info("Calling the validate function")
    from fedn.utils.kerasweights import KerasWeightsHelper

    helper = KerasWeightsHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model('.')
    model.set_weights(weights)

    results = model.validate('../data/Annotation/list1_fedn.txt')
    report = {
        "training_loss": results[0],
        "test_loss": results[1],
    }

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))



