import yaml
from pathlib import Path

config = {
    'mixture_spec': [
        {
            'dataset_path': {
                'agibot': ['/share/project/cyfu/agibot_train_data']
            },
            'dataset_weight': 1.0,
            'distribute_weights': True
        }
    ]
}

print('Python dict:')
print(config)

print('\nYAML representation:')
print(yaml.dump(config, default_flow_style=False))

import json
print('\nJSON representation:')
print(json.dumps(config, indent=2))
