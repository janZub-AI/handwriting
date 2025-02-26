import json
import os

class FileManager():
    def rename_files(path, generate_model_name):
        """Renames directories with tensorbord data.
        
        Takes in:
        - 'path' which should be the same as provided to tuner,
        - 'generate_model_name' function
            - should take in (iterable, **kwarg) and extract kwarg['hp'] which should correspond to hyperparameters
            - should produce unique, new name which will be displayed in tensorboard
        """
        path = f'{path}/mist_tuner'
        folders = os.listdir(path)
        
        for f in folders:
            if(f.startswith('trial')):   
                with open(f'{path}/{f}/trial.json') as json_file:
                    data = json.load(json_file)
                    name = data['trial_id']
                    new_name = generate_model_name(hp = data['hyperparameters']['values'])
                    head, _ = os.path.split(path)
                    try:
                        os.rename(f'{head}/{name}',f'{head}/{new_name}')
                    except:    
                        print(f'failed to rename {name} to {new_name}')