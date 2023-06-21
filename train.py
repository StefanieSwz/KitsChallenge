# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:49:10 2023

@author: st_sc
"""
## just garbage code
        # Get the current working directory
        current_directory = os.getcwd()

        # Print the current working directory
        print("Current working directory:", current_directory)

        # Specify the desired directory path
        new_directory = 'segment-anything'

        # Change the current working directory
        os.chdir(new_directory)

        # Print the updated working directory
        print("Updated working directory:", os.getcwd())

        sys.path.append("\..")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(sam)
        
        
    
    def get_dataloader(self, train = True, train_ratio = 0.7):
        total_size = self.__len__()
        train_indices, valid_indices = split_indices_with_seed(train_ratio=train_ratio, total_size=total_size)
        
        if train:
            data = zip(self.__getitem__(train_indices))
        else:
            data = zip(self.__getitem__(valid_indices))
        
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train)
    
    
    
def split_indices_with_seed(train_ratio, total_size):
    # Calculate sizes for each split
    train_size = int(total_size * train_ratio)

    # Create a list of indices
    indices = list(range(total_size))

    # Shuffle the indices
    random.shuffle(indices)

    # Split the indices
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    return train_indices, valid_indices