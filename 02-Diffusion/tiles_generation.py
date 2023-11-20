import torch
from classifier_free_guidance import Unet, GaussianDiffusion
from pathlib import Path
from torchvision import utils
from argparse import ArgumentParser
from ema_pytorch import EMA


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--devices", type=str, default="gpu", required=True, help='gpu or cpu? note: do not insert "cuda"')
    parser.add_argument('--path_to_model', type=str, default=None, required=True, help='path to the model to use for the generation')
    parser.add_argument('--sampling_timesteps', type=int, default=250, required=False)
    parser.add_argument('--num_classes', type=int, default=5, required=True)
    parser.add_argument('--save_folder', type=str, default=None, required=True,help='where to save new syntetich images')
    parser.add_argument('--tissue', type=str, required=True,
                        choices= ['Brain', 'Kidney','Lung','Pancreas','Uterus'] ,
                        help='which tissue generate')

    args = parser.parse_args()





    number_model = Path(args.path_to_model).stem
    sampling_timesteps = args.sampling_timesteps
    #samp_t = str(sampling_timesteps)
    tissue_name = args.tissue


    if args.devices == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'running on {device}')

    num_classes = args.num_classes

    unet = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    ).to(device)

    model = GaussianDiffusion(
        unet,
        image_size = 256,
        timesteps = 1000,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2'
    ).to(device)

    ema = EMA(model, beta = 0.995, update_every = 2)

    checkpoint = torch.load(args.path_to_model, map_location=device)
    ema.load_state_dict(checkpoint['ema'])

    if num_classes == 5:
        tissue_dict = {
            'Brain': torch.tensor([0]),
            'Kidney' : torch.tensor([1]),
            'Lung' : torch.tensor([2]),
            'Pancreas' : torch.tensor([3]),
            'Uterus' : torch.tensor([4])
        }


    print(f'starting generating images for {tissue_name}')
    
    for index in range(0,11): 
        sampled_images = ema.ema_model.sample(
                                    classes = tissue_dict.get(tissue_name).to(device),
                                    cond_scale = 3.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
                                    )
        utils.save_image(sampled_images, f'{args.save_folder}/{tissue_name}/trueEma_sample_{number_model}_ema_sampling-steps{sampling_timesteps}_{index}_{tissue_name}.png')
