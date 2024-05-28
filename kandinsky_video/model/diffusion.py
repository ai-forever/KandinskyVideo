from diffusers import DDPMScheduler
from tqdm import tqdm

import torch

def get_sampler(
    num_train_timesteps = 1000,
    num_inference_timesteps = 50,
    dynamic_thresholding_ratio = 0.995,
    thresholding = True,
    variance_type = 'fixed_small_log',
    beta_schedule = 'squaredcos_cap_v2',
    prediction_type='epsilon',
    rescale_betas_zero_snr=True,
    timestep_spacing='leading',
    _type='ddpm'
):
    sampler =  DDPMScheduler(
        num_train_timesteps = num_train_timesteps,
        dynamic_thresholding_ratio = dynamic_thresholding_ratio,
        thresholding = thresholding,
        variance_type = variance_type,
        beta_schedule = beta_schedule,
        prediction_type=prediction_type,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
        timestep_spacing=timestep_spacing, # linspace, trailing, leading
    )
    sampler.set_timesteps(num_inference_timesteps)
    return sampler
    
class BaseDiffusion:

    def __init__(self):
        self.interpolation_sampler = get_sampler(
            _type='ddpm',
            num_train_timesteps = 1000,
            num_inference_timesteps = 10,
            dynamic_thresholding_ratio = 0.995, 
            thresholding = True,
            variance_type = 'fixed_small_log',
            beta_schedule = 'squaredcos_cap_v2',
            prediction_type='v_prediction',
            rescale_betas_zero_snr=True,
            timestep_spacing='trailing',
        )
        
    def p_sample_loop_image(
        self, model, shape, device, context, context_mask, null_embedding, guidance_weight
    ):
        sampler = get_sampler()
        
        x = torch.randn(*shape, device=device)
        condition = torch.zeros_like(x)
        mask = torch.zeros_like(x[:, :1])
        
        uncondition_context, uncondition_context_mask = self._get_unconditional_context(
            context, context_mask, null_embedding
        )        
        large_context = torch.cat([context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask])

        large_condition = torch.cat([condition, condition], dim=0)
        large_mask = torch.cat([mask, mask], dim=0)
        
        for time in tqdm(sampler.timesteps):
            t = torch.tensor([time.to(torch.int64)] * shape[0], device=device)
            large_t = t.repeat(2)
            large_x = x.repeat(2, 1, 1, 1)
                            
            large_x = torch.cat([large_x, large_condition, large_mask], dim=1).to(context.dtype)
            
            pred_large_noise = model(
                large_x, 
                large_t.to(context.dtype), 
                large_context, 
                large_context_mask.bool()
            )
            
            pred_noise, uncond_pred_noise= torch.chunk(pred_large_noise, 2)
            pred_noise = (guidance_weight + 1.) * pred_noise - guidance_weight * uncond_pred_noise
            
            pred_prev_sample = sampler.step(pred_noise, time, x).prev_sample
            
            x = pred_prev_sample

        return x.to(torch.float32)
    
    def p_sample_loop(
        self, 
        model, 
        shape, 
        device, 
        context, 
        context_mask, 
        null_embedding, 
        guidance_weight, 
        num_frames=None, 
        key_frame=None,        
        guidance_weight_image=None, 
        motion_score=50, 
        noise_augmentation=20
    ):
        sampler = get_sampler()
        x = torch.randn(*shape, device=device)
        
        temporal = torch.arange(num_frames, device=device)
        noise_aug_time = noise_augmentation * torch.ones(size=(shape[0],), dtype=torch.int32, device=device)
        motion_score = motion_score * torch.ones(size=(shape[0],), dtype=torch.int32, device=device)
        
        noise = torch.rand_like(key_frame)
        key_frame = sampler.add_noise(key_frame, noise, torch.tensor([noise_augmentation]))       
        condition = torch.zeros_like(x)
        mask = torch.zeros_like(x[:, :1])
        condition[0] = key_frame
        mask[0] = 1
        
        uncondition_context, uncondition_context_mask = self._get_unconditional_context(context, context_mask, null_embedding)        
        large_condition = torch.cat([condition, condition, torch.zeros_like(condition)], dim=0)
        large_mask = torch.cat([mask, mask, torch.zeros_like(mask)], dim=0)
        large_context = torch.cat([context, uncondition_context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask, uncondition_context_mask])

        large_temporal = torch.cat([temporal, temporal, temporal]).to(context.dtype)
        noise_aug_time = torch.cat([noise_aug_time, noise_aug_time, noise_aug_time])
        motion_score = torch.cat([motion_score, motion_score, motion_score])

        for time in tqdm(sampler.timesteps):
            t = torch.tensor([time.to(torch.int64)] * shape[0], device=device)
            large_t = t.repeat(3)
            large_x = x.repeat(3, 1, 1, 1)
                            
            large_x = torch.cat([large_x, large_condition, large_mask], dim=1).to(context.dtype)
            
            pred_large_noise = model(
                large_x, 
                large_t.to(context.dtype), 
                large_context, 
                large_context_mask.bool(), 
                large_temporal, num_frames, noise_aug_time, motion_score
            )
            
            pred_noise, pred_noise_I, uncond_pred_noise= torch.chunk(pred_large_noise, 3)
            pred_noise = (1-guidance_weight_image)*uncond_pred_noise + (guidance_weight_image-guidance_weight)*pred_noise_I + guidance_weight*pred_noise
            
            pred_prev_sample = sampler.step(pred_noise, time, x).prev_sample
            
            x = pred_prev_sample

        return x.to(torch.float32)
    
    def p_sample_loop_interpolation(
        self, 
        model,
        shape,
        device,
        key_frames,
        context,
        context_mask,
        null_embedding, 
        guidance_weight,
        perturbation_time=0, 
        skip_frames=3,
        use_temporal_mask=True,
    ):
        num_predicted_groups = shape[0]
        x = torch.randn(*shape, device=device)
        
        left_base_frames, right_base_frames = key_frames[:-1], key_frames[1:]
        skip_frames = skip_frames * torch.ones(size=(num_predicted_groups,), dtype=torch.int32, device=device)
        perturbation_time = perturbation_time * torch.ones(size=(num_predicted_groups,), dtype=torch.int32, device=device)
        temporal_mask = torch.ones(x.shape[0], device=x.device)
        
        uncondition_context, uncondition_context_mask = self._get_unconditional_context(context, context_mask, null_embedding)        
        large_context = torch.cat([context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask])

        if use_temporal_mask:
            temporal_mask=torch.cat([temporal_mask, torch.zeros_like(temporal_mask)])
        else:
            temporal_mask=torch.cat([temporal_mask, temporal_mask])
        
        perturbation_time = torch.cat([perturbation_time,  perturbation_time])
        skip_frames = torch.cat([skip_frames, skip_frames])
        left_base_frames = torch.cat([left_base_frames, torch.zeros_like(left_base_frames)])
        right_base_frames = torch.cat([right_base_frames, torch.zeros_like(right_base_frames)])
        
        for time in tqdm(self.interpolation_sampler.timesteps):
            t = torch.tensor([time.to(torch.int64)] * shape[0], device=device)
            large_t = t.repeat(2)
 
            large_x = x.repeat(2, 1, 1, 1)
            large_x = torch.cat([left_base_frames, large_x, right_base_frames], axis=1)

            pred_large_noise = model(
                large_x, 
                large_t, 
                large_context, 
                large_context_mask.bool(), 
                perturbation_time=perturbation_time, 
                skip_frames=skip_frames, 
                num_predicted_groups=num_predicted_groups,
                temporal_mask=temporal_mask
            )
            pred_noise, uncond_pred_noise= torch.chunk(pred_large_noise, 2)
            pred_noise = (guidance_weight + 1.) * pred_noise - guidance_weight * uncond_pred_noise
            
            pred_prev_sample = self.interpolation_sampler.step(pred_noise, time, x).prev_sample
            
            x = pred_prev_sample
        return x
    
    def _get_unconditional_context(self, context, context_mask, null_embedding):
        uncondition_context = torch.zeros_like(context)
        uncondition_context_mask = torch.zeros_like(context_mask)
        uncondition_context[:, 0] = null_embedding
        uncondition_context_mask[:, 0] = 1
        
        return uncondition_context, uncondition_context_mask