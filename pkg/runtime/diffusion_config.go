package runtime

func applyGeneration(config *DiffusionConfig, generationBlock map[string]any) error {
	if value, ok, err := optionalInt(generationBlock, diffusionManifestPrefix, "height"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Height = value
	}

	if value, ok, err := optionalInt(generationBlock, diffusionManifestPrefix, "width"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Width = value
	}

	if value, ok, err := optionalInt(generationBlock, diffusionManifestPrefix, "latent_channels"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.LatentChannels = value
	}

	if value, ok, err := optionalInt(generationBlock, diffusionManifestPrefix, "latent_downsample"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.LatentDownsample = value
	}

	if value, ok, err := optionalInt(generationBlock, diffusionManifestPrefix, "max_sequence_length"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.MaxSequenceLength = value
	}

	if value, ok, err := optionalInt(generationBlock, diffusionManifestPrefix, "pad_token_id"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.PadTokenID = value
	}

	if value, ok, err := optionalInt64(generationBlock, diffusionManifestPrefix, "seed"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Seed = value
	}

	if value, ok, err := optionalString(generationBlock, diffusionManifestPrefix, "output"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Output = value
	}

	if value, ok, err := optionalRawString(generationBlock, diffusionManifestPrefix, "prompt_template"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.PromptTemplate = value
	}

	return nil
}

func applyScheduler(config *DiffusionConfig, schedulerBlock map[string]any) error {
	if value, ok, err := optionalString(schedulerBlock, diffusionManifestPrefix, "type"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Scheduler.Type = value
	}

	if value, ok, err := optionalInt(schedulerBlock, diffusionManifestPrefix, "num_inference_steps"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Scheduler.Steps = value
	}

	return nil
}
