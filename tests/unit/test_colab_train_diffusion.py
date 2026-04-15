from scripts.colab_train_diffusion import build_parser


def test_build_parser_sets_stable_diffusion_style_sampling_defaults():
    parser = build_parser()

    args = parser.parse_args([
        "--data-output", "/tmp/data",
        "--run-output", "/tmp/runs",
    ])

    assert args.sampler == "ddim"
    assert args.guidance_scale == 1.0
    assert args.num_inference_steps == 50
