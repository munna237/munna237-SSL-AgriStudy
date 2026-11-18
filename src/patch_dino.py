import sys

def patch_dino(repo):
    """
    Minimal patches to the vanilla facebookresearch/dino repo (each must match exactly once):
      1. --init_weights PATH : initialise the student BACKBONE from a checkpoint
         (DINO then copies the student into the teacher). New projection head.
      2. --init_full PATH    : initialise student AND teacher (backbone + projection head)
         and the DINO loss centre from a full DINO training checkpoint, e.g. the released
         dino_deitsmall16_pretrain_full_checkpoint.pth. Optimizer/epoch are NOT restored.
      3. torch.load(..., weights_only=False) for PyTorch >= 2.6.
      4. torch.hub.list('facebookresearch/xcit') calls removed (unused, needs network).
    No other change to training.
    """
    def sub_once(text, old, new, fname):
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f'{fname}: expected 1 match for patch, found {n}:\n{old[:90]}')
        return text.replace(old, new)

    f = f'{repo}/main_dino.py'
    s = open(f).read()
    if 'init_weights' not in s:
        s = sub_once(s, "                + torchvision_archs + torch.hub.list(\"facebookresearch/xcit:main\"),",
                        "                + torchvision_archs,", f)
        s = sub_once(s, "    elif args.arch in torch.hub.list(\"facebookresearch/xcit:main\"):",
                        "    elif False:  # xcit disabled (not used in this study)", f)
        s = sub_once(s, "    parser.add_argument('--seed', default=0, type=int, help='Random seed.')",
                        "    parser.add_argument('--seed', default=0, type=int, help='Random seed.')\n"
                        "    parser.add_argument('--init_weights', default='', type=str,\n"
                        "        help='Optional backbone checkpoint to initialise the student from.')", f)
        s = sub_once(s, "    # multi-crop wrapper handles forward with inputs of different resolutions\n",
            "    if args.init_weights:\n"
            "        _sd = torch.load(args.init_weights, map_location='cpu', weights_only=False)\n"
            "        for _k in ('teacher', 'student', 'model', 'state_dict'):\n"
            "            if isinstance(_sd, dict) and _k in _sd and isinstance(_sd[_k], dict):\n"
            "                _sd = _sd[_k]; break\n"
            "        _sd = {k.replace('module.', '').replace('backbone.', ''): v\n"
            "               for k, v in _sd.items() if not k.replace('module.', '').replace('backbone.', '').startswith('head')}\n"
            "        _msg = student.load_state_dict(_sd, strict=True)\n"
            "        print(f'Initialised student backbone from {args.init_weights}: {_msg}')\n"
            "\n"
            "    # multi-crop wrapper handles forward with inputs of different resolutions\n", f)
    if 'init_full' not in s:
        s = sub_once(s, "        help='Optional backbone checkpoint to initialise the student from.')",
                        "        help='Optional backbone checkpoint to initialise the student from.')\n"
                        "    parser.add_argument('--init_full', default='', type=str,\n"
                        "        help='Optional full DINO checkpoint: student+teacher (backbone and head) and loss centre.')", f)
        s = sub_once(s, "    teacher_without_ddp.load_state_dict(student.module.state_dict())\n",
            "    teacher_without_ddp.load_state_dict(student.module.state_dict())\n"
            "    _ck = None\n"
            "    if args.init_full:\n"
            "        _ck = torch.load(args.init_full, map_location='cpu', weights_only=False)\n"
            "        _st = {k.replace('module.', ''): v for k, v in _ck['student'].items()}\n"
            "        _te = {k.replace('module.', ''): v for k, v in _ck['teacher'].items()}\n"
            "        print('init_full student:', student.module.load_state_dict(_st, strict=True))\n"
            "        print('init_full teacher:', teacher_without_ddp.load_state_dict(_te, strict=True))\n", f)
        s = sub_once(s, "        args.epochs,\n    ).cuda()\n",
            "        args.epochs,\n    ).cuda()\n"
            "    if _ck is not None and 'dino_loss' in _ck:\n"
            "        print('init_full loss centre:', dino_loss.load_state_dict(_ck['dino_loss']))\n", f)
    open(f, 'w').write(s)

    f = f'{repo}/utils.py'
    u = open(f).read()
    if 'weights_only=False' not in u:
        u = sub_once(u, 'checkpoint = torch.load(ckp_path, map_location="cpu")',
                        'checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)', f)
        u = sub_once(u, 'state_dict = torch.load(pretrained_weights, map_location="cpu")',
                        'state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)', f)
        open(f, 'w').write(u)
    print('✓ DINO repo patched (init_weights, init_full, weights_only, xcit removed)')

if __name__ == '__main__':
    patch_dino(sys.argv[1])
