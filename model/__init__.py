def build_model(args):
    use_vti_impl = getattr(args, "use_vti_impl", False)

    if args.model_name in {"LLaVA-7B", "LLaVA-13B"}:
        if use_vti_impl:
            from .LLaVA_vti import LLaVA
        else:
            from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "MiniGPT4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif args.model_name == "mPLUG_Owl2":
        from .mPLUG_Owl2 import mPLUG_Owl2
        model = mPLUG_Owl2(args)
    elif args.model_name == "Qwen_VL_Chat":
        if use_vti_impl:
            from .Qwen_VL_Chat_vti import Qwen_VL_Chat
        else:
            from .Qwen_VL_Chat import Qwen_VL_Chat
        model = Qwen_VL_Chat(args)
    else:
        model = None
        
    return model
