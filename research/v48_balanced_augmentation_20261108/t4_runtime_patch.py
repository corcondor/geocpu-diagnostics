#!/usr/bin/env python3
"""Reproducible T4 16-GiB modification of the pinned v47 training runtime.

Keep the exact logical batch, optimizer-step count, data/weights, context,
teacher loss and 36,864-token sampler budget. Only reduce transient microchunk
memory and use T4-compatible fp16 AMP + non-reentrant activation checkpointing.

The original source must already be pinned by the caller's SHA check. Each
edit fails closed when its precise original anchor changes.
"""
from __future__ import annotations
import ast


def replace_once(text: str, before: str, after: str) -> str:
    actual=text.count(before)
    if actual!=1:
        raise ValueError(f"Expected one T4 patch anchor; found {actual}: {before[:110]!r}")
    return text.replace(before,after,1)


def patch_runtime(source: str) -> str:
    out=source

    out=replace_once(out,
        "def forward_backward_chunked(model, batch, policy, dev, use_amp, weight=1.0, teacher=None, distill_weight=1.0,\n"
        "                             distill_temperature=1.0):",
        "def forward_backward_chunked(model, batch, policy, dev, use_amp, weight=1.0, teacher=None, distill_weight=1.0,\n"
        "                             distill_temperature=1.0, scaler=None, amp_dtype=torch.bfloat16):")

    out=replace_once(out,
        "with torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):",
        "with torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=amp_dtype):")

    out=replace_once(out,
        "        loss.backward()\n        total += float(loss.detach())",
        "        if scaler is not None and scaler.is_enabled():\n"
        "            scaler.scale(loss).backward()\n"
        "        else:\n"
        "            loss.backward()\n"
        "        total += float(loss.detach())")

    out=replace_once(out,
        "def step_with_retry(model, optimizer, batch, policy, dev, use_amp, teacher=None, distill_weight=1.0,\n"
        "                    distill_temperature=1.0, kl_log=None, step=0):",
        "def step_with_retry(model, optimizer, batch, policy, dev, use_amp, teacher=None, distill_weight=1.0,\n"
        "                    distill_temperature=1.0, kl_log=None, step=0, scaler=None, amp_dtype=torch.bfloat16):")

    out=replace_once(out,
        "    while True:\n        optimizer.zero_grad(set_to_none=True)\n        try:\n"
        "            loss_val, kl_val = forward_backward_chunked(model, batch, policy, dev, use_amp, 1.0, teacher,\n"
        "                                                        distill_weight, distill_temperature)\n",
        "    fp16_overflow_retries = 0\n"
        "    while True:\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        try:\n"
        "            loss_val, kl_val = forward_backward_chunked(model, batch, policy, dev, use_amp, 1.0, teacher,\n"
        "                                                        distill_weight, distill_temperature, scaler=scaler,\n"
        "                                                        amp_dtype=amp_dtype)\n")

    out=replace_once(out,
        "        if loss_val is not None:\n"
        "            if kl_val is not None and kl_log is not None:\n",
        "        if loss_val is not None:\n"
        "            if scaler is not None and scaler.is_enabled():\n"
        "                scaler.unscale_(optimizer)\n"
        "                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,\n"
        "                    error_if_nonfinite=False)\n"
        "                if not bool(torch.isfinite(norm)):\n"
        "                    fp16_overflow_retries += 1\n"
        "                    previous_scale = float(scaler.get_scale())\n"
        "                    if fp16_overflow_retries > 12 or previous_scale <= 1.0:\n"
        "                        raise RuntimeError('T4 FP16 gradient overflow cannot be recovered without skipping an optimizer update')\n"
        "                    next_scale = max(1.0, previous_scale / 2.0)\n"
        "                    scaler.update(new_scale=next_scale)\n"
        "                    print(f'[T4] FP16 overflow: re-running full SAME batch with loss scale {next_scale:.0f}; '\n"
        "                          f'no example or optimizer step skipped', flush=True)\n"
        "                    continue\n"
        "            if kl_val is not None and kl_log is not None:\n")

    out=replace_once(out,
        "    model.to(dev)\n    out_dir = Path(args.output_dir)\n",
        "    model.to(dev)\n"
        "    # T4 activations are checkpointed; the weights/architecture are unchanged.\n"
        "    if dev.type == 'cuda' and 'T4' in torch.cuda.get_device_name(dev):\n"
        "        from torch.utils.checkpoint import checkpoint as _activation_checkpoint\n"
        "        for block in (*model.base.encoder, *model.base.decoder, *model.fine_decoder):\n"
        "            raw_forward = block.forward\n"
        "            def _t4_checkpointed_forward(*args, _raw=raw_forward, **kwargs):\n"
        "                if not torch.is_grad_enabled():\n"
        "                    return _raw(*args, **kwargs)\n"
        "                return _activation_checkpoint(_raw, *args, use_reentrant=False,\n"
        "                    preserve_rng_state=True, **kwargs)\n"
        "            block.forward = _t4_checkpointed_forward\n"
        "        print('[T4] activation checkpointing enabled for base encoder/decoder + fine decoder', flush=True)\n"
        "    out_dir = Path(args.output_dir)\n")

    out=replace_once(out,
        "        use_amp = (dev.type == \"cuda\")\n"
        "        gpu_name, free_gb = \"cpu\", 0.0\n",
        "        use_amp = (dev.type == 'cuda')\n"
        "        amp_dtype = (torch.float16 if (dev.type == 'cuda' and not torch.cuda.is_bf16_supported())\n"
        "                     else torch.bfloat16)\n"
        "        scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda' and amp_dtype == torch.float16),\n"
        "                                     init_scale=256.0, growth_interval=2000)\n"
        "        gpu_name, free_gb = 'cpu', 0.0\n")

    out=replace_once(out,
        "        print(f\"  Execution Mode:  AMP bfloat16={use_amp} | Dynamic Token Budget: {max_tokens_per_micro:,} tok/micro (Max B={max_micro_batch}, Accum={grad_accum_steps})\")\n",
        "        print(f'  Execution Mode: AMP dtype={amp_dtype}, GradScaler={scaler.is_enabled()} | '\n"
        "              f'FULL logical sampler budget={max_tokens_per_micro:,} tok/update, '\n"
        "              f'activation microchunk starts={chunk_start:,} tokens', flush=True)\n")

    out=replace_once(out,
        "                        \"source_exposures\": exposure_counts, \"chunk\": policy.describe()}\n",
        "                         \"source_exposures\": exposure_counts, \"chunk\": policy.describe(),\n"
        "                         \"amp_dtype\": str(amp_dtype), \"amp_scaler_enabled\": scaler.is_enabled()}\n")

    out=replace_once(out,
        "                      \"practical_variant\": args.practical_variant,\n",
        "                      \"practical_variant\": args.practical_variant,\n"
        "                      \"amp_scaler_state\": scaler.state_dict() if scaler.is_enabled() else None,\n")

    out=replace_once(out,
        "            if state[\"gpu_name\"] != (torch.cuda.get_device_name(dev) if dev.type == \"cuda\" else \"cpu\"):\n"
        "                raise RuntimeError(\"GPU model changed on exact resume\")\n",
        "            current_gpu = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'\n"
        "            if state['gpu_name'] != current_gpu:\n"
        "                # Only a checked new data phase may change A100 -> T4.\n"
        "                if not (args.new_dataset_sampler and 'T4' in current_gpu\n"
        "                        and 'A100' in state['gpu_name']):\n"
        "                    raise RuntimeError('GPU model changed outside declared A100-to-T4 new dataset phase')\n"
        "                print(f'[T4] declared controlled GPU transition: {state[\"gpu_name\"]} -> {current_gpu}', flush=True)\n"
        "            if state.get('amp_scaler_state') is not None:\n"
        "                if not scaler.is_enabled():\n"
        "                    raise RuntimeError('Cannot restore FP16 scaler in non-FP16 training')\n"
        "                scaler.load_state_dict(state['amp_scaler_state'])\n")

    out=replace_once(out,
        "            accum_loss = step_with_retry(model, optimizer, batch, policy, dev, use_amp, teacher, args.distill_weight,\n"
        "                                         args.distill_temperature, kl_log, step)\n",
        "            accum_loss = step_with_retry(model, optimizer, batch, policy, dev, use_amp, teacher, args.distill_weight,\n"
        "                                         args.distill_temperature, kl_log, step,\n"
        "                                         scaler=scaler, amp_dtype=amp_dtype)\n")

    out=replace_once(out,
        "            optimizer.step()\n            for idx in batch_sampler.last_batch:\n",
        "            if scaler.is_enabled():\n"
        "                scale_before = float(scaler.get_scale())\n"
        "                scaler.step(optimizer)\n"
        "                scaler.update()\n"
        "                if float(scaler.get_scale()) < scale_before:\n"
        "                    raise RuntimeError('FP16 scaler unexpectedly skipped an optimizer step; do not record exposure')\n"
        "            else:\n"
        "                optimizer.step()\n"
        "            for idx in batch_sampler.last_batch:\n")

    # The sampler remains batch_size=64, step_tokens=36864. OOM only affects
    # how many sequences are materialized in one forward/backward, never which
    # original training rows are sampled.
    ast.parse(out)
    if not all(piece in out for piece in (
        "same batch is run again",
        "new_dataset_sampler",
        "step_tokens",
        "phase_source_supervised",
        "GradScaler",
        "activation checkpointing",
        "re-running full SAME batch",
        "amp_scaler_state",
    )):
        raise RuntimeError('T4 patch lost required training semantics')
    return out
