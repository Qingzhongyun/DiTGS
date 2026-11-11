"""User-submitted16-rank camera-fixed adapter+GS warmup then residual CA joint20k.

No startup preflight or ledger/asset/source receipt is required. Each update
is collectively checked before Adam/EMA; normal checkpoint loading stays strict.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
from collections import deque
from contextlib import nullcontext
from functools import partial
from .config import parse_config

def main(argv=None):
    cfg = parse_config(argv)
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from .model import MultiViewModel
    from .runtime import fsdp_storage_summary
    from .data import make_dataset, make_dataloader
    from .checkpoint import save_checkpoint, load_checkpoint, load_model_weights
    from .launch_setup import validate_warmup_initialization, warmup_model_contract
    from .recovery import loss_reasons, gradient_reasons, resume_decision, gs_gradient_groups
    from .training_health import update_health, collapse_reason
    from .numerics import capture_rng_state, restore_rng_state, write_json
    from .runtime_monitor import NumericalMonitor, BufferedJsonl
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 16 or world != cfg.world_size:
        raise RuntimeError('This recipe requires exactly16 distributed ranks')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    workspace = Path(cfg.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    sharded = cfg.distributed == 'fsdp'
    diagnostic = False
    raw = MultiViewModel(cfg, initialize_weights=rank == 0 or not sharded)
    if rank == 0:
        write_json(workspace / 'freeze_manifest.json', raw.freeze_manifest())
        write_json(workspace / 'backbone_initialization.json', raw.flashworld_initialization)
        write_json(workspace / 'source_initialization.json', raw.source_initialization)
    if sharded:

        def initialize_meta(module):
            if any((p.is_meta for p in module.parameters(recurse=False))) or any((b.is_meta for b in module.buffers(recurse=False))):
                module.to_empty(device=device, recurse=False)
        model = FSDP(raw, device_id=device, use_orig_params=True, sync_module_states=True, param_init_fn=initialize_meta, backward_prefetch=None, forward_prefetch=False, limit_all_gathers=True, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=raw.fsdp_block_types()))
    else:
        raw.to(device)
        model = DDP(raw, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
    groups = raw.parameter_groups()
    for group in groups:
        group['base_lr'] = group['lr']
    optimizer = torch.optim.AdamW(groups, betas=cfg.optimizer_betas, weight_decay=cfg.weight_decay, foreach=False)
    ema = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    writer = BufferedJsonl()
    append_jsonl = writer.append
    monitor = NumericalMonitor(model, workspace, rank, world, sharded=sharded, writer=writer, spike_factor=cfg.diagnostic_spike_ratio, detail_every=cfg.diagnostic_capture_every)
    if not sharded:
        monitor.attach_pre_reduce_hooks(raw)
    raw.set_activation_recorder(monitor.record_activation)
    progress = {'completed_steps': 0, 'accepted_updates': 0, 'initial_scan_done': False, 'impact_count': 0}
    state_fingerprint = {'recipe_hash': cfg.recipe_hash(), 'implementation_sha256': None, 'input_sha256': None, 'launch_preflight': False, 'ledger_policy': 'user_managed'}
    run_id_items = [uuid.uuid4().hex[:12] if rank == 0 else None]
    dist.broadcast_object_list(run_id_items, src=0)
    run_id = run_id_items[0]
    checkpoint_serial = 0
    loss_history = deque(maxlen=100)
    safe_pointer = workspace / 'latest_safe.json'
    cursor_path = workspace / 'recovery_cursor.json'

    def save(tag):
        nonlocal checkpoint_serial
        writer.flush()
        checkpoint_serial += 1
        folder = workspace / 'checkpoints' / f'{run_id}_{checkpoint_serial:04d}_{tag}'
        metadata = {**state_fingerprint, 'progress': dict(progress), 'gradient_history': list(monitor.history), 'protocol': cfg.protocol, 'view_layout': 'independent', 'accepted_update_budget': cfg.steps, 'model_contract': cfg.model_contract(), 'recipe': cfg.recipe, 'loss_history': list(loss_history), 'initialization_checkpoint': cfg.init_checkpoint, 'unsafe_post_update_state': tag.startswith('CORRUPT_')}
        result = str(save_checkpoint(model, optimizer, ema, folder, rank=rank, world_size=world, sharded=sharded, metadata=metadata, rng_state=capture_rng_state()))
        error=[None]
        if rank==0:
            write_json(safe_pointer,{'checkpoint':result,'progress':dict(progress),'run_id':run_id})
            try:
                from .checkpoint_retention import prune_replaced_checkpoints
                prune_replaced_checkpoints(cfg,result)
            except Exception as exc:error[0]=f'{type(exc).__name__}:{exc}'
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        return result

    def restore(folder):
        metadata_before_load = json.loads((Path(folder) / 'COMPLETE.json').read_text())['metadata']
        if metadata_before_load.get('protocol') != cfg.protocol or metadata_before_load.get('model_contract') != cfg.model_contract() or metadata_before_load.get('recipe') != cfg.recipe or (metadata_before_load.get('accepted_update_budget') != cfg.steps):
            raise RuntimeError('Checkpoint architecture/protocol differs; start a new decoder-adapter run')
        if metadata_before_load.get('unsafe_post_update_state', False):
            raise RuntimeError('Refusing normal resume from a CORRUPT post-update checkpoint')
        state = load_checkpoint(model, optimizer, ema, folder, rank=rank, world_size=world, sharded=sharded)
        metadata = state['metadata']
        progress.clear()
        progress.update(metadata['progress'])
        monitor.history.clear()
        monitor.history.extend(metadata['gradient_history'])
        loss_history.clear()
        loss_history.extend(metadata.get('loss_history', []))
        restore_rng_state(state['rng_state'])
        optimizer.zero_grad(set_to_none=True)
    resume_policy = resume_decision(cfg, workspace, explicit=getattr(cfg, '_manual_resume', False), restarting=int(os.environ.get('DITGS_RECOVERY_RESTART', '0')) > 0)
    resume_path = resume_policy['checkpoint']
    if resume_path:
        restore(resume_path)
    elif cfg.recipe == 'joint':
        validate_warmup_initialization(cfg, cfg.init_checkpoint)
        metadata = load_model_weights(model, cfg.init_checkpoint, rank=rank, world_size=world, sharded=sharded, expected_contract=warmup_model_contract(cfg))
        progress['completed_steps'] = int(metadata['progress']['completed_steps'])
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in ema:
                    ema[name].copy_(parameter.detach())
        if rank == 0:
            write_json(workspace / 'warmup_initialization.json', {'checkpoint': cfg.init_checkpoint, 'source_progress': metadata['progress'], 'source_protocol': metadata['protocol'], 'target_protocol': cfg.protocol, 'new_accepted_update_budget': cfg.steps, 'source_accepted_updates': 10000, 'data_start_step': progress['completed_steps'], 'optimizer_restored': False, 'ema_restored': False, 'lr_clock_restored': False, 'learned_ca_kv_gamma_restored': True})
    if resume_path and rank == 0:
        write_json(safe_pointer, {'checkpoint': resume_path, 'progress': dict(progress), 'run_id': run_id})
    if cursor_path.exists() and resume_policy['use_failure_cursor']:
        progress['completed_steps'] = max(progress['completed_steps'], int(json.loads(cursor_path.read_text())['next_data_step']))
    dataset = make_dataset(cfg)
    if rank == 0:
        write_json(workspace / 'resolved_config.json', cfg.to_dict())
        write_json(workspace / 'runtime.json', {'run_id': run_id, 'torch': torch.__version__, 'rank_count': world, 'protocol': cfg.protocol, 'model_contract': cfg.model_contract(), 'parameter_storage': 'fp32', 'autocast': cfg.mixed_precision, 'distributed': cfg.distributed, 'scene_fingerprint': dataset.scene_fingerprint, 'runtime_validation': 'running_user_submitted_job', 'source_fingerprint': None, 'startup_preflight': False, 'ledger_policy': 'user_managed', 'backward_schedule': 'joint_graph', 'training_rollout': False, 'dit_forward_calls_per_batch': 1, 'feature_source': cfg.feature_source, 'view_fusion': cfg.view_fusion, 'sparse_encoder_policy': cfg.sparse_encoder_policy, 'source_attention': cfg.source_attention, 'data_recovery_policy': cfg.data_recovery_policy, 'gs_memory_tokens': 2 * (cfg.image_size // 8) * (cfg.image_width // 8), 'dit_memory_tokens': cfg.dit_memory_views * (cfg.image_size // 8) * (cfg.image_width // 8), 'kv_sharing_policy': cfg.kv_sharing_policy, 'dit_residual_scaling': cfg.dit_residual_scaling, 'dit_layerscale_initialization': cfg.dit_layerscale_initialization, 'residual_post_norm': cfg.residual_post_norm, 'original_gs_frozen': cfg.freeze_original_gs, 'initialization_checkpoint': cfg.init_checkpoint, 'continuation_initialization': 'corrected warmup final model/buffers only; fresh optimizer/EMA/LR' if cfg.recipe == 'joint' else 'original FlashWorld + TokenGS base1024; fresh adapter/CA copy-init', 'recovered_own_branch': bool(resume_path), 'backbone_source': cfg.backbone_source, 'flashworld_revision': cfg.flashworld_revision, 'wan_including_condition_frozen': cfg.freeze_dit and (not cfg.train_dit_condition), 'fm_optimization_weight': cfg.fm_weight, 'visibility_policy': cfg.visibility_policy, 'visibility_distance_scale': cfg.visibility_distance_scale, 'optimizer_group_lrs': {g['name']: g['base_lr'] for g in optimizer.param_groups}, 'phase_new_updates': cfg.steps, 'lineage_offset': cfg.lineage_offset, 'gs_gradient_counter_semantics': 'accepted update with ANY trainable GS group nonzero; all-source counter separate', 'residual_rms_scope': 'diagnostic only; never modifies branch amplitude', 'vae_decoder_frozen': True, 'vae_input_requires_grad_in_this_schedule': not cfg.freeze_dit, 'adapter_ray_encoding': cfg.ray_encoding, 'quality_scope': 'random_sigma_training_losses_not_sparse_only_benchmark', 'gradient_accumulation_semantics': 'one_joint_graph_one_dit_forward', 'autocast_weight_cache': not sharded, 'fsdp_backward_prefetch': None, 'fsdp_forward_prefetch': False if sharded else None, 'fsdp_limit_all_gathers': True if sharded else None, 'fsdp_local_pre_reduce_hooks': 'unavailable' if sharded else 'installed', 'runtime_optimization': 'batched-numerics-v1', 'steps_jsonl_are_sampled': True, 'authoritative_update_counters': 'training_health.json and checkpoint metadata.progress', 'normal_rank0_log_every': cfg.log_every, 'detailed_log_every': cfg.diagnostic_capture_every})

    def move(batch):
        host_keys = {'noise_seed', 'timestep_seed', 'valid_case'}
        return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) and key not in host_keys else value for key, value in batch.items()}

    def scalar(value):
        if isinstance(value, torch.Tensor):
            return value.flatten()[0].item()
        return value[0] if isinstance(value, (list, tuple)) else value

    def generators(batch):
        noise_seed, timestep_seed = (int(scalar(batch['noise_seed'])), int(scalar(batch['timestep_seed'])))
        return (torch.Generator(device=device).manual_seed(noise_seed % (2 ** 63 - 1)), torch.Generator(device=device).manual_seed(timestep_seed % (2 ** 63 - 1)))

    def forward(batch, step, component=None, capture_monitor=False):
        noise, timestep = generators(batch)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=not sharded) if cfg.mixed_precision == 'bf16' else nullcontext()
        with autocast:
            return model(batch, step=step, generator=noise, timestep_generator=timestep, loss_mode=component, capture_monitor=capture_monitor)

    def log_batch(batch, step, phase, *, force=False):
        if not force and step % cfg.diagnostic_capture_every != 0:
            return
        append_jsonl(workspace / 'batches' / f'{run_id}_rank{rank}.jsonl', {'phase': phase, 'step': step, 'metadata_json': batch['metadata_json']})

    def metrics(output):
        if 'metric_values' not in output:
            keys = list(output['losses'])
            scalars = [output['losses'][key].detach().float().reshape(()) for key in keys]
            scalars.append(output['loss'].detach().float().reshape(()))
            values = torch.stack(scalars).cpu().tolist()
            output['metric_values'] = dict(zip(keys, values[:-1]))
            output['total_loss_value'] = values[-1]
        return output['metric_values']

    def record_memory(stage, step, component=None, error=None):
        if error is None and (not (step < 0 or step % cfg.diagnostic_capture_every == 0)):
            return
        try:
            if stage == 'before_forward':
                torch.cuda.reset_peak_memory_stats(device)
            row = {'run_id': run_id, 'rank': rank, 'step': step, 'stage': stage, 'component': component, 'error': error, 'allocated_bytes': torch.cuda.memory_allocated(device), 'reserved_bytes': torch.cuda.memory_reserved(device), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            row['backward_schedule'] = 'joint_graph'
            row['last_vjp_stage'] = monitor.context.get('vjp_stage')
            if sharded:
                row['fsdp_storage'] = fsdp_storage_summary(model)
            if error is not None:
                row['activation_statistics'] = dict(monitor.activation_rows)
                row['first_anomaly'] = monitor.first_anomaly
            append_jsonl(workspace / 'numerics' / f'{run_id}_memory_rank{rank}.jsonl', row, force=error is not None)
        except Exception as memory_error:
            print(f'Could not record memory on rank{rank}: {memory_error}', flush=True)

    def fresh_backward(batch, step, component=None, *, batch_cpu):
        optimizer.zero_grad(set_to_none=True)
        monitor.begin_step(step, batch_cpu, int(scalar(batch_cpu['noise_seed'])))
        record_memory('before_forward', step, component)
        output = forward(batch, step, component)
        record_memory('after_forward', step, component)
        loss = output['loss']
        reasons = list(output.get('numerical_rejections', []))
        reasons.extend(loss_reasons(metrics(output), cfg, loss_history))
        finite = monitor.coordinated_finite(not reasons and math.isfinite(output['total_loss_value']))
        output['rejection_reasons'] = reasons
        loss.backward()
        record_memory('after_backward', step, component)
        return (output, finite)

    def update_lr(addressed_step):
        clock = progress['accepted_updates']
        warm = min(1.0, (clock + 1) / max(1, cfg.warmup_steps))
        fraction = max(0.0, min(1.0, (clock - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)))
        multiplier = warm if diagnostic else warm * 0.5 * (1 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * multiplier
    failure_snapshots = 0

    def run_step(batch_cpu, step, *, accept=True, phase='train'):
        nonlocal failure_snapshots
        batch = move(batch_cpu)
        log_batch(batch_cpu, step, phase)
        data_valid = bool(scalar(batch.get('valid_case', True)))
        if not monitor.coordinated_finite(data_valid):
            optimizer.zero_grad(set_to_none=True)
            record = {'step': step, 'accepted': False, 'all_finite': True, 'rejection_reasons': ['exhausted_training_scene_replacements'], 'local_valid_case': data_valid}
            log_batch(batch_cpu, step, phase, force=True)
            append_jsonl(workspace / 'steps' / f'{run_id}_rank{rank}.jsonl', record, force=True)
            return record
        try:
            output, finite_loss = fresh_backward(batch, step, batch_cpu=batch_cpu)
        except BaseException as error:
            record_memory('forward_or_backward_exception', step, error=f'{type(error).__name__}: {error}')
            try:
                monitor.save_failure_context(batch_cpu, 'forward_or_backward_exception', {'error': f'{type(error).__name__}: {error}', 'phase': phase})
            except BaseException as context_error:
                print(f'Could not save failure context on rank{rank}: {context_error}', flush=True)
            raise
        report = monitor.inspect_gradients(update_history=False)
        reasons = list(output['rejection_reasons']) + gradient_reasons(report, cfg)
        finite = monitor.coordinated_finite(finite_loss and (not reasons))
        numerically_finite = report['all_finite'] and math.isfinite(output['total_loss_value'])
        coverage = report['module_gradient_coverage']
        gs_groups = gs_gradient_groups(cfg)
        gs_has_gradient = cfg.use_gs and any((coverage[k]['global_any_nonzero_gradient'] for k in gs_groups))
        all_sources = cfg.use_gs and all((coverage[k]['global_any_nonzero_gradient'] for k in gs_groups))
        stats = output.get('gaussian_statistics', {})
        renderer_flags = torch.tensor([int(stats.get('renderer_image_gradient_l2', 0) > 0), int(stats.get('visibility_gradient_l2', 0) > 0)], device=device, dtype=torch.int32)
        dist.all_reduce(renderer_flags, op=dist.ReduceOp.MAX)
        renderer_nonzero, visibility_nonzero = map(bool, renderer_flags.cpu().tolist())
        record = {'step': step, 'phase': phase, 'accepted': False, 'all_finite': numerically_finite, 'update_permitted': finite, 'gs_has_nonzero_gradient': gs_has_gradient, 'gs_all_sources_nonzero': all_sources, 'renderer_has_nonzero_gradient': renderer_nonzero, 'visibility_has_nonzero_gradient': visibility_nonzero, 'zero_gradient_groups': [k for k in gs_groups if cfg.use_gs and (not coverage[k]['global_any_nonzero_gradient'])], 'gaussian_statistics': output.get('gaussian_statistics', {}), 'global_preclip_l2': report['global_l2'], 'spike': report['spike'], 'accepted_updates_before': progress['accepted_updates'], 'rejection_reasons': reasons, 'losses': metrics(output), 'valid_case': bool(scalar(batch.get('valid_case', True))), 'sample_timesteps': output.get('sample_timesteps'), 'source_attention': cfg.source_attention}
        if 'residual_rms' in output['diagnostics']:
            record['residual_rms_columns'] = list(output['diagnostics']['residual_rms_columns'])
            record['residual_rms'] = output['diagnostics']['residual_rms'].detach().cpu().tolist()
        if not finite and failure_snapshots < 4:
            record['failure_context'] = monitor.save_failure_context(batch_cpu, 'rejected_before_update', record)
            failure_snapshots += 1
        capture = False
        if accept and finite:
            clipped = monitor.clip_global(cfg.gradient_clip)
            record['clip_scale'] = clipped['clip_scale']
            finite = clipped['update_permitted']
            record['all_finite'] = record['all_finite'] and clipped['all_finite']
            record['update_permitted'] = finite
            if not finite:
                record['rejection_reasons'].append('nonfinite_after_clip')
                if failure_snapshots < 4:
                    record['failure_context'] = monitor.save_failure_context(batch_cpu, 'nonfinite_after_clip', record)
                    failure_snapshots += 1
            if finite:
                update_lr(step)
                optimizer.step()
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in ema:
                            ema[name].lerp_(parameter.detach(), 1 - cfg.ema_decay)
                states = monitor.capture_optimizer_state(optimizer, ema, detailed=capture or step % cfg.diagnostic_capture_every == 0)
                record_memory('after_optimizer', step)
                record['post_update_all_finite'] = states['all_finite']
                if not states['all_finite']:
                    record['update_permitted'] = False
                    record['state_validation'] = states
                    record['rejection_reasons'].append('invalid_post_update_state')
                    record['rolled_back_on_recovery'] = True
                    record['failure_context'] = monitor.save_failure_context(batch_cpu, 'nonfinite_after_update', record)
                    append_jsonl(workspace / 'steps' / f'{run_id}_rank{rank}.jsonl', record, force=True)
                    raise FloatingPointError('Invalid post-update state: supervisor restores latest safe checkpoint')
                progress['accepted_updates'] += 1
                record['accepted'] = True
        if record['accepted']:
            monitor.history.append(report['global_l2'])
            mean_loss = output['loss'].detach().float().clone()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            loss_history.append(float(mean_loss.item()) / world)
        optimizer.zero_grad(set_to_none=True)
        record['accepted_updates_after'] = progress['accepted_updates']
        if not record['accepted']:
            log_batch(batch_cpu, step, phase, force=True)
        if not record['accepted'] or step % cfg.diagnostic_capture_every == 0 or (rank == 0 and step % cfg.log_every == 0):
            append_jsonl(workspace / 'steps' / f'{run_id}_rank{rank}.jsonl', record, force=not record['accepted'])
        del output
        return record
    try:
        if not resume_path:
            save('initial')
        rejected_streak = 0
        last_log_time = time.monotonic()
        last_log_accepted = progress['accepted_updates']
        while progress['accepted_updates'] < cfg.steps:
            start = progress['completed_steps']
            loader = make_dataloader(cfg, rank, world, start_step=start, stop_step=start + 1000, dataset=dataset)
            iterator = iter(loader)
            for step in range(start, start + 1000):
                if progress['accepted_updates'] >= cfg.steps:
                    break
                if rank == 0:
                    write_json(cursor_path, {'next_data_step': step + 1, 'run_id': run_id})
                batch = next(iterator)
                record = run_step(batch, step)
                progress['completed_steps'] = step + 1
                health = update_health(progress, record, cfg.use_gs)
                collapse = collapse_reason(progress, cfg)
                if collapse:
                    writer.flush()
                    if rank == 0:
                        write_json(workspace / 'training_failed.json', {'reason': collapse, 'step': step, 'progress': dict(progress), 'health': health, 'action': 'stop this training; no automatic retry, no successful completion', 'latest_safe_is_numerical_only': True})
                        write_json(workspace / 'training_health.json', {'run_id': run_id, 'step': step, 'finished': False, **health, 'failed': True, 'failure_reason': collapse})
                    raise RuntimeError('Sustained GS/reconstruction-gradient collapse; inspect training_failed.json')
                if rank == 0 and (step % cfg.log_every == 0 or health['state'] != 'training' or progress['accepted_updates'] >= cfg.steps):
                    write_json(workspace / 'training_health.json', {'run_id': run_id, 'step': step, 'finished': progress['accepted_updates'] >= cfg.steps, **health})
                    if step % cfg.log_every == 0:
                        now = time.monotonic()
                        updates = progress['accepted_updates'] - last_log_accepted
                        print(json.dumps({'step': step, 'training_health': health, 'losses_rank0': record.get('losses', {}), 'global_preclip_l2': record.get('global_preclip_l2'), 'rejection_reasons': record.get('rejection_reasons', []), 'amortized_seconds_per_accepted_update': (now - last_log_time) / updates if updates else None, 'timing_interval_accepted_updates': updates}), flush=True)
                        last_log_time, last_log_accepted = (now, progress['accepted_updates'])
                if record['accepted']:
                    rejected_streak = 0
                    if progress['accepted_updates']<cfg.steps and progress['accepted_updates'] % cfg.save_every == 0:
                        save(f'update{progress['accepted_updates']}')
                else:
                    rejected_streak += 1
                    if rejected_streak % 32 == 0:
                        if rank == 0:
                            write_json(workspace / 'recovery_attention.json', {'state': 'repeated_rejections', 'consecutive_rejections': rejected_streak, 'progress': dict(progress), 'action': 'backoff_and_continue_with_new_addresses'})
                        time.sleep(cfg.recovery_backoff_seconds)
            del iterator, loader
        final_checkpoint = save('final')
        if rank == 0:
            write_json(workspace / 'completion.json', {'status': 'training_complete', 'progress': progress, 'checkpoint': final_checkpoint, 'protocol': cfg.protocol, 'model_contract': cfg.model_contract(), 'lineage_accepted_updates': cfg.lineage_offset + progress['accepted_updates'], 'claims': 'Camera-fixed adapter+GS warmup10k / subsequent joint20k; no benchmark verdict.'})
        dist.barrier()
    except Exception as error:
        write_json(workspace / 'recovery_requests' / f'rank{rank}.json', {'run_id': run_id, 'rank': rank, 'progress': dict(progress), 'error': f'{type(error).__name__}: {error}', 'action': 'stop_training_no_retry' if collapse_reason(progress, cfg) else 'restart_all_ranks_from_latest_safe'})
        raise
    finally:
        monitor.close()
        writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
if __name__ == '__main__':
    main()
