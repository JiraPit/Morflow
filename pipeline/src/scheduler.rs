use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use abi_stable::std_types::{RBox, RString, RVec, Tuple2};
use core_types::{
    scalar_number, tensor_scalar_text, ActionArgs, Audio, AudioChannelLayout, AudioLayout, Image,
    ImageLayout, Payload,
};
use parser::ast::{
    ActionCall, BinaryOp, Condition, FlowChain, FlowStep, SliceItem, Statement, Value, VarRef,
};
use rayon::prelude::*;

use crate::engine::MorflowError;
use crate::outputs::PipelineOutputs;
use crate::registry::LoadedAction;

/// A high-performance, DAG-driven execution scheduler that auto-parallelizes independent
/// flows in a pipeline and dynamically expands `each` loop iterations across Rayon's work-stealing pool.
pub struct AutoParallelScheduler {
    actions: HashMap<String, Arc<LoadedAction>>,
}

struct FlowTask {
    index: usize,
    statement: Statement,
    reads: HashSet<String>,
}

/// Zero-copy layered variable environment passed through parallel flow branches and loop slices.
pub enum EnvRef<'a> {
    Shared(&'a RwLock<HashMap<String, Payload>>),
    Layered(&'a LayeredEnv<'a>),
}

pub struct LayeredEnv<'a> {
    parent: &'a EnvRef<'a>,
    local_name: &'a str,
    local_val: &'a Payload,
    local_writes: RwLock<HashMap<String, Payload>>,
}

impl<'a> EnvRef<'a> {
    pub fn get(&self, name: &str) -> Option<Payload> {
        match self {
            EnvRef::Shared(rw) => rw.read().unwrap().get(name).cloned(),
            EnvRef::Layered(layered) => {
                if name == layered.local_name {
                    return Some(layered.local_val.clone());
                }
                if let Some(val) = layered.local_writes.read().unwrap().get(name) {
                    return Some(val.clone());
                }
                layered.parent.get(name)
            }
        }
    }

    pub fn insert(&self, name: String, payload: Payload) {
        match self {
            EnvRef::Shared(rw) => {
                rw.write().unwrap().insert(name, payload);
            }
            EnvRef::Layered(layered) => {
                layered.local_writes.write().unwrap().insert(name, payload);
            }
        }
    }
}

impl AutoParallelScheduler {
    pub fn new(actions: HashMap<String, Arc<LoadedAction>>) -> Self {
        Self { actions }
    }

    /// Executes the pipeline statements with automatic DAG dependency discovery and Rayon multi-core parallelism.
    /// Returns a `PipelineOutputs` containing all payloads emitted to the host.
    pub fn execute(
        &self,
        statements: &[Statement],
        initial_env: HashMap<String, Payload>,
    ) -> Result<PipelineOutputs, MorflowError> {
        let blackboard = RwLock::new(initial_env);
        let emitted_outputs = Mutex::new(HashMap::new());

        // 1. Analyze variable reads for all flows
        let mut pending_tasks: Vec<FlowTask> = statements
            .iter()
            .enumerate()
            .map(|(i, stmt)| {
                let reads = extract_dependencies(stmt);
                FlowTask {
                    index: i,
                    statement: stmt.clone(),
                    reads,
                }
            })
            .collect();

        // 2. DAG Scheduling loop: dynamically dispatch all ready flows in parallel
        while !pending_tasks.is_empty() {
            let (ready_tasks, remaining): (Vec<FlowTask>, Vec<FlowTask>) = {
                let env_guard = blackboard.read().unwrap();
                pending_tasks.into_iter().partition(|task| {
                    // A flow task is ready if all its required input variables are available in the blackboard
                    task.reads.iter().all(|var| env_guard.contains_key(var))
                })
            };

            if ready_tasks.is_empty() {
                return Err(MorflowError::Execution(
                    "Deadlock or unresolvable variable dependency among flows in pipeline"
                        .to_string(),
                ));
            }

            pending_tasks = remaining;

            // Execute all ready flows concurrently across Rayon threads
            let root_env = EnvRef::Shared(&blackboard);
            let results: Result<Vec<()>, MorflowError> = ready_tasks
                .into_par_iter()
                .map(|task| {
                    let Statement::Flow(chain) = &task.statement;
                    let mut step_res = None;

                    self.execute_flow_parallel(
                        chain,
                        task.index,
                        None,
                        &root_env,
                        &mut step_res,
                        Some(&emitted_outputs),
                    )?;

                    Ok(())
                })
                .collect();

            results?;
        }

        let outputs = emitted_outputs.into_inner().unwrap();
        Ok(PipelineOutputs::new(outputs))
    }

    #[allow(clippy::only_used_in_recursion)]
    fn execute_flow_parallel(
        &self,
        flow: &FlowChain,
        flow_idx: usize,
        initial: Option<Payload>,
        env: &EnvRef<'_>,
        step_result: &mut Option<Payload>,
        emitted_outputs: Option<&Mutex<HashMap<String, Payload>>>,
    ) -> Result<(), MorflowError> {
        let mut current: Option<Payload> = initial;

        for step in &flow.steps {
            match step {
                FlowStep::Var(var_ref) => {
                    let val = self.eval_var_ref(var_ref, env)?;
                    current = Some(val);
                }
                FlowStep::Tap(var_name) => {
                    if let Some(payload) = &current {
                        env.insert(var_name.clone(), payload.clone());
                    } else {
                        return Err(MorflowError::Execution(format!(
                            "Cannot tap into '${}': no active stream payload",
                            var_name
                        )));
                    }
                }
                FlowStep::Action(call) => {
                    if call.name == "emit" {
                        let Some(emitted_lock) = emitted_outputs else {
                            return Err(MorflowError::Execution(
                                "Emission ('emit') is not allowed inside an 'each' block or nested sub-flow".to_string(),
                            ));
                        };

                        let payload = current
                            .as_ref()
                            .ok_or_else(|| {
                                MorflowError::Execution(
                                    "Cannot emit: no active stream payload".to_string(),
                                )
                            })?
                            .clone();

                        let name = if let Some(val) = call.positional_args.first() {
                            match val {
                                Value::String(s) => s.clone(),
                                _ => {
                                    return Err(MorflowError::Execution(
                                        "emit argument must be a string name".to_string(),
                                    ))
                                }
                            }
                        } else if let Some((_, val)) =
                            call.named_args.iter().find(|(k, _)| k == "name")
                        {
                            match val {
                                Value::String(s) => s.clone(),
                                _ => {
                                    return Err(MorflowError::Execution(
                                        "emit name argument must be a string".to_string(),
                                    ))
                                }
                            }
                        } else {
                            String::new()
                        };

                        emitted_lock.lock().unwrap().insert(name, payload);
                        // Pass-through: retain `current` so downstream actions continue seamlessly
                        continue;
                    }

                    let (payload_in, args) =
                        { self.prepare_action_input(call, env, current.take())? };

                    let action = self.actions.get(&call.name).ok_or_else(|| {
                        MorflowError::Action(format!("Action '{}' was not preloaded", call.name))
                    })?;

                    let final_payload_in = if args.positional.is_empty() && args.named.is_empty() {
                        payload_in
                    } else {
                        Payload::WithArgs {
                            payload: RBox::new(payload_in),
                            args,
                        }
                    };

                    let out = action.process(final_payload_in);
                    if let Payload::Error(err) = &out {
                        return Err(MorflowError::Execution(format!(
                            "Action '{}' returned error: {}",
                            call.name, err
                        )));
                    }
                    current = Some(out);
                }
                FlowStep::IfElse(branch) => {
                    let cond_met = eval_condition(&branch.condition, env, current.as_ref());

                    let target_branch = if cond_met {
                        Some(&branch.then_branch)
                    } else {
                        branch.else_branch.as_ref()
                    };

                    if let Some(branch_stmts) = target_branch {
                        let mut branch_in = current.take();
                        for (sub_idx, sub_stmt) in branch_stmts.iter().enumerate() {
                            let Statement::Flow(sub_flow) = sub_stmt;
                            let mut sub_step_res = None;
                            self.execute_flow_parallel(
                                sub_flow,
                                flow_idx * 1000 + sub_idx,
                                branch_in.take(),
                                env,
                                &mut sub_step_res,
                                None,
                            )?;
                            branch_in = sub_step_res.take();
                        }
                        current = branch_in;
                    }
                }
                FlowStep::Each(each_loop) => {
                    let payload_in = current.take().ok_or_else(|| {
                        MorflowError::Execution("Loop 'each' requires an input tensor".to_string())
                    })?;

                    // Resolve the iteration axis from the payload variant so the
                    // loop iterates channels for images and multi-channel audio,
                    // and preserves the payload type on every iteration.
                    let (_, num_slices) = payload_in
                        .each_axis()
                        .map_err(|e| MorflowError::TypeMismatch(e.to_string()))?;

                    // AUTO-PARALLELIZATION:
                    // 1. Expand the loop into N independent slice execution sub-flows.
                    // 2. Dispatch them across Rayon's parallel thread pool with zero-copy layered env.
                    // 3. Independent iterations run 100% in parallel (no emission allowed inside each).
                    let slice_results: Result<Vec<Payload>, MorflowError> = (0..num_slices)
                        .into_par_iter()
                        .map(|i| {
                            // Zero-copy slice view for this iteration, carrying the
                            // original payload variant where the shape allows it.
                            let slice_payload = payload_in
                                .each_slice(i)
                                .map_err(|e| MorflowError::TypeMismatch(e.to_string()))?;

                            // The env holds the loop variable, so it needs its own
                            // handle to the same iteration payload.
                            let loop_val = slice_payload.clone();
                            let layered_env = LayeredEnv {
                                parent: env,
                                local_name: &each_loop.var_name,
                                local_val: &loop_val,
                                local_writes: RwLock::new(HashMap::new()),
                            };
                            let env_ref = EnvRef::Layered(&layered_env);

                            let mut iter_in = Some(slice_payload);
                            for (sub_idx, stmt) in each_loop.body.iter().enumerate() {
                                let Statement::Flow(sub_flow) = stmt;
                                let mut sub_step_res = None;
                                self.execute_flow_parallel(
                                    sub_flow,
                                    flow_idx * 10000 + i * 100 + sub_idx,
                                    iter_in.take(),
                                    &env_ref,
                                    &mut sub_step_res,
                                    None,
                                )?;
                                iter_in = sub_step_res.take();
                            }

                            iter_in.ok_or_else(|| {
                                MorflowError::Execution(format!(
                                    "Iteration {} in 'each' produced no output",
                                    i
                                ))
                            })
                        })
                        .collect();

                    let collected = slice_results?;

                    // Restack the iteration outputs into a single payload,
                    // preserving the variant and rejecting mismatched results
                    // rather than silently dropping iterations.
                    current = Some(
                        payload_in
                            .each_restack(&collected)
                            .map_err(|e| MorflowError::TypeMismatch(e.to_string()))?,
                    );
                }
                FlowStep::Route(route_block) => {
                    let mut matched_branch = None;
                    for arm in &route_block.arms {
                        if eval_condition(&arm.condition, env, current.as_ref()) {
                            matched_branch = Some(&arm.body);
                            break;
                        }
                    }
                    if matched_branch.is_none() {
                        matched_branch = route_block.default_arm.as_ref();
                    }

                    if let Some(branch_stmts) = matched_branch {
                        let mut branch_in = current.take();
                        for (sub_idx, sub_stmt) in branch_stmts.iter().enumerate() {
                            let Statement::Flow(sub_flow) = sub_stmt;
                            let mut sub_step_res = None;
                            self.execute_flow_parallel(
                                sub_flow,
                                flow_idx * 1000 + sub_idx,
                                branch_in.take(),
                                env,
                                &mut sub_step_res,
                                None,
                            )?;
                            branch_in = sub_step_res.take();
                        }
                        current = branch_in;
                    }
                }
            }
        }

        *step_result = current;
        Ok(())
    }

    fn prepare_action_input(
        &self,
        call: &ActionCall,
        env: &EnvRef<'_>,
        current: Option<Payload>,
    ) -> Result<(Payload, ActionArgs), MorflowError> {
        let mut positional_strs = Vec::new();
        let mut named_tuples = Vec::new();
        let mut resolved_input = current;

        for (i, val) in call.positional_args.iter().enumerate() {
            if i == 0 && resolved_input.is_none() {
                if let Value::Var(var_ref) = val {
                    resolved_input = Some(self.eval_var_ref(var_ref, env)?);
                    continue;
                }
            }
            positional_strs.push(self.eval_value_to_string(val, env));
        }

        for (key, val) in &call.named_args {
            let str_val = self.eval_value_to_string(val, env);
            named_tuples.push(Tuple2(RString::from(key.clone()), RString::from(str_val)));
        }

        let input_payload = resolved_input.unwrap_or_else(|| Payload::Data {
            buffer: RVec::new(),
        });

        let args = ActionArgs {
            positional: RVec::from(
                positional_strs
                    .into_iter()
                    .map(RString::from)
                    .collect::<Vec<_>>(),
            ),
            named: RVec::from(named_tuples),
        };

        Ok((input_payload, args))
    }

    fn eval_var_ref(&self, var_ref: &VarRef, env: &EnvRef<'_>) -> Result<Payload, MorflowError> {
        let mut payload = env.get(&var_ref.name).ok_or_else(|| {
            MorflowError::Execution(format!(
                "Variable '${}' not found in environment",
                var_ref.name
            ))
        })?;
        for slices in &var_ref.slices {
            payload = Self::slice_payload(payload, slices, &var_ref.name)?;
        }
        Ok(payload)
    }

    fn slice_payload(
        mut base_payload: Payload,
        slices: &[SliceItem],
        name: &str,
    ) -> Result<Payload, MorflowError> {
        let mut slices = slices.to_vec();
        if slices.is_empty() {
            return Ok(base_payload);
        }

        while let Payload::Composite(items) = base_payload.unwrap_payload() {
            let Some(slice) = slices.first() else {
                break;
            };
            let SliceItem::Index(index) = slice else {
                return Err(MorflowError::Execution(format!(
                    "Composite ${} supports integer indexes only",
                    name
                )));
            };
            let index = usize::try_from(*index).map_err(|_| {
                MorflowError::Execution("Composite indexes must be nonnegative".into())
            })?;
            base_payload = items.get(index).cloned().ok_or_else(|| {
                MorflowError::Execution(format!(
                    "Composite index {index} out of bounds for {} components in ${}",
                    items.len(),
                    name
                ))
            })?;
            slices.remove(0);
        }
        if slices.is_empty() {
            return Ok(base_payload);
        }
        let base_payload = base_payload.unwrap_payload().clone();
        match base_payload {
            Payload::Tensor(tensor) => {
                let mut sliced = tensor;
                let mut current_axis = 0usize;

                for slice_item in &slices {
                    match slice_item {
                        SliceItem::NamedDim { dim_name, index } => {
                            let axis = match dim_name.as_str() {
                                "dim" | "axis" => *index as usize,
                                _ => 0,
                            };
                            sliced = sliced
                                .slice_axis_index(axis, *index as usize)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                        }
                        SliceItem::Range { start, end, step } => {
                            let s = start.unwrap_or(0) as usize;
                            let e = end.map(|x| x as usize).unwrap_or_else(|| {
                                sliced.shape.get(current_axis).copied().unwrap_or(0)
                            });
                            let st = step.unwrap_or(1) as usize;
                            sliced = sliced
                                .slice_range(current_axis, s, e, st)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                            current_axis += 1;
                        }
                        SliceItem::Index(idx) => {
                            sliced = sliced
                                .slice_axis_index(current_axis, *idx as usize)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                        }
                        SliceItem::Full => {
                            current_axis += 1;
                        }
                    }
                }
                Ok(Payload::from_tensor(sliced))
            }
            Payload::Image(image) => {
                let mut sliced = image.tensor;
                let mut current_axis = 0usize;

                for slice_item in &slices {
                    match slice_item {
                        SliceItem::NamedDim { dim_name, index } => {
                            let axis = match dim_name.as_str() {
                                "dim" | "axis" => *index as usize,
                                "y" | "row" | "height" => match image.layout {
                                    ImageLayout::Hwc => 0,
                                    ImageLayout::Chw => 1,
                                },
                                "x" | "col" | "width" => match image.layout {
                                    ImageLayout::Hwc => 1,
                                    ImageLayout::Chw => 2,
                                },
                                "c" | "channel" | "channels" => match image.layout {
                                    ImageLayout::Hwc => 2,
                                    ImageLayout::Chw => 0,
                                },
                                _ => 0,
                            };
                            sliced = sliced
                                .slice_axis_index(axis, *index as usize)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                        }
                        SliceItem::Range { start, end, step } => {
                            let s = start.unwrap_or(0) as usize;
                            let e = end.map(|x| x as usize).unwrap_or_else(|| {
                                sliced.shape.get(current_axis).copied().unwrap_or(0)
                            });
                            let st = step.unwrap_or(1) as usize;
                            sliced = sliced
                                .slice_range(current_axis, s, e, st)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                            current_axis += 1;
                        }
                        SliceItem::Index(idx) => {
                            sliced = sliced
                                .slice_axis_index(current_axis, *idx as usize)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                        }
                        SliceItem::Full => {
                            current_axis += 1;
                        }
                    }
                }

                // Slicing an image must leave something an Image can still hold,
                // so a slice that drops below rank 2 is refused rather than
                // silently changing the payload's type.
                match Image::new(sliced.clone(), image.color_space, image.layout) {
                    Ok(img) => Ok(Payload::Image(img)),
                    Err(e) => Err(MorflowError::TypeMismatch(format!(
                        "Cannot slice {}: {}",
                        name, e
                    ))),
                }
            }
            Payload::Audio(audio) => {
                let mut sliced = audio.tensor;
                let mut current_axis = 0usize;

                for slice_item in &slices {
                    match slice_item {
                        SliceItem::NamedDim { dim_name, index } => {
                            let axis = match dim_name.as_str() {
                                "dim" | "axis" => *index as usize,
                                "ch" | "channel" | "channels" => match audio.layout {
                                    AudioLayout::Planar => 0,
                                    AudioLayout::Interleaved => 1,
                                },
                                "t" | "time" | "sample" | "samples" => match audio.layout {
                                    AudioLayout::Planar => 1,
                                    AudioLayout::Interleaved => 0,
                                },
                                _ => 0,
                            };
                            sliced = sliced
                                .slice_axis_index(axis, *index as usize)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                        }
                        SliceItem::Range { start, end, step } => {
                            let s = start.unwrap_or(0) as usize;
                            let e = end.map(|x| x as usize).unwrap_or_else(|| {
                                sliced.shape.get(current_axis).copied().unwrap_or(0)
                            });
                            let st = step.unwrap_or(1) as usize;
                            sliced = sliced
                                .slice_range(current_axis, s, e, st)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                            current_axis += 1;
                        }
                        SliceItem::Index(idx) => {
                            sliced = sliced
                                .slice_axis_index(current_axis, *idx as usize)
                                .map_err(|e| MorflowError::Execution(e.to_string()))?;
                        }
                        SliceItem::Full => {
                            current_axis += 1;
                        }
                    }
                }

                let rank = sliced.rank();
                // A single sample is a rank-0 value, which Audio cannot hold, so
                // it becomes a Scalar rather than changing the type to a tensor.
                if rank == 0 {
                    return Ok(Payload::Scalar(sliced));
                }
                let ch_count = match (rank, audio.layout) {
                    (1, _) => 1,
                    (2, AudioLayout::Planar) => sliced.shape[0],
                    (2, AudioLayout::Interleaved) => sliced.shape[1],
                    _ => 0,
                };

                if ch_count > 0 {
                    let channel_layout = AudioChannelLayout::from_channel_count(ch_count);
                    if let Ok(aud) = Audio::new(
                        sliced.clone(),
                        audio.sample_rate,
                        channel_layout,
                        audio.layout,
                    ) {
                        return Ok(Payload::Audio(aud));
                    }
                }
                Err(MorflowError::TypeMismatch(format!(
                    "Cannot slice ${}: the result has rank {} and {} channels, which is not a valid Audio payload",
                    name, rank, ch_count
                )))
            }
            Payload::Data { buffer } => {
                if slices.len() != 1 {
                    return Err(MorflowError::Execution(
                        "Bytes support one range slice per bracket group".into(),
                    ));
                }
                if let Some(SliceItem::Range {
                    start,
                    end,
                    step: _,
                }) = slices.first()
                {
                    let s = start.unwrap_or(0) as usize;
                    let e = end
                        .map(|x| x as usize)
                        .unwrap_or(buffer.len())
                        .min(buffer.len());
                    if s <= e && s <= buffer.len() {
                        let sub_buf = buffer[s..e].to_vec();
                        Ok(Payload::Data {
                            buffer: RVec::from(sub_buf),
                        })
                    } else {
                        Err(MorflowError::Execution(format!(
                            "Slice out of bounds for buffer of len {}",
                            buffer.len()
                        )))
                    }
                } else {
                    Err(MorflowError::Execution(
                        "Bytes support range slices only".into(),
                    ))
                }
            }
            // Arguments are plain values and scalars are rank-0, so neither has
            // a dimension to slice.
            Payload::Arg(_) | Payload::Scalar(_) => Err(MorflowError::TypeMismatch(format!(
                "Cannot slice ${}: it holds a value, not a payload",
                name
            ))),
            other => Err(MorflowError::TypeMismatch(format!(
                "Cannot index {}",
                core_types::payload_kind_name(&other)
            ))),
        }
    }

    fn eval_value_to_string(&self, val: &Value, env: &EnvRef<'_>) -> String {
        match val {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Var(v) => {
                if let Some(payload) = env.get(&v.name) {
                    match payload.unwrap_payload() {
                        Payload::Data { buffer } => {
                            String::from_utf8_lossy(buffer.as_slice()).to_string()
                        }
                        Payload::Arg(bytes) => {
                            String::from_utf8_lossy(bytes.as_slice()).to_string()
                        }
                        Payload::Scalar(t) => tensor_scalar_text(t),
                        Payload::Tensor(t) => {
                            format!("<Tensor shape={:?}>", t.shape.as_slice())
                        }
                        Payload::Image(img) => {
                            format!(
                                "<Image {}x{}, {:?}, {:?}>",
                                img.width(),
                                img.height(),
                                img.color_space,
                                img.dtype()
                            )
                        }
                        Payload::Audio(aud) => {
                            format!(
                                "<Audio {}ch, {}Hz, {:.2}s, {:?}>",
                                aud.channels(),
                                aud.sample_rate,
                                aud.duration_seconds(),
                                aud.dtype()
                            )
                        }
                        _ => "<Payload>".to_string(),
                    }
                } else {
                    format!("${}", v.name)
                }
            }
        }
    }
}

pub(crate) fn extract_dependencies(stmt: &Statement) -> HashSet<String> {
    let mut reads = HashSet::new();

    let Statement::Flow(chain) = stmt;
    for step in &chain.steps {
        match step {
            FlowStep::Var(var_ref) => {
                reads.insert(var_ref.name.clone());
            }
            FlowStep::Tap(_) => {}
            FlowStep::Action(call) => {
                for arg in &call.positional_args {
                    if let Value::Var(v) = arg {
                        reads.insert(v.name.clone());
                    }
                }
                for (_, arg) in &call.named_args {
                    if let Value::Var(v) = arg {
                        reads.insert(v.name.clone());
                    }
                }
            }
            FlowStep::IfElse(branch) => {
                if let Value::Var(v) = &branch.condition.left {
                    reads.insert(v.name.clone());
                }
                if let Value::Var(v) = &branch.condition.right {
                    reads.insert(v.name.clone());
                }
                for sub_stmt in &branch.then_branch {
                    let sub_reads = extract_dependencies(sub_stmt);
                    reads.extend(sub_reads);
                }
                if let Some(else_branch) = &branch.else_branch {
                    for sub_stmt in else_branch {
                        let sub_reads = extract_dependencies(sub_stmt);
                        reads.extend(sub_reads);
                    }
                }
            }
            FlowStep::Each(each_loop) => {
                for sub_stmt in &each_loop.body {
                    let sub_reads = extract_dependencies(sub_stmt);
                    for r in sub_reads {
                        if r != each_loop.var_name {
                            reads.insert(r);
                        }
                    }
                }
            }
            FlowStep::Route(route) => {
                for arm in &route.arms {
                    if let Value::Var(v) = &arm.condition.left {
                        reads.insert(v.name.clone());
                    }
                    if let Value::Var(v) = &arm.condition.right {
                        reads.insert(v.name.clone());
                    }
                    for sub_stmt in &arm.body {
                        let sub_reads = extract_dependencies(sub_stmt);
                        reads.extend(sub_reads);
                    }
                }
                if let Some(default_arm) = &route.default_arm {
                    for sub_stmt in default_arm {
                        let sub_reads = extract_dependencies(sub_stmt);
                        reads.extend(sub_reads);
                    }
                }
            }
        }
    }

    reads
}

pub(crate) fn extract_writes(stmt: &Statement) -> HashSet<String> {
    let mut writes = HashSet::new();

    let Statement::Flow(chain) = stmt;
    for step in &chain.steps {
        match step {
            FlowStep::Tap(name) => {
                writes.insert(name.clone());
            }
            FlowStep::IfElse(branch) => {
                for sub_stmt in &branch.then_branch {
                    writes.extend(extract_writes(sub_stmt));
                }
                if let Some(else_branch) = &branch.else_branch {
                    for sub_stmt in else_branch {
                        writes.extend(extract_writes(sub_stmt));
                    }
                }
            }
            FlowStep::Each(each_loop) => {
                for sub_stmt in &each_loop.body {
                    writes.extend(extract_writes(sub_stmt));
                }
            }
            FlowStep::Route(route) => {
                for arm in &route.arms {
                    for sub_stmt in &arm.body {
                        writes.extend(extract_writes(sub_stmt));
                    }
                }
                if let Some(default_arm) = &route.default_arm {
                    for sub_stmt in default_arm {
                        writes.extend(extract_writes(sub_stmt));
                    }
                }
            }
            _ => {}
        }
    }

    writes
}

fn eval_condition(cond: &Condition, env: &EnvRef<'_>, current: Option<&Payload>) -> bool {
    let left_num = resolve_number_or_field(&cond.left, env, current);
    let right_num = resolve_number_or_field(&cond.right, env, current);

    if let (Some(l), Some(r)) = (left_num, right_num) {
        match cond.op {
            BinaryOp::Eq => (l - r).abs() < f64::EPSILON,
            BinaryOp::NotEq => (l - r).abs() >= f64::EPSILON,
            BinaryOp::Lt => l < r,
            BinaryOp::LtEq => l <= r,
            BinaryOp::Gt => l > r,
            BinaryOp::GtEq => l >= r,
        }
    } else {
        true
    }
}

fn resolve_number_or_field(
    val: &Value,
    env: &EnvRef<'_>,
    current: Option<&Payload>,
) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::Var(var_ref) => {
            let payload_opt = if var_ref.name.is_empty() || var_ref.name == "payload" {
                current.cloned()
            } else {
                env.get(&var_ref.name)
            };

            if let Some(payload) = payload_opt {
                match var_ref.field.as_deref() {
                    Some("sample_rate") | Some("rate") => match payload.unwrap_payload() {
                        Payload::Audio(aud) => Some(aud.sample_rate as f64),
                        _ => None,
                    },
                    Some("duration") | Some("duration_seconds") => match payload.unwrap_payload() {
                        Payload::Audio(aud) => Some(aud.duration_seconds()),
                        _ => None,
                    },
                    Some("num_samples") | Some("samples") => match payload.unwrap_payload() {
                        Payload::Audio(aud) => Some(aud.num_samples() as f64),
                        _ => None,
                    },
                    Some("width") => match payload.unwrap_payload() {
                        Payload::Image(img) => Some(img.width() as f64),
                        Payload::Tensor(t) if t.rank() >= 2 => Some(t.shape[1] as f64),
                        _ => None,
                    },
                    Some("height") => match payload.unwrap_payload() {
                        Payload::Image(img) => Some(img.height() as f64),
                        Payload::Tensor(t) if t.rank() >= 2 => Some(t.shape[0] as f64),
                        _ => None,
                    },
                    Some("channels") => match payload.unwrap_payload() {
                        Payload::Audio(aud) => Some(aud.channels() as f64),
                        Payload::Image(img) => Some(img.channels() as f64),
                        Payload::Tensor(t) if t.rank() >= 3 => Some(t.shape[2] as f64),
                        _ => None,
                    },
                    Some("peak") | Some("peak_abs") => match payload.unwrap_payload() {
                        Payload::Tensor(t) | Payload::Scalar(t) => Some(t.peak_abs()),
                        Payload::Image(img) => Some(img.tensor.peak_abs()),
                        Payload::Audio(aud) => Some(aud.tensor.peak_abs()),
                        _ => None,
                    },
                    Some("rms") => match payload.unwrap_payload() {
                        Payload::Tensor(t) | Payload::Scalar(t) => Some(t.rms()),
                        Payload::Image(img) => Some(img.tensor.rms()),
                        Payload::Audio(aud) => Some(aud.tensor.rms()),
                        _ => None,
                    },
                    Some("mean") => match payload.unwrap_payload() {
                        Payload::Tensor(t) | Payload::Scalar(t) => Some(t.mean()),
                        Payload::Image(img) => Some(img.tensor.mean()),
                        Payload::Audio(aud) => Some(aud.tensor.mean()),
                        _ => None,
                    },
                    Some("len") | Some("length") => match payload.unwrap_payload() {
                        Payload::Tensor(t) | Payload::Scalar(t) => Some(t.num_elements() as f64),
                        Payload::Image(img) => Some(img.tensor.num_elements() as f64),
                        Payload::Audio(aud) => Some(aud.tensor.num_elements() as f64),
                        Payload::Data { buffer } => Some(buffer.len() as f64),
                        Payload::Arg(bytes) => Some(bytes.len() as f64),
                        _ => None,
                    },
                    _ => match payload.unwrap_payload() {
                        Payload::Data { buffer } => {
                            if let Ok(s) = std::str::from_utf8(buffer.as_slice()) {
                                s.trim().parse::<f64>().ok()
                            } else if buffer.len() == 4 {
                                Some(f32::from_ne_bytes(buffer.as_slice().try_into().unwrap())
                                    as f64)
                            } else if buffer.len() == 8 {
                                Some(f64::from_ne_bytes(buffer.as_slice().try_into().unwrap()))
                            } else {
                                None
                            }
                        }
                        // An argument parameter is readable as a number when it
                        // holds one.
                        Payload::Arg(bytes) => std::str::from_utf8(bytes.as_slice())
                            .ok()
                            .and_then(|s| s.trim().parse::<f64>().ok()),
                        // A scalar is a single number of whatever dtype it holds.
                        Payload::Scalar(t) => scalar_number(t),
                        _ => None,
                    },
                }
            } else {
                None
            }
        }
        _ => None,
    }
}
