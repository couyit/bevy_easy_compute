use core::panic;
use std::{
    marker::PhantomData,
    ops::Deref,
    time::{Duration, SystemTime},
};

use bevy::{
    prelude::{Res, ResMut, Resource},
    render::{
        render_resource::{Buffer, ComputePipeline},
        renderer::{RenderDevice, RenderQueue},
    },
    utils::{hashbrown::hash_map::Entry, HashMap},
};
use bytemuck::{bytes_of, cast_slice, from_bytes, AnyBitPattern, NoUninit};
use wgpu::{
    BindGroupEntry, BindingResource, CommandEncoder, CommandEncoderDescriptor,
    ComputePassDescriptor, TextureViewDescriptor,
};

use crate::{
    error::{Error, Result},
    pipeline_cache::{AppPipelineCache, CachedAppComputePipelineId},
    traits::ComputeWorker,
    worker_builder::{AppComputeWorkerBuilder, BindingData},
};

#[derive(PartialEq, Clone, Copy)]
pub enum RunMode {
    Continuous,
    OneShot(bool),
}

#[derive(PartialEq)]
pub enum WorkerState {
    Created,
    Available,
    Working { start_time: SystemTime },
    FinishedWorking,
}

#[derive(Clone, Debug)]
pub(crate) enum Step {
    ComputePass(ComputePass),
    Swap(String, String),
}

#[derive(Clone, Debug)]
pub(crate) struct ComputePass {
    pub(crate) workgroups: [u32; 3],
    pub(crate) vars: Vec<String>,
    pub(crate) shader_type_path: String,
}

#[derive(Clone, Debug)]
pub(crate) struct StagingBuffer {
    pub(crate) mapped: bool,
    pub(crate) buffer: Buffer,
}

/// Struct to manage data transfers from/to the GPU
/// it also handles the logic of your compute work.
/// By default, the run mode of the workers is set to continuous,
/// meaning it will run every frames. If you want to run it deterministically
/// use the function `one_shot()` in the builder
#[derive(Resource)]
pub struct AppComputeWorker<W: ComputeWorker> {
    pub(crate) state: WorkerState,
    render_device: RenderDevice,
    render_queue: RenderQueue,
    cached_pipeline_ids: HashMap<String, CachedAppComputePipelineId>,
    pipelines: HashMap<String, Option<ComputePipeline>>,
    binding_data: HashMap<String, BindingData>,
    steps: Vec<Step>,
    command_encoder: Option<CommandEncoder>,
    run_mode: RunMode,
    submission_queue_processed: bool,
    /// Maximum duration the compute shader will run asyncronously before being set to synchronous.
    ///
    /// 0 seconds means the shader will immediately be polled synchronously. None emeans the shader will only run asynchronously.
    maximum_async_time: Option<Duration>,
    _phantom: PhantomData<W>,
}

impl<W: ComputeWorker> From<&AppComputeWorkerBuilder<'_, W>> for AppComputeWorker<W> {
    /// Create a new [`AppComputeWorker<W>`].
    fn from(builder: &AppComputeWorkerBuilder<W>) -> Self {
        let render_device = builder.world.resource::<RenderDevice>().clone();
        let render_queue = builder.world.resource::<RenderQueue>().clone();

        let pipelines = builder
            .cached_pipeline_ids
            .iter()
            .map(|(type_path, _)| (type_path.clone(), None))
            .collect();

        let command_encoder =
            Some(render_device.create_command_encoder(&CommandEncoderDescriptor { label: None }));

        Self {
            state: WorkerState::Created,
            render_device,
            render_queue,
            cached_pipeline_ids: builder.cached_pipeline_ids.clone(),
            pipelines,
            binding_data: builder.binding_data.clone(),
            steps: builder.steps.clone(),
            command_encoder,
            run_mode: builder.run_mode,
            _phantom: PhantomData,
            maximum_async_time: builder.maximum_async_time,
            submission_queue_processed: false,
        }
    }
}

impl<W: ComputeWorker> AppComputeWorker<W> {
    #[inline]
    fn dispatch(&mut self, index: usize) -> Result<()> {
        let compute_pass = match &self.steps[index] {
            Step::ComputePass(compute_pass) => compute_pass,
            Step::Swap(_, _) => return Err(Error::InvalidStep(format!("{:?}", self.steps[index]))),
        };

        let mut entries = vec![];

        for (index, var) in compute_pass.vars.iter().enumerate() {
            let Some(data) = self.binding_data.get(var) else {
                return Err(Error::BufferNotFound(var.to_owned()));
            };

            let entry = match data {
                BindingData::Buffer(buffer) => BindGroupEntry {
                    binding: index as u32,
                    resource: buffer.as_entire_binding(),
                },
                BindingData::Sampler(sampler) => BindGroupEntry {
                    binding: index as u32,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindingData::TextureView(view) => BindGroupEntry {
                    binding: index as u32,
                    resource: BindingResource::TextureView(&view),
                },
                _ => unreachable!(),
            };
            entries.push(entry);
        }

        let Some(maybe_pipeline) = self.pipelines.get(&compute_pass.shader_type_path) else {
            return Err(Error::PipelinesEmpty);
        };

        let Some(pipeline) = maybe_pipeline else {
            return Err(Error::PipelineNotReady);
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group =
            self.render_device
                .create_bind_group(None, &bind_group_layout.into(), &entries);

        let Some(encoder) = &mut self.command_encoder else {
            return Err(Error::EncoderIsNone);
        };
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                compute_pass.workgroups[0],
                compute_pass.workgroups[1],
                compute_pass.workgroups[2],
            )
        }

        Ok(())
    }

    #[inline]
    fn swap(&mut self, index: usize) -> Result<()> {
        let data_name = match &self.steps[index] {
            Step::ComputePass(_) => {
                return Err(Error::InvalidStep(format!("{:?}", self.steps[index])))
            }
            Step::Swap(a, b) => [a.as_str(), b.as_str()],
        };

        for key in data_name {
            if !self.binding_data.contains_key(key) {
                return Err(Error::BufferNotFound(key.to_string()));
            }
        }

        let data_array = self.binding_data.get_many_mut(data_name).unwrap();

        for (name, data) in data_name.into_iter().zip(data_array.iter()) {
            match data {
                BindingData::Buffer(_) | BindingData::TextureView(_) => {}
                _ => {
                    return Err(Error::BufferNotFound(name.to_string()));
                }
            }
        }

        let [data_a, data_b] = data_array;

        std::mem::swap(data_a, data_b);

        Ok(())
    }

    fn submit(&mut self) -> &mut Self {
        let encoder = self.command_encoder.take().unwrap();
        self.render_queue.submit(Some(encoder.finish()));
        self.state = WorkerState::Working {
            start_time: SystemTime::now(),
        };
        self
    }

    #[inline]
    fn poll(&mut self) -> bool {
        let WorkerState::Working { start_time } = self.state else {
            // We will always be in this state at this point
            panic!("Called AppComputeWorker::poll() without being in the Working state!");
        };

        let is_async = self
            .maximum_async_time
            .map(|x| {
                SystemTime::now()
                    .duration_since(start_time)
                    .unwrap_or_default()
                    < x
            })
            .unwrap_or(true);

        if is_async {
            match self
                .render_device
                .wgpu_device()
                .poll(wgpu::MaintainBase::Poll)
            {
                // The first few times the poll occurs the queue will be empty, because wgpu hasn't started anything yet.
                // We need to wait until `MaintainResult::Ok`, which means wgpu has started to process our data.
                // Then, the next time the queue is empty (`MaintainResult::SubmissionQueueEmpty`), wgpu has finished processing the data and we are done.
                wgpu::MaintainResult::SubmissionQueueEmpty => {
                    let res = self.submission_queue_processed;
                    self.submission_queue_processed = false;
                    res
                }
                wgpu::MaintainResult::Ok => {
                    self.submission_queue_processed = true;
                    false
                }
            }
        } else {
            match self
                .render_device
                .wgpu_device()
                .poll(wgpu::MaintainBase::Wait)
            {
                wgpu::MaintainResult::SubmissionQueueEmpty => true,
                wgpu::MaintainResult::Ok => false,
            }
        }
    }

    /// Check if the worker is ready to be read from.
    #[inline]
    pub fn ready(&self) -> bool {
        self.state == WorkerState::FinishedWorking
    }

    /// Tell the worker to execute the compute shader at the end of the current frame
    #[inline]
    pub fn execute(&mut self) {
        match self.run_mode {
            RunMode::Continuous => {}
            RunMode::OneShot(_) => self.run_mode = RunMode::OneShot(true),
        }
    }

    #[inline]
    fn ready_to_execute(&self) -> bool {
        (!matches!(self.state, WorkerState::Working { start_time: _ }))
            && (self.run_mode != RunMode::OneShot(false))
    }

    pub(crate) fn run(mut worker: ResMut<Self>) {
        if worker.ready() {
            worker.state = WorkerState::Available;
        }

        if worker.ready_to_execute() {
            // Workaround for interior mutability
            for i in 0..worker.steps.len() {
                let result = match worker.steps[i] {
                    Step::ComputePass(_) => worker.dispatch(i),
                    Step::Swap(_, _) => worker.swap(i),
                };

                if let Err(err) = result {
                    match err {
                        Error::PipelineNotReady => return,
                        _ => panic!("{:?}", err),
                    }
                }
            }

            // worker.read_staging_buffers().unwrap();
            worker.submit();
            // worker.map_staging_buffers();
        }

        if worker.run_mode != RunMode::OneShot(false) && worker.poll() {
            // for (_, staging_buffer) in worker.staging_buffers.iter_mut() {
            //     // By this the staging buffers would've been mapped.
            //     staging_buffer.mapped = true;
            // }

            worker.state = WorkerState::FinishedWorking;
            worker.command_encoder = Some(
                worker
                    .render_device
                    .create_command_encoder(&CommandEncoderDescriptor { label: None }),
            );

            match worker.run_mode {
                RunMode::Continuous => {}
                RunMode::OneShot(_) => worker.run_mode = RunMode::OneShot(false),
            };
        }
    }

    pub(crate) fn extract_pipelines(
        mut worker: ResMut<Self>,
        pipeline_cache: Res<AppPipelineCache>,
    ) {
        for (type_path, cached_id) in &worker.cached_pipeline_ids.clone() {
            let Some(pipeline) = worker.pipelines.get(type_path) else {
                continue;
            };

            if pipeline.is_some() {
                continue;
            };

            let cached_id = *cached_id;

            worker.pipelines.insert(
                type_path.clone(),
                pipeline_cache.get_compute_pipeline(cached_id).cloned(),
            );
        }
    }
}
