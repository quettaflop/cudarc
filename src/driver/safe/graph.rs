use std::sync::Arc;

use crate::driver::{result, sys};

use super::{capture_pool::PoolCapture, CudaStream, DriverError};

/// Represents a replay-able Cuda Graph. Create with [CudaStream::begin_capture()] and [CudaStream::end_capture()].
///
/// Once created you can replay with [CudaGraph::launch()].
///
/// # On Thread safety
///
/// This object is **NOT** thread safe.
///
/// From official docs:
///
/// > Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed concurrently from multiple threads. API calls accessing the same graph object must be serialized externally.
/// >
/// > Note that this includes APIs which may appear to be read-only, such as cudaGraphClone() (cuGraphClone()) and cudaGraphInstantiate() (cuGraphInstantiate()). No API or pair of APIs is guaranteed to be safe to call on the same graph object from two different threads without serialization.
///
/// <https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety>
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    /// The capture into a [CapturePool](super::CapturePool) that recorded this graph, whose
    /// memory it uses.
    pool_capture: Option<PoolCapture>,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let ctx = &self.stream.ctx;

        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            ctx.record_err(unsafe { result::graph::exec_destroy(cu_graph_exec) });
        }

        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            ctx.record_err(unsafe { result::graph::destroy(cu_graph) });
        }

        // A replay still running is ordered before anything later on the pool's stream.
        if let Some(capture) = self.pool_capture.take() {
            capture.graph_dropped();
        }
    }
}

impl CudaStream {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143)
    pub fn begin_capture(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::begin_capture(self.cu_stream, mode) }?;
        // A capture into a pool that was not ended through end_capture is over.
        if let Some(stale) = self.end_pool_capture() {
            stale.end(false);
        }
        Ok(())
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c)
    ///
    /// `flags` is passed to [cuGraphInstantiate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1)
    ///
    /// Memory that a capture begun with [CudaStream::begin_capture_in] left allocated stays the
    /// graph's while the graph lives; without a graph, it is the pool's again once freed.
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        self.ctx.bind_to_thread()?;
        let pool_capture = self.end_pool_capture();
        // SAFETY: the stream is valid; ending a capture it is not in is an error, not UB.
        let graph = unsafe { result::stream::end_capture(self.cu_stream) }.and_then(|cu_graph| {
            if cu_graph.is_null() {
                return Ok(None);
            }
            // SAFETY: `cu_graph` is the graph just captured, which this function owns; it is
            // destroyed here only if no executable graph takes it over.
            match unsafe { result::graph::instantiate(cu_graph, flags) } {
                Ok(cu_graph_exec) => Ok(Some((cu_graph, cu_graph_exec))),
                Err(err) => {
                    self.ctx
                        .record_err(unsafe { result::graph::destroy(cu_graph) });
                    Err(err)
                }
            }
        });
        if let Some(capture) = &pool_capture {
            capture.end(matches!(graph, Ok(Some(_))));
        }
        Ok(graph?.map(|(cu_graph, cu_graph_exec)| CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.clone(),
            pool_capture,
        }))
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca)
    pub fn capture_status(&self) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::is_capturing(self.cu_stream) }
    }
}

impl CudaGraph {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Pre-uploads the graph's resources to the device so that the
    /// first [CudaGraph::launch()] does not incur setup overhead.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb81438b083d42a26693f6f2bce150cd)
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Get the underlying [sys::CUgraph].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not destroy the graph**, as it is still
    /// owned by the [CudaGraph].
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Get the underlying [sys::CUgraphExec].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not destroy the graph exec**, as it is still
    /// owned by the [CudaGraph].
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }
}
