//! Device memory for the allocations of stream captures, see [CapturePool].

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{atomic::Ordering, Arc, Mutex, PoisonError};
use std::vec::Vec;

use crate::driver::{
    result::{self, DriverError},
    sys,
};

use super::{block_cache::StreamKey, CudaStream};

/// Blocks are multiples of this many bytes, so they stay as aligned as `cuMemAlloc`'s memory.
const ALIGNMENT: usize = 512;

/// The smallest segment a pool takes from the driver. Larger ones are powers of two while memory
/// allows, so that graphs recorded one after another with growing allocations hold at most about
/// twice the largest.
const MIN_SEGMENT: usize = 2 << 20;

/// Device memory that the captures of one stream allocate from, apart from all other memory, so
/// that their graphs hold no memory nodes: kernels then depend on kernels directly, as
/// programmatic dependent launch needs.
///
/// [CudaStream::begin_capture_in] begins a relaxed capture of the pool's stream. Until
/// [CudaStream::end_capture], allocations on the stream come from the pool and go back to it when
/// freed, neither in the stream's order, so the graph records neither. The pool takes new memory
/// from `cuMemAlloc`, which a relaxed capture allows and records nothing for. Peers reach that
/// memory through peer access between contexts, not through memory pool access.
///
/// The graphs captured into a pool share its memory, as graphs share the driver's graph memory
/// pool: memory that one capture freed serves the next. That is sound because they all run on the
/// pool's stream ([CudaGraph::launch](super::CudaGraph::launch) launches a graph on the stream it
/// was captured on), one at a time, and each writes the memory it allocated before reading it, so
/// they may run in any order. Within a graph, memory one kernel freed serves later kernels in the
/// stream's order only: with programmatic dependent launch, a kernel must not touch such memory
/// before it waits for the kernels it depends on (`griddepcontrol.wait`).
///
/// Memory that a capture leaves allocated is the exception. The pool keeps it from later
/// captures while their graph lives, even once freed, but graphs captured before may have used it
/// and write it when they run: read it before running such a graph, or allocate memory that
/// outlives the runs of other graphs before the capture. [CapturePool::in_use_bytes] tells
/// whether captures left memory allocated.
///
/// For the per-thread default stream, capture, launch and drop from the thread that created the
/// pool: another thread's per-thread stream is another stream. Allocations on other streams that
/// join a capture, such as forks of the stream, are not the pool's. A capture begun in a pool
/// must be ended with [CudaStream::end_capture]: until then, the pool keeps its stream and
/// context alive.
///
/// The pool keeps its memory until the pool, its graphs and the memory they allocated are all
/// dropped, then returns it to the driver once its stream's work is done. Running out of device
/// memory during a capture fails the allocation, not the capture: drop the graph of a capture
/// whose work failed.
pub struct CapturePool {
    stream: Arc<CudaStream>,
    /// The stream as the thread that created the pool sees it.
    key: StreamKey,
    blocks: Mutex<Blocks>,
}

impl CapturePool {
    /// A pool for the captures of `stream`.
    pub fn new(stream: &Arc<CudaStream>) -> Arc<Self> {
        Arc::new(Self {
            stream: stream.clone(),
            key: StreamKey::of(stream.cu_stream),
            blocks: Mutex::new(Blocks::default()),
        })
    }

    /// The stream whose captures allocate from the pool.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Whether the pool serves the captures of `stream`, used from the calling thread.
    pub(crate) fn serves(&self, stream: &CudaStream) -> bool {
        self.stream.ctx == stream.ctx && self.key.is(stream.cu_stream)
    }

    /// Bytes of device memory the pool holds.
    pub fn reserved_bytes(&self) -> usize {
        self.blocks.lock().unwrap().reserved()
    }

    /// Bytes of the pool's memory that slices hold, or graphs since their capture ended.
    pub fn in_use_bytes(&self) -> usize {
        self.blocks.lock().unwrap().in_use
    }

    /// Starts a capture into the pool.
    pub(crate) fn begin(self: &Arc<Self>) -> PoolCapture {
        PoolCapture {
            id: self.blocks.lock().unwrap().begin(),
            pool: self.clone(),
        }
    }
}

impl std::fmt::Debug for CapturePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturePool")
            .field("stream", &self.key)
            .field(
                "reserved_bytes",
                &self.blocks.try_lock().map(|blocks| blocks.reserved()).ok(),
            )
            .finish()
    }
}

impl Drop for CapturePool {
    fn drop(&mut self) {
        let blocks = self
            .blocks
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if blocks.segments.is_empty() {
            return;
        }
        let (ctx, stream) = (&self.stream.ctx, self.stream.cu_stream);
        // Errors recorded on the context before are not this drop's to report or act on.
        // SAFETY: the context is alive, as the pool's stream holds it.
        if let Err(err) = unsafe { result::ctx::set_current(ctx.cu_ctx) } {
            ctx.record_err(Err::<(), _>(err));
            return;
        }
        // The last replays of the pool's graphs may still run: wait for them, or else keep the
        // memory rather than free it under them. Another thread's per-thread stream is waited for
        // with the context. Synchronizing a stream in a capture would invalidate the capture: the
        // memory then stays allocated, which is no error of the capture's.
        let waited = match self.key.is(stream) {
            // SAFETY: the pool's stream, alive as the pool holds it.
            true => match unsafe { result::stream::is_capturing(stream) } {
                Ok(sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE) => {
                    // SAFETY: as above.
                    unsafe { result::stream::synchronize(stream) }
                }
                Ok(_) => return,
                Err(err) => Err(err),
            },
            false => result::ctx::synchronize(),
        };
        if waited.is_err() {
            ctx.record_err(waited);
            return;
        }
        relaxed(|| {
            for &segment in blocks.segments.keys() {
                // SAFETY: the pool allocated the segment with cuMemAlloc, and nothing uses it any
                // more: the graphs and slices that held the pool are gone and their work is done.
                ctx.record_err(unsafe { result::free_sync(segment) });
            }
        });
    }
}

/// `f` with the calling thread's capture interaction mode relaxed: allocating or freeing device
/// memory is prohibited while another thread captures in global mode, unless relaxed.
fn relaxed<T>(f: impl FnOnce() -> T) -> T {
    let mut mode = sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
    // SAFETY: swaps the calling thread's mode, which the call below swaps back.
    let swapped = unsafe { sys::cuThreadExchangeStreamCaptureMode(&mut mode) }.result();
    let out = f();
    if swapped.is_ok() {
        // SAFETY: as above.
        unsafe { sys::cuThreadExchangeStreamCaptureMode(&mut mode) };
    }
    out
}

/// A capture into a [CapturePool]: where the memory it allocates comes from, and, held by its
/// graph, the graph's claim on that memory.
#[derive(Debug, Clone)]
pub(crate) struct PoolCapture {
    pool: Arc<CapturePool>,
    id: u64,
}

impl PoolCapture {
    /// A block of at least `bytes` for this capture.
    pub(crate) fn alloc(&self, bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
        let mut blocks = self.pool.blocks.lock().unwrap();
        // SAFETY: new device memory, which the pool owns from now on; a relaxed capture allows
        // cuMemAlloc and records nothing for it.
        relaxed(|| blocks.take(bytes, |size| unsafe { result::malloc_sync(size) }))
    }

    /// Takes back `block`, which this capture allocated.
    pub(crate) fn free(&self, block: sys::CUdeviceptr) {
        self.blocks().give_back(self.id, block);
    }

    /// Keeps the memory this capture frees from now on for its graph, before the capture ends.
    fn ending(&self) {
        self.blocks().ending(self.id);
    }

    /// Ends this capture, which recorded a graph or not.
    pub(crate) fn end(&self, graph: bool) {
        self.blocks().end(self.id, graph);
    }

    /// Releases the memory this capture's graph held, once the graph is dropped.
    pub(crate) fn graph_dropped(&self) {
        self.blocks().graph_dropped(self.id);
    }

    /// The pool's blocks, also after a thread panicked holding them: frees and drops go on.
    fn blocks(&self) -> std::sync::MutexGuard<'_, Blocks> {
        self.pool
            .blocks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl CudaStream {
    /// Begins a [relaxed](sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) capture of
    /// this stream whose allocations on it come from `pool` and go back to it, recording no memory
    /// nodes; see [CapturePool]. [CudaStream::end_capture] ends it.
    ///
    /// # Panics
    ///
    /// If `pool` is another stream's.
    pub fn begin_capture_in(&self, pool: &Arc<CapturePool>) -> Result<(), DriverError> {
        assert!(
            pool.serves(self),
            "a capture pool serves the captures of its own stream"
        );
        self.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let capture = pool.begin();
        let key = StreamKey::of(self.cu_stream);
        self.ctx.pool_captures.lock().unwrap().push((key, capture));
        self.ctx.num_pool_captures.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// The capture into a pool that this stream is in, used from the calling thread.
    pub(crate) fn pool_capture(&self) -> Result<Option<PoolCapture>, DriverError> {
        if self.ctx.num_pool_captures.load(Ordering::Acquire) == 0 {
            return Ok(None);
        }
        let key = StreamKey::of(self.cu_stream);
        let captures = self.ctx.pool_captures.lock().unwrap();
        let Some((_, capture)) = captures.iter().find(|(k, _)| *k == key) else {
            return Ok(None);
        };
        // A capture ended other than through end_capture is no longer the pool's.
        // SAFETY: querying a valid stream.
        let status = unsafe { result::stream::is_capturing(self.cu_stream) }?;
        Ok(
            (status == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
                .then(|| capture.clone()),
        )
    }

    /// Takes this stream's capture into a pool, if it is in one, from the context's, after which
    /// the memory the capture frees stays its graph's, should it record one.
    pub(crate) fn end_pool_capture(&self) -> Option<PoolCapture> {
        if self.ctx.num_pool_captures.load(Ordering::Acquire) == 0 {
            return None;
        }
        let key = StreamKey::of(self.cu_stream);
        let mut captures = (self.ctx.pool_captures.lock()).unwrap_or_else(PoisonError::into_inner);
        let index = captures.iter().position(|(k, _)| *k == key)?;
        // Before the capture leaves the context's, so that no free in between misses it.
        captures[index].1.ending();
        self.ctx.num_pool_captures.fetch_sub(1, Ordering::AcqRel);
        Some(captures.swap_remove(index).1)
    }
}

/// A block of a segment: its size, the segment's address, and whether it is free.
#[derive(Debug, Clone, Copy)]
struct Block {
    size: usize,
    segment: sys::CUdeviceptr,
    free: bool,
}

/// A pool's segments and the blocks they are split into: the best-fitting free block serves an
/// allocation, split off what it does not need, and freed blocks merge with free neighbours of
/// their segment. Driver calls stay with the caller, so this runs without a device.
#[derive(Debug, Default)]
pub(crate) struct Blocks {
    /// Segments by address, and their sizes.
    segments: BTreeMap<sys::CUdeviceptr, usize>,
    /// Every block by address.
    blocks: BTreeMap<sys::CUdeviceptr, Block>,
    /// Free blocks by size and address.
    free: BTreeSet<(usize, sys::CUdeviceptr)>,
    /// Captures whose graphs live, with the blocks they allocated that were freed after the
    /// capture ended, which their graphs still use.
    graphs: HashMap<u64, Vec<sys::CUdeviceptr>>,
    /// Captures begun so far.
    captures: u64,
    /// Bytes of the blocks that are not free.
    in_use: usize,
}

impl Blocks {
    fn begin(&mut self) -> u64 {
        self.captures += 1;
        self.captures
    }

    /// From now on, the blocks of `capture` freed stay its graph's, should it record one.
    fn ending(&mut self, capture: u64) {
        self.graphs.insert(capture, Vec::new());
    }

    fn end(&mut self, capture: u64, graph: bool) {
        if graph {
            self.graphs.entry(capture).or_default();
        } else {
            self.graph_dropped(capture);
        }
    }

    fn graph_dropped(&mut self, capture: u64) {
        for block in self.graphs.remove(&capture).into_iter().flatten() {
            self.release(block);
        }
    }

    fn reserved(&self) -> usize {
        self.segments.values().sum()
    }

    /// A block of at least `bytes`, from a new segment that `segment` allocates if no free block is
    /// large enough: a power of two, or if that fails, the least multiple of the smallest segment.
    fn take<E>(
        &mut self,
        bytes: usize,
        mut segment: impl FnMut(usize) -> Result<sys::CUdeviceptr, E>,
    ) -> Result<sys::CUdeviceptr, E> {
        // A size past the address space cannot be allocated: the driver says so for the segment.
        let size = (bytes.max(1).checked_next_multiple_of(ALIGNMENT)).unwrap_or(usize::MAX);
        let found = self.free.range((size, 0)..).next().copied();
        let (address, block) = match found {
            Some((_, address)) => {
                let block = self.blocks[&address];
                self.free.remove(&(block.size, address));
                (address, block)
            }
            None => {
                let least = (size.checked_next_multiple_of(MIN_SEGMENT)).unwrap_or(size);
                let rounded = (size.checked_next_power_of_two()).map_or(least, |s| s.max(least));
                let (address, segment_size) = match segment(rounded) {
                    Ok(address) => (address, rounded),
                    Err(_) if least < rounded => (segment(least)?, least),
                    Err(err) => return Err(err),
                };
                self.segments.insert(address, segment_size);
                let block = Block {
                    size: segment_size,
                    segment: address,
                    free: true,
                };
                (address, block)
            }
        };
        // Sizes are multiples of the alignment, so any rest can serve another allocation.
        if block.size > size {
            let rest = Block {
                size: block.size - size,
                ..block
            };
            let rest_address = address + size as sys::CUdeviceptr;
            self.blocks.insert(rest_address, rest);
            self.free.insert((rest.size, rest_address));
        }
        let block = Block {
            size,
            free: false,
            ..block
        };
        self.blocks.insert(address, block);
        self.in_use += size;
        Ok(address)
    }

    /// Takes back `block`, allocated by capture `capture`: its graph keeps it while it lives.
    fn give_back(&mut self, capture: u64, block: sys::CUdeviceptr) {
        match self.graphs.get_mut(&capture) {
            Some(held) => held.push(block),
            None => self.release(block),
        }
    }

    /// Frees `address`, merged with the free blocks next to it in its segment.
    fn release(&mut self, mut address: sys::CUdeviceptr) {
        let mut block = self.blocks.remove(&address).expect("a block of this pool");
        debug_assert!(!block.free, "a block is freed once");
        self.in_use -= block.size;
        let next = address + block.size as sys::CUdeviceptr;
        if let Some(&after) = self.blocks.get(&next) {
            if after.free && after.segment == block.segment {
                self.blocks.remove(&next);
                self.free.remove(&(after.size, next));
                block.size += after.size;
            }
        }
        if let Some((&previous, &before)) = self.blocks.range(..address).next_back() {
            let adjacent = previous + before.size as sys::CUdeviceptr == address;
            if before.free && before.segment == block.segment && adjacent {
                self.blocks.remove(&previous);
                self.free.remove(&(before.size, previous));
                block.size += before.size;
                address = previous;
            }
        }
        block.free = true;
        self.blocks.insert(address, block);
        self.free.insert((block.size, address));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{sys::CUgraphNodeType, CudaContext, CudaGraph};

    const FLAGS: sys::CUgraphInstantiate_flags =
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

    /// The types of `graph`'s nodes, as numbers.
    fn node_types(graph: &CudaGraph) -> Vec<u32> {
        let mut count = 0;
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }
            .result()
            .unwrap();
        let mut nodes = std::vec![std::ptr::null_mut(); count];
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) }
            .result()
            .unwrap();
        (nodes.iter())
            .map(|&node| {
                let mut kind = 0u32;
                let kind_ptr = &mut kind as *mut u32 as *mut CUgraphNodeType;
                unsafe { sys::cuGraphNodeGetType(node, kind_ptr) }
                    .result()
                    .unwrap();
                kind
            })
            .collect()
    }

    #[test]
    fn test_pool_capture_records_no_memory_nodes() {
        let ctx = CudaContext::new(0).unwrap();
        // One stream: no waits on events recorded inside a capture.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        let first = stream.alloc_zeros::<f32>(1000).unwrap();
        let block = first.cu_device_ptr;
        drop(first);
        let mut again = unsafe { stream.alloc::<u8>(3000) }.unwrap();
        assert_eq!(again.cu_device_ptr, block);
        stream.memset_zeros(&mut again).unwrap();
        let graph = stream.end_capture(FLAGS).unwrap().unwrap();
        let memset = CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET as u32;
        assert_eq!(node_types(&graph), [memset; 2]);
        graph.launch().unwrap();
        stream.synchronize().unwrap();
        assert_eq!(pool.reserved_bytes(), MIN_SEGMENT);
    }

    #[test]
    fn test_pool_capture_keeps_graph_memory() {
        let ctx = CudaContext::new(0).unwrap();
        // One stream: no waits on events recorded inside a capture.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        let kept = stream.alloc_zeros::<u8>(4096).unwrap();
        let first = stream.end_capture(FLAGS).unwrap().unwrap();
        let block = kept.cu_device_ptr;
        // The first graph still writes it.
        drop(kept);
        stream.begin_capture_in(&pool).unwrap();
        let other = stream.alloc_zeros::<u8>(4096).unwrap();
        assert_ne!(other.cu_device_ptr, block);
        drop(other);
        let second = stream.end_capture(FLAGS).unwrap().unwrap();
        drop(first);
        stream.begin_capture_in(&pool).unwrap();
        let again = stream.alloc_zeros::<u8>(4096).unwrap();
        assert_eq!(again.cu_device_ptr, block);
        drop(again);
        let third = stream.end_capture(FLAGS).unwrap().unwrap();
        second.launch().unwrap();
        third.launch().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_pool_capture_survives_a_failed_capture() {
        let ctx = CudaContext::new(0).unwrap();
        // One stream: no waits on events recorded inside a capture.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        let kept = stream.alloc_zeros::<u8>(4096).unwrap();
        // Synchronizing a capturing stream invalidates its capture.
        assert!(stream.synchronize().is_err());
        assert!(stream.end_capture(FLAGS).is_err());
        let block = kept.cu_device_ptr;
        drop(kept);
        stream.begin_capture_in(&pool).unwrap();
        let again = stream.alloc_zeros::<u8>(4096).unwrap();
        assert_eq!(again.cu_device_ptr, block);
        drop(again);
        let graph = stream.end_capture(FLAGS).unwrap().unwrap();
        graph.launch().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_pool_capture_follows_the_stream_not_its_wrapper() {
        use crate::driver::safe::core::Origin;
        let ctx = CudaContext::new(0).unwrap();
        unsafe { ctx.disable_event_tracking() };
        let (stream, wrapper) = (ctx.per_thread_stream(), ctx.per_thread_stream());
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        // Another wrapper of the stream, in this thread, allocates from the pool.
        let mut here = unsafe { wrapper.alloc::<u8>(4096) }.unwrap();
        assert!(matches!(here.origin, Origin::Pool(_)));
        wrapper.memset_zeros(&mut here).unwrap();
        drop(here);
        // Another thread's per-thread stream is another stream.
        let theirs = ctx.clone();
        let pooled = std::thread::spawn(move || {
            let slice = theirs.per_thread_stream().alloc_zeros::<u8>(4096).unwrap();
            matches!(slice.origin, Origin::Pool(_))
        });
        assert!(!pooled.join().unwrap());
        // Ended through the other wrapper, the graph holds the pool all the same.
        let graph = wrapper.end_capture(FLAGS).unwrap().unwrap();
        assert_eq!(Arc::strong_count(&pool), 2);
        let after = unsafe { stream.alloc::<u8>(16) }.unwrap();
        assert!(!matches!(after.origin, Origin::Pool(_)));
        graph.launch().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_pool_capture_ended_elsewhere_is_over() {
        use crate::driver::safe::core::Origin;
        let ctx = CudaContext::new(0).unwrap();
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        let cu_graph = unsafe { result::stream::end_capture(stream.cu_stream) }.unwrap();
        unsafe { result::graph::destroy(cu_graph) }.unwrap();
        let eager = unsafe { stream.alloc::<u8>(16) }.unwrap();
        assert!(!matches!(eager.origin, Origin::Pool(_)));
        // The next capture lets go of the stale one.
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        stream.end_capture(FLAGS).unwrap();
        assert_eq!(Arc::strong_count(&pool), 1);
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_pool_dropped_during_a_capture_leaves_it_alone() {
        let ctx = CudaContext::new(0).unwrap();
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        drop(stream.alloc_zeros::<u8>(4096).unwrap());
        let graph = stream.end_capture(FLAGS).unwrap().unwrap();
        graph.launch().unwrap();
        drop(graph);
        let mut kept = stream.alloc_zeros::<u8>(4096).unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        // The last reference: the pool cannot wait for its stream now, and keeps its memory.
        drop(pool);
        stream.memset_zeros(&mut kept).unwrap();
        let graph = stream.end_capture(FLAGS).unwrap().unwrap();
        graph.launch().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_pool_counts_what_captures_leave_allocated() {
        let ctx = CudaContext::new(0).unwrap();
        // One stream: no waits on events recorded inside a capture.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let pool = CapturePool::new(&stream);
        stream.begin_capture_in(&pool).unwrap();
        drop(stream.alloc_zeros::<u8>(4096).unwrap());
        let kept = stream.alloc_zeros::<u8>(1000).unwrap();
        let graph = stream.end_capture(FLAGS).unwrap().unwrap();
        assert_eq!(pool.in_use_bytes(), 1024);
        drop(kept);
        // The graph still writes it.
        assert_eq!(pool.in_use_bytes(), 1024);
        drop(graph);
        assert_eq!(pool.in_use_bytes(), 0);
    }

    #[test]
    #[should_panic(expected = "a capture pool serves the captures of its own stream")]
    fn test_pool_capture_needs_the_pools_stream() {
        let ctx = CudaContext::new(0).unwrap();
        let pool = CapturePool::new(&ctx.new_stream().unwrap());
        let _ = ctx.new_stream().unwrap().begin_capture_in(&pool);
    }

    #[cfg(feature = "nvrtc")]
    #[test]
    fn test_pool_captures_chain_kernels_and_replay_apart() {
        use crate::driver::{CudaSlice, LaunchConfig, PushKernelArg};

        let ptx = crate::nvrtc::compile_ptx(
            "extern \"C\" __global__ void affine(float *y, const float *x, float a, float b,
                                                 unsigned n) {
                unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) y[i] = a * x[i] + b;
            }",
        )
        .unwrap();
        let ctx = CudaContext::new(0).unwrap();
        // Everything runs on one stream.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let affine = ctx
            .load_module(ptx)
            .unwrap()
            .load_function("affine")
            .unwrap();
        let n = 1000u32;
        let cfg = LaunchConfig::for_num_elems(n);
        let x = stream.clone_htod(&[1.0f32; 1000]).unwrap();
        let pool = CapturePool::new(&stream);
        // Records out = c * (b * (a * x + 1) + 1) + 1 through blocks freed and reused.
        let record = |[a, b, c]: [f32; 3]| -> (CudaGraph, CudaSlice<f32>) {
            let mut out = stream.alloc_zeros::<f32>(1000).unwrap();
            stream.begin_capture_in(&pool).unwrap();
            let mut first = unsafe { stream.alloc::<f32>(1000) }.unwrap();
            let one = 1.0f32;
            let launch = |y: &mut CudaSlice<f32>, x: &CudaSlice<f32>, a: f32| {
                let mut builder = stream.launch_builder(&affine);
                builder.arg(y).arg(x).arg(&a).arg(&one).arg(&n);
                unsafe { builder.launch(cfg) }.unwrap();
            };
            launch(&mut first, &x, a);
            let mut second = unsafe { stream.alloc::<f32>(1000) }.unwrap();
            launch(&mut second, &first, b);
            drop(first);
            launch(&mut out, &second, c);
            drop(second);
            (stream.end_capture(FLAGS).unwrap().unwrap(), out)
        };
        let (g, g_out) = record([2.0, 3.0, 4.0]);
        let (h, h_out) = record([5.0, 6.0, 7.0]);
        let kernel = CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL as u32;
        assert_eq!(node_types(&g), [kernel; 3]);
        assert_eq!(node_types(&h), [kernel; 3]);
        for _ in 0..3 {
            g.launch().unwrap();
            h.launch().unwrap();
            assert_eq!(stream.clone_dtoh(&g_out).unwrap(), [4.0 * 10.0 + 1.0; 1000]);
            assert_eq!(stream.clone_dtoh(&h_out).unwrap(), [7.0 * 37.0 + 1.0; 1000]);
        }
    }

    /// Segments from a fake address space, one after another without gaps.
    fn segments() -> impl FnMut(usize) -> Result<sys::CUdeviceptr, ()> {
        let mut next = 1 << 40;
        move |size| {
            let address = next;
            next += size as sys::CUdeviceptr;
            Ok(address)
        }
    }

    fn take(
        blocks: &mut Blocks,
        bytes: usize,
        alloc: &mut impl FnMut(usize) -> Result<u64, ()>,
    ) -> u64 {
        blocks.take(bytes, &mut *alloc).unwrap()
    }

    #[test]
    fn splits_the_best_fit_and_merges_freed_neighbours() {
        let mut alloc = segments();
        let mut blocks = Blocks::default();
        let c = blocks.begin();
        let a = take(&mut blocks, 1000, &mut alloc);
        let b = take(&mut blocks, 512, &mut alloc);
        let d = take(&mut blocks, 4096, &mut alloc);
        assert_eq!(blocks.reserved(), MIN_SEGMENT);
        assert_eq!((b - a, d - b), (1024, 512));
        blocks.give_back(c, b);
        // The freed 512 bytes fit best.
        assert_eq!(take(&mut blocks, 1, &mut alloc), b);
        blocks.give_back(c, b);
        blocks.give_back(c, a);
        blocks.give_back(c, d);
        assert_eq!(
            blocks.free.iter().copied().collect::<Vec<_>>(),
            [(MIN_SEGMENT, a)]
        );
        assert_eq!(blocks.blocks.len(), 1);
    }

    #[test]
    fn rounds_large_segments_to_powers_of_two() {
        let mut alloc = segments();
        let mut blocks = Blocks::default();
        let c = blocks.begin();
        let a = take(&mut blocks, 3 << 20, &mut alloc);
        assert_eq!(blocks.reserved(), 4 << 20);
        // The rest of the segment serves smaller allocations.
        let b = take(&mut blocks, 1 << 20, &mut alloc);
        assert_eq!(b - a, 3 << 20);
        assert_eq!(blocks.reserved(), 4 << 20);
        blocks.give_back(c, a);
        blocks.give_back(c, b);
        // A larger one needs a new segment, and the old one serves what fits.
        let large = take(&mut blocks, 5 << 20, &mut alloc);
        assert_eq!(blocks.reserved(), 12 << 20);
        assert_eq!(take(&mut blocks, 4 << 20, &mut alloc), a);
        blocks.give_back(c, large);
    }

    #[test]
    fn keeps_segments_apart() {
        let mut alloc = segments();
        let mut blocks = Blocks::default();
        let c = blocks.begin();
        let a = take(&mut blocks, MIN_SEGMENT, &mut alloc);
        let b = take(&mut blocks, MIN_SEGMENT, &mut alloc);
        assert_eq!(b - a, MIN_SEGMENT as u64);
        blocks.give_back(c, a);
        blocks.give_back(c, b);
        // Adjacent segments are separate allocations: no block spans two.
        assert_eq!(blocks.free.len(), 2);
        assert!(blocks.take(2 * MIN_SEGMENT, |_| Err(())).is_err());
    }

    #[test]
    fn graphs_keep_what_their_captures_left_allocated() {
        let mut alloc = segments();
        let mut blocks = Blocks::default();
        let first = blocks.begin();
        let kept = take(&mut blocks, 4096, &mut alloc);
        let temporary = take(&mut blocks, 4096, &mut alloc);
        blocks.give_back(first, temporary);
        blocks.ending(first);
        blocks.end(first, true);
        // Freed once the capture ended, while its graph lives: the graph still uses it.
        blocks.give_back(first, kept);
        let second = blocks.begin();
        let other = take(&mut blocks, 4096, &mut alloc);
        assert_eq!(other, temporary);
        let more = take(&mut blocks, 4096, &mut alloc);
        assert_ne!(more, kept);
        blocks.give_back(second, other);
        blocks.give_back(second, more);
        blocks.ending(second);
        blocks.end(second, true);
        blocks.graph_dropped(first);
        let third = blocks.begin();
        assert_eq!(take(&mut blocks, 4096, &mut alloc), kept);
        blocks.end(third, false);
        blocks.graph_dropped(second);
    }

    #[test]
    fn falls_back_to_the_least_segment() {
        let mut blocks = Blocks::default();
        blocks.begin();
        let mut alloc = segments();
        // Room for 6 MiB but not the 8 MiB power of two.
        let tight = |size| match size {
            size if size <= 6 << 20 => alloc(size),
            _ => Err(()),
        };
        let block = blocks.take((5 << 20) + 1, tight).unwrap();
        assert_eq!(blocks.reserved(), 6 << 20);
        assert_eq!(blocks.segments[&block], 6 << 20);
    }

    #[test]
    fn fails_sizes_past_the_address_space() {
        let mut blocks = Blocks::default();
        blocks.begin();
        let asked = std::cell::Cell::new(0);
        let failed = blocks.take(usize::MAX - 10, |size| {
            asked.set(size);
            Err(())
        });
        assert!(failed.is_err());
        assert_eq!(asked.get(), usize::MAX);
        assert_eq!(blocks.reserved(), 0);
    }

    #[test]
    fn failed_captures_keep_nothing() {
        let mut alloc = segments();
        let mut blocks = Blocks::default();
        let failed = blocks.begin();
        let kept = take(&mut blocks, 4096, &mut alloc);
        let freed_while_ending = take(&mut blocks, 4096, &mut alloc);
        blocks.ending(failed);
        blocks.give_back(failed, freed_while_ending);
        assert_eq!(blocks.in_use, 8192);
        blocks.end(failed, false);
        assert_eq!(blocks.in_use, 4096);
        blocks.give_back(failed, kept);
        assert_eq!(blocks.in_use, 0);
        let next = blocks.begin();
        assert_eq!(take(&mut blocks, 4096, &mut alloc), kept);
        blocks.end(next, false);
    }
}
