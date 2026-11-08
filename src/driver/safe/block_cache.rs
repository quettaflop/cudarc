//! Freed device memory kept for reuse on the stream that freed it, see
//! [CudaContext::cache_freed_blocks](super::CudaContext::cache_freed_blocks).

use std::collections::HashMap;
use std::thread::ThreadId;
use std::vec::Vec;

use crate::driver::sys;

/// The handle of the calling thread's default stream, the same in every thread.
pub(crate) const PER_THREAD_STREAM: usize = 0x2;

/// A stream as memory freed on it is ordered: its handle and, for the per-thread default stream,
/// the thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StreamKey(usize, Option<ThreadId>);

impl StreamKey {
    /// The key of `stream` used from the calling thread.
    pub(crate) fn of(stream: sys::CUstream) -> Self {
        let per_thread = stream as usize == PER_THREAD_STREAM;
        Self(
            stream as usize,
            per_thread.then(|| std::thread::current().id()),
        )
    }

    /// Whether this key's stream is `stream` used from the calling thread.
    pub(crate) fn is(&self, stream: sys::CUstream) -> bool {
        *self == Self::of(stream)
    }

    pub(crate) fn stream(&self) -> sys::CUstream {
        self.0 as sys::CUstream
    }
}

/// The size of the block that serves an allocation of `bytes`: a multiple of 512 bytes up to
/// 4 KiB, then one of four sizes per power of two, so a block is at most a quarter larger than
/// any allocation it serves.
pub(crate) fn size_class(bytes: usize) -> usize {
    if bytes <= 4096 {
        return bytes.div_ceil(512).max(1) * 512;
    }
    let top = usize::BITS - 1 - (bytes - 1).leading_zeros();
    let step = 1usize << (top - 2);
    bytes.div_ceil(step) * step
}

#[derive(Debug)]
pub(crate) struct BlockCache {
    /// Freed blocks by the stream that freed them and their size class.
    free: HashMap<(StreamKey, usize), Vec<sys::CUdeviceptr>>,
    /// Bytes in `free`.
    held: usize,
    /// The most bytes `free` may hold.
    limit: usize,
}

impl BlockCache {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            free: HashMap::new(),
            held: 0,
            limit,
        }
    }

    /// A freed block of size class `class` on `stream`, if one is held.
    pub(crate) fn take(&mut self, stream: StreamKey, class: usize) -> Option<sys::CUdeviceptr> {
        let block = self.free.get_mut(&(stream, class))?.pop()?;
        self.held -= class;
        Some(block)
    }

    /// Holds `block` of size class `class`, freed on `stream`; false if that would hold more
    /// than the limit.
    pub(crate) fn put(&mut self, stream: StreamKey, class: usize, block: sys::CUdeviceptr) -> bool {
        if self.held + class > self.limit {
            return false;
        }
        self.held += class;
        self.free.entry((stream, class)).or_default().push(block);
        true
    }

    /// Gives up the blocks held for the streams `which` selects.
    pub(crate) fn drain(
        &mut self,
        which: impl Fn(&StreamKey) -> bool,
    ) -> Vec<(StreamKey, sys::CUdeviceptr)> {
        let mut blocks = Vec::new();
        self.free.retain(|(stream, class), held| {
            if !which(stream) {
                return true;
            }
            self.held -= *class * held.len();
            blocks.extend(held.drain(..).map(|block| (*stream, block)));
            false
        });
        blocks
    }

    pub(crate) fn held(&self) -> usize {
        self.held
    }

    pub(crate) fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_serve_their_range() {
        let mut previous = 0;
        for bytes in 1..1 << 20 {
            let class = size_class(bytes);
            assert!(class >= bytes && class >= previous, "{bytes} -> {class}");
            assert!(
                class <= (bytes + 511).max(bytes + bytes / 4),
                "{bytes} -> {class}"
            );
            assert_eq!(size_class(class), class, "{bytes} -> {class}");
            previous = class;
        }
    }

    #[test]
    fn holds_up_to_the_limit() {
        let stream = StreamKey::of(std::ptr::null_mut());
        let mut cache = BlockCache::new(2048);
        assert!(cache.put(stream, 1024, 1));
        assert!(cache.put(stream, 1024, 2));
        assert!(!cache.put(stream, 512, 3));
        assert_eq!(cache.take(stream, 1024), Some(2));
        assert_eq!(cache.take(stream, 512), None);
        assert_eq!(cache.held(), 1024);
        assert_eq!(cache.drain(|_| true), [(stream, 1)]);
        assert_eq!(cache.held(), 0);
    }

    #[test]
    fn keys_the_per_thread_stream_by_thread() {
        let per_thread = || StreamKey::of(PER_THREAD_STREAM as sys::CUstream);
        let there = std::thread::spawn(per_thread).join().unwrap();
        assert_ne!(per_thread(), there);
        let null = || StreamKey::of(std::ptr::null_mut());
        let null_there = std::thread::spawn(null).join().unwrap();
        assert_eq!(null(), null_there);
    }
}
