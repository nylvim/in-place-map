//! This crate provides the functionality of converting between the same container type of different
//! element types, reusing allocation if possible.
//!
//! The core trait of this crate, [`MapInPlace`], is implemented for the following types:
//! - Common collections: [`Vec`], [`VecDeque`] and [`BinaryHeap`].
//! - Smart pointers: [`Box<T>`], [`Rc<T>`], [`Arc<T>`], [`Cow<'a, T>`], [`Box<[T]>`](Box),
//!   [`Rc<[T]>`](Rc), [`Arc<[T]>`](Arc) and [`Cow<'a, [T]>`](Cow).
//! - Fixed-sized arrays: [`[T; N]`](https://doc.rust-lang.org/stable/core/primitive.array.html).
//! - Other container types: [`Option`], [`Result`], [`Cell`], [`RefCell`], [`UnsafeCell`],
//!   [`Mutex`], and [`RwLock`].
//!
//! The extension trait, [`FilterMapInPlace`], is implemented for [`Vec`], [`VecDeque`],
//! [`BinaryHeap`], [`Cow<'a, [T]>`](Cow) and [`Option`]. The collection implementations can reuse
//! excess capacity when their layouts permit it.
//!
//! ## Notes
//!
//! The definition of "in-place" in this crate might differ from what you think, see
//! [`MapInPlace`]'s documentation for detailed explanation.
//!
//! [`Rc`] and [`Arc`]'s implementations will clone the inner value to a new allocation if the
//! pointer is not unique, thus require `T: Clone`.
//!
//! # Examples
//!
//! ```rust
//! use in_place_map::{FilterMapInPlace, MapInPlace};
//! let vec = vec![1i32, 2, 3, 4];
//! let slice_addr = vec.as_ptr() as usize;
//!
//! let new_vec = vec.clone().try_map_in_place(|x| u32::try_from(x));
//! assert!(new_vec.is_ok());
//! assert_eq!(new_vec.unwrap(), [1u32, 2, 3, 4]);
//!
//! let new_vec = vec.clone().filter_map_in_place(|x| (x % 2 == 0).then_some(x as u32));
//! assert_eq!(new_vec, [2u32, 4]);
//!
//! let new_vec = vec.map_in_place(|x| x + 10);
//! assert_eq!(new_vec, [11i32, 12, 13, 14]);
//! assert_eq!(slice_addr, new_vec.as_ptr() as usize);
//! ```

#![warn(missing_docs)]

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::{BinaryHeap, VecDeque};
use std::convert::Infallible;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use std::{mem, ptr, slice};

/// Conversion between the same container type of different element types, reusing allocation if
/// possible.
///
/// The contract of this trait does not include that allocation is always reused; however,
/// implementors are encouraged to do this whenever possible.
///
/// This trait also does not promise that converted elements will be written back to the original
/// memory location, i.e. strictly "in-place". This would be impossible for types that directly hold
/// the elements inside, for example, arrays.
///
/// # Implementation limitations
///
/// Any implementation intended to reuse the original allocation should check if the types `T` and
/// `U` have the same size and alignment, otherwise writing the converted result back to the
/// original location is usually unviable. This can be ensured at compile-time by custom
/// implementations using the [`assert_in_place_compatible`] function.
///
/// However, if the container tolerates excess capacity, it may also accept those types `T` whose
/// size is greater than `U`'s. In this case use the [`assert_reuse_compatible`] function. Check its
/// documentation for more details.
pub trait MapInPlace<T, U>: Sized {
    /// The output type, usually the same container type, but with its element type changed to `U`.
    type Output;

    /// Attempt to map the container, reusing allocation if possible.
    fn try_map_in_place<E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<Self::Output, E>;

    /// Map the container, reusing allocation if possible.
    #[inline]
    fn map_in_place(self, f: impl FnMut(T) -> U) -> Self::Output {
        let mut f = f;
        self.try_map_in_place::<Infallible>(move |x| Ok(f(x))).unwrap()
    }
}

/// Filter and map a collection, reusing allocation if possible.
///
/// Collection implementations should tolerate excess capacity when they reuse an allocation. This
/// trait may also be implemented by containers with different filtering semantics, such as
/// [`Option`].
///
/// See [`MapInPlace`]'s documentation for more details.
pub trait FilterMapInPlace<T, U>: MapInPlace<T, U> {
    /// Filter and map a collection, reusing allocation if possible.
    fn filter_map_in_place(self, f: impl FnMut(T) -> Option<U>) -> Self::Output;
}

/// Assert at compile-time that the types `T` and `U` are compatible for in-place conversions.
///
/// This happens when and only when they have the same size and alignment, otherwise writing the
/// converted result back to the original place is unviable.
///
/// This function is exposed for the convenience of implementing [`MapInPlace`] for custom types; it
/// guarantees compile-time checking even when called outside a `const` context.
#[inline(always)]
pub const fn assert_in_place_compatible<T, U>() {
    const {
        assert!(size_of::<T>() == size_of::<U>(), "types must have the same size");
        assert!(align_of::<T>() == align_of::<U>(), "types must have the same alignment");
    }
}

/// Assert at compile-time that a memory chunk containing values of type `T` can be safely reused to
/// store values of type `U`.
///
/// This is true when `U` is zero-sized, or when `T` and `U` have the same alignment and `T`'s size
/// is not smaller than `U`'s size. Otherwise reusing memory is unviable.
///
/// If `U` is zero-sized, this is always true, as ZSTs don't take any space at all. In this case,
/// the original allocation should be freed.
///
/// If `T`'s size is not a multiple of `U`'s, the collection might need to shrink or expand its
/// capacity, or allocate some new memory. For example, the allocation of a `Vec<[u8; 3]>` with
/// capacity 3 could not be directly reused for a `Vec<[u8; 2]>`.
///
/// This function is exposed for the convenience of implementing [`MapInPlace`] for custom types; it
/// guarantees compile-time checking even when called outside a `const` context.
#[inline(always)]
pub const fn assert_reuse_compatible<T, U>() {
    const {
        assert!(size_of::<T>() >= size_of::<U>(), "`T`'s size must not be smaller than `U`'s");
        assert!(
            size_of::<U>() == 0 || align_of::<T>() == align_of::<U>(),
            "types must have the same alignment"
        );
    }
}

impl<T, U> MapInPlace<T, U> for Option<T> {
    type Output = Option<U>;

    /// Equivalent to `self.map(f).transpose()`.
    #[inline]
    fn try_map_in_place<E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Option<U>, E> {
        self.map(f).transpose()
    }

    /// Equivalent to [`Option::map`].
    #[inline]
    fn map_in_place(self, f: impl FnOnce(T) -> U) -> Option<U> {
        self.map(f)
    }
}

impl<T, U> FilterMapInPlace<T, U> for Option<T> {
    /// Equivalent to [`Option::and_then`].
    #[inline]
    fn filter_map_in_place(self, f: impl FnOnce(T) -> Option<U>) -> Option<U> {
        self.and_then(f)
    }
}

impl<T, U, RE> MapInPlace<T, U> for Result<T, RE> {
    type Output = Result<U, RE>;

    /// Essentially map and transpose the result.
    #[inline]
    fn try_map_in_place<E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Result<U, RE>, E> {
        Ok(match self {
            Ok(val) => Ok(f(val)?),
            Err(err) => Err(err),
        })
    }

    /// Equivalent to [`Result::map`].
    #[inline]
    fn map_in_place(self, f: impl FnOnce(T) -> U) -> Result<U, RE> {
        self.map(f)
    }
}

macro_rules! impl_cell {
    ($cell_ty:ident, $self_name:ident, $into_inner:expr $(,)?) => {
        impl<T, U> MapInPlace<T, U> for $cell_ty<T> {
            type Output = $cell_ty<U>;

            #[inline]
            fn try_map_in_place<E>(
                self,
                f: impl FnOnce(T) -> Result<U, E>,
            ) -> Result<$cell_ty<U>, E> {
                let $self_name = self;
                Ok($cell_ty::new(f($into_inner)?))
            }

            #[inline]
            // independent from `try_map_in_place` to make `f: impl FnOnce` possible
            fn map_in_place(self, f: impl FnOnce(T) -> U) -> $cell_ty<U> {
                let $self_name = self;
                $cell_ty::new(f($into_inner))
            }
        }
    };
}

impl_cell!(Cell, this, this.into_inner());
impl_cell!(RefCell, this, this.into_inner());
impl_cell!(UnsafeCell, this, this.into_inner());
impl_cell!(Mutex, this, this.into_inner().unwrap_or_else(|e| e.into_inner()));
impl_cell!(RwLock, this, this.into_inner().unwrap_or_else(|e| e.into_inner()));

// LazyCell::new does not work here because of some default type param issues
// <LazyCell<T> as From<T>> was not stable before 1.96.0, which is very recent
// the same for LazyLock

// impl_cell!(LazyCell, this, {
//     let ptr = &raw const *LazyCell::force(&this);
//     mem::forget(this);
//     unsafe { ptr::read(ptr) }
// });
// impl_cell!(LazyLock, this, {
//     let ptr = &raw const *LazyLock::force(&this);
//     mem::forget(this);
//     unsafe { ptr::read(ptr) }
// });

impl<T, U, const N: usize> MapInPlace<T, U> for [T; N] {
    type Output = [U; N];

    /// Equivalent to
    /// [`<[T; N]>::try_map`](https://doc.rust-lang.org/stable/std/primitive.array.html#method.try_map).
    fn try_map_in_place<E>(mut self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<[U; N], E> {
        struct Guard<T, U, const N: usize> {
            t_elems: *mut T,
            u_elems: *mut U,
            n_converted_elems: usize,
        }

        impl<T, U, const N: usize> Drop for Guard<T, U, N> {
            fn drop(&mut self) {
                unsafe {
                    let converted_slice =
                        slice::from_raw_parts_mut(self.u_elems, self.n_converted_elems);
                    ptr::drop_in_place(converted_slice);

                    let t_elems = self.t_elems.add(self.n_converted_elems + 1);
                    let unconverted_len = N - self.n_converted_elems - 1;
                    let unconverted_slice = slice::from_raw_parts_mut(t_elems, unconverted_len);
                    ptr::drop_in_place(unconverted_slice);
                }
            }
        }

        let mut u_arr = MaybeUninit::<[U; N]>::uninit();
        let t_elems = self.as_mut_ptr();
        let u_elems = u_arr.as_mut_ptr().cast::<U>();
        let mut guard = Guard::<T, U, N> { t_elems, u_elems, n_converted_elems: 0 };
        mem::forget(self);

        while guard.n_converted_elems < N {
            unsafe {
                let curr_t_elem = t_elems.add(guard.n_converted_elems);
                let curr_u_elem = u_elems.add(guard.n_converted_elems);
                let val = ptr::read(curr_t_elem);
                let converted = f(val)?;
                ptr::write(curr_u_elem, converted);
                guard.n_converted_elems += 1;
            }
        }

        mem::forget(guard);
        Ok(unsafe { u_arr.assume_init() })
    }

    /// Equivalent to
    /// [`<[T; N]>::map`](https://doc.rust-lang.org/stable/std/primitive.array.html#method.map).
    #[inline]
    fn map_in_place(self, f: impl FnMut(T) -> U) -> [U; N] {
        self.map(f)
    }
}

fn vec_helper<T, U, E>(
    mut this: Vec<T>,
    mut f: impl FnMut(T) -> Option<Result<U, E>>,
) -> Result<Vec<U>, E> {
    assert_reuse_compatible::<T, U>();

    if size_of::<U>() == 0 || this.capacity() * size_of::<T>() % size_of::<U>() != 0 {
        return this.into_iter().filter_map(f).collect();
    }

    struct Guard<T, U> {
        elems: *mut T,
        len: usize,
        cap: usize,
        n_converted_elems: usize,
        n_consumed_elems: usize,
        _marker: PhantomData<U>,
    }

    impl<T, U> Drop for Guard<T, U> {
        fn drop(&mut self) {
            unsafe {
                // drop converted elements of type `U`
                let u_elems = self.elems.cast::<U>();
                let converted_slice = slice::from_raw_parts_mut(u_elems, self.n_converted_elems);
                ptr::drop_in_place(converted_slice);

                // the `self.n_converted_elems`-th element is consumed by `f`,
                // and since a panic occurred, there is no returned result to drop

                // drop unconsumed elements of type `T`
                let t_elems = self.elems.add(self.n_consumed_elems + 1);
                let unconsumed_len = self.len - self.n_consumed_elems - 1;
                let unconsumed_slice = slice::from_raw_parts_mut(t_elems, unconsumed_len);
                ptr::drop_in_place(unconsumed_slice);

                mem::drop(Vec::from_raw_parts(self.elems, 0, self.cap));
            }
        }
    }

    // manual `into_raw_parts` for lower MSRV
    let elems = this.as_mut_ptr();
    let len = this.len();
    let cap = this.capacity();
    mem::forget(this);

    let mut guard = Guard {
        elems,
        len,
        cap,
        n_converted_elems: 0,
        n_consumed_elems: 0,
        _marker: PhantomData::<U>,
    };

    while guard.n_consumed_elems < len {
        unsafe {
            let curr_t_elem = elems.add(guard.n_consumed_elems);
            let curr_u_elem = elems.cast::<U>().add(guard.n_converted_elems);
            let val = ptr::read(curr_t_elem);
            let converted = f(val).transpose()?;
            guard.n_consumed_elems += 1;
            if let Some(converted) = converted {
                ptr::write(curr_u_elem, converted);
                guard.n_converted_elems += 1;
            }
        }
    }

    let n_converted_elems = guard.n_converted_elems;
    mem::forget(guard);

    let new_cap = cap * (size_of::<T>() / size_of::<U>());
    Ok(unsafe { Vec::from_raw_parts(elems.cast(), n_converted_elems, new_cap) })
}

impl<T, U> MapInPlace<T, U> for Vec<T> {
    type Output = Vec<U>;

    /// Attempt to map the vector, reusing allocation if possible.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn try_map_in_place<E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<Vec<U>, E> {
        vec_helper(self, move |x| Some(f(x)))
    }

    /// Map the vector, reusing allocation if possible.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn map_in_place(self, mut f: impl FnMut(T) -> U) -> Vec<U> {
        vec_helper::<T, U, Infallible>(self, move |x| Some(Ok(f(x)))).unwrap()
    }
}

impl<T, U> FilterMapInPlace<T, U> for Vec<T> {
    /// Filter and map the vector, reusing allocation if possible.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn filter_map_in_place(self, mut f: impl FnMut(T) -> Option<U>) -> Vec<U> {
        vec_helper::<T, U, Infallible>(self, move |x| f(x).map(Ok)).unwrap()
    }
}

impl<T, U> MapInPlace<T, U> for VecDeque<T> {
    type Output = VecDeque<U>;

    /// Attempt to map the deque, reusing allocation if possible.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn try_map_in_place<E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<VecDeque<U>, E> {
        Vec::from(self).try_map_in_place(f).map(Into::into)
    }

    /// Map the deque, reusing allocation if possible.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn map_in_place(self, f: impl FnMut(T) -> U) -> VecDeque<U> {
        Vec::from(self).map_in_place(f).into()
    }
}

impl<T, U> FilterMapInPlace<T, U> for VecDeque<T> {
    /// Filter and map the deque, reusing allocation if possible.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn filter_map_in_place(self, f: impl FnMut(T) -> Option<U>) -> VecDeque<U> {
        Vec::from(self).filter_map_in_place(f).into()
    }
}

impl<T, U: Ord> MapInPlace<T, U> for BinaryHeap<T> {
    type Output = BinaryHeap<U>;

    /// Attempt to map the heap, reusing allocation if possible.
    ///
    /// This will rebuild the heap, and has *O*(*n*) time complexity.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn try_map_in_place<E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<BinaryHeap<U>, E> {
        Vec::from(self).try_map_in_place(f).map(Into::into)
    }

    /// Map the heap, reusing allocation if possible.
    ///
    /// This will rebuild the heap, and has *O*(*n*) time complexity.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn map_in_place(self, f: impl FnMut(T) -> U) -> BinaryHeap<U> {
        Vec::from(self).map_in_place(f).into()
    }
}

impl<T, U: Ord> FilterMapInPlace<T, U> for BinaryHeap<T> {
    /// Filter and map the heap, reusing allocation if possible.
    ///
    /// This will rebuild the heap, and has *O*(*n*) time complexity.
    ///
    /// If `T`'s size is not a multiple of `U`, this may allocate new memory.
    #[inline]
    fn filter_map_in_place(self, f: impl FnMut(T) -> Option<U>) -> BinaryHeap<U> {
        Vec::from(self).filter_map_in_place(f).into()
    }
}

impl<T: Clone, U> MapInPlace<Self, U> for Cow<'_, T> {
    type Output = U;

    /// Equivalent to calling `f` directly.
    #[inline]
    fn try_map_in_place<E>(self, f: impl FnOnce(Self) -> Result<U, E>) -> Result<U, E> {
        f(self)
    }

    /// Equivalent to calling `f` directly.
    #[inline]
    fn map_in_place(self, f: impl FnOnce(Self) -> U) -> U {
        f(self)
    }
}

impl<'a, T: Clone, U> MapInPlace<Cow<'a, T>, U> for Cow<'a, [T]> {
    type Output = Vec<U>;

    /// Attempt to map the copy-on-write slice.
    ///
    /// The function `f` receives a `Cow` pointer to a single element in the slice or vector.
    #[inline]
    fn try_map_in_place<E>(
        self,
        mut f: impl FnMut(Cow<'a, T>) -> Result<U, E>,
    ) -> Result<Vec<U>, E> {
        match self {
            // collecting to `Result<Vec>` is short-circuit
            Self::Borrowed(slc) => slc.iter().map(move |x| f(Cow::Borrowed(x))).collect(),
            Self::Owned(vec) => vec.try_map_in_place(move |x| f(Cow::Owned(x))),
        }
    }

    /// Map the copy-on-write slice.
    ///
    /// The function `f` receives a `Cow` pointer to a single element in the slice or vector.
    #[inline]
    fn map_in_place(self, mut f: impl FnMut(Cow<'a, T>) -> U) -> Vec<U> {
        match self {
            Self::Borrowed(slc) => slc.iter().map(move |x| f(Cow::Borrowed(x))).collect(),
            Self::Owned(vec) => vec.map_in_place(move |x| f(Cow::Owned(x))),
        }
    }
}

impl<'a, T: Clone, U> FilterMapInPlace<Cow<'a, T>, U> for Cow<'a, [T]> {
    /// Filter and map the copy-on-write slice.
    ///
    /// The function `f` receives a `Cow` pointer to a single element in the slice or vector.
    #[inline]
    fn filter_map_in_place(self, mut f: impl FnMut(Cow<'a, T>) -> Option<U>) -> Vec<U> {
        match self {
            Self::Borrowed(slc) => slc.iter().filter_map(move |x| f(Cow::Borrowed(x))).collect(),
            Self::Owned(vec) => vec.filter_map_in_place(move |x| f(Cow::Owned(x))),
        }
    }
}

macro_rules! impl_smart_pointer {(
    $ptr_ty:ident,
    $($t_bounds:path)?,
    $helper_fn:ident,
    $self_name:ident,
    $pre_into_raw:expr;
    $($doc1:expr)?,
    $($doc2:expr)?,
    $($doc3:expr)?,
    $($doc4:expr)? $(,)?
) => {
    fn $helper_fn<T: $($t_bounds)?, U, E>(
        this: $ptr_ty<T>,
        f: impl FnOnce(T) -> Result<U, E>
    ) -> Result<$ptr_ty<U>, E> {
        assert_in_place_compatible::<T, U>();

        struct Guard<T>(*mut T);

        impl<T> Drop for Guard<T> {
            fn drop(&mut self) {
                // make sure the inner value's destructor is not run
                let ptr = self.0.cast::<ManuallyDrop<T>>();
                mem::drop(unsafe { $ptr_ty::from_raw(ptr) });
            }
        }

        #[allow(unused_mut)]
        let mut $self_name = this;
        #[allow(clippy::no_effect)]
        $pre_into_raw;

        let ptr = $ptr_ty::into_raw($self_name) as *mut T;
        let val = unsafe { ptr::read(ptr) };
        let guard = Guard(ptr);
        let converted = f(val)?;
        mem::forget(guard);
        unsafe {
            ptr::write(ptr.cast(), converted);
            Ok($ptr_ty::from_raw(ptr.cast()))
        }
    }


    impl<T: $($t_bounds)?, U> MapInPlace<T, U> for $ptr_ty<T> {
        type Output = $ptr_ty<U>;

        $(#[doc = $doc1])?
        #[inline]
        fn try_map_in_place<E>(
            self,
            f: impl FnOnce(T) -> Result<U, E>,
        ) -> Result<$ptr_ty<U>, E> {
            $helper_fn(self, f)
        }

        $(#[doc = $doc2])?
        #[inline]
        // independent from `try_map_in_place` to make `f: impl FnOnce` possible
        fn map_in_place(self, f: impl FnOnce(T) -> U) -> $ptr_ty<U> {
            $helper_fn::<T, U, Infallible>(self, |x| Ok(f(x))).unwrap()
        }
    }

    impl<T: $($t_bounds)?, U> MapInPlace<T, U> for $ptr_ty<[T]> {
        type Output = $ptr_ty<[U]>;

        $(#[doc = $doc3])?
        fn try_map_in_place<E>(
            self,
            mut f: impl FnMut(T) -> Result<U, E>
        ) -> Result<$ptr_ty<[U]>, E> {
            assert_in_place_compatible::<T, U>();

            struct Guard<T, U> {
                elems: *mut T,
                len: usize,
                n_converted_elems: usize,
                _marker: PhantomData<U>,
            }

            impl<T, U> Drop for Guard<T, U> {
                fn drop(&mut self) {
                    unsafe {
                        // drop converted elements of type `U`
                        let u_elems = self.elems.cast::<U>();
                        let converted_slice =
                            slice::from_raw_parts_mut(u_elems, self.n_converted_elems);
                        ptr::drop_in_place(converted_slice);

                        // the `self.n_converted_elems`-th element is consumed by `f`,
                        // and since a panic occurred, there is no returned result to drop

                        // drop unconverted elements of type `T`
                        let t_elems = self.elems.add(self.n_converted_elems + 1);
                        let unconverted_len = self.len - self.n_converted_elems - 1;
                        let unconverted_slice = slice::from_raw_parts_mut(t_elems, unconverted_len);
                        ptr::drop_in_place(unconverted_slice);

                        // make sure the inner value's destructor is not run
                        let elems = self.elems.cast::<ManuallyDrop<T>>();
                        let slice = slice::from_raw_parts_mut(elems, self.len);
                        mem::drop($ptr_ty::from_raw(slice as *mut [ManuallyDrop<T>]))
                    }
                }
            }

            #[allow(unused_mut)]
            let mut $self_name = self;
            #[allow(clippy::no_effect)]
            $pre_into_raw;

            let len = $self_name.len();
            let elems = $ptr_ty::into_raw($self_name) as *mut T;
            let mut guard =
                Guard { elems, len, n_converted_elems: 0, _marker: PhantomData::<U> };

            while guard.n_converted_elems < len {
                unsafe {
                    let curr_elem = elems.add(guard.n_converted_elems);
                    let val = ptr::read(curr_elem);
                    let converted = f(val)?;
                    ptr::write(curr_elem.cast(), converted);
                    guard.n_converted_elems += 1;
                }
            }

            mem::forget(guard);
            unsafe {
                let slice = slice::from_raw_parts_mut(elems.cast(), len);
                Ok($ptr_ty::from_raw(slice))
            }
        }

        $(#[doc = $doc4])?
        fn map_in_place(self, mut f: impl FnMut(T) -> U) -> $ptr_ty<[U]> {
            self.try_map_in_place::<Infallible>(move |x| Ok(f(x))).unwrap()
        }
    }
};}

macro_rules! rc_arc_doc {
    ($ptr_ty:literal, $how_to_map:literal, $inner_ty:literal $(, $func:literal)?) => {
        concat!(
            $how_to_map, " the ", $inner_ty, " in an `", $ptr_ty,
            "`, reusing the allocation if possible.\n\nIf there are other `", $ptr_ty,
            "` pointers to the same allocation, then this will clone the inner ", $inner_ty,
            " to a new allocation before mapping.\n\nHowever, if there are no other `", $ptr_ty,
            "` pointers to this allocation, but some `Weak` pointers, then the `Weak` pointers \
             will be disassociated and the inner ", $inner_ty,
            " will not be cloned, but moved to a new allocation.",
            $("\n\nSimilar to [`", $func, "`], but `f` takes ownership of the value, and does not \
            accept types of different sizes.")?
        )
    };
}

impl_smart_pointer! {
    Box, , box_helper,
    this, ();
    "Similar to [`Box::try_map`], but does not accept types of different sizes.",
    "Similar to [`Box::map`], but does not accept types of different sizes.", ,
}

impl_smart_pointer! {
    Rc, Clone, rc_helper,
    this, Rc::make_mut(&mut this);
    rc_arc_doc!("Rc", "Attempt to map", "value", "Rc::try_map"),
    rc_arc_doc!("Rc", "Map", "value", "Rc::map"),
    rc_arc_doc!("Rc", "Attempt to map", "slice"),
    rc_arc_doc!("Rc", "Map", "slice"),
}

impl_smart_pointer! {
    Arc, Clone, arc_helper,
    this, Arc::make_mut(&mut this);
    rc_arc_doc!("Arc", "Attempt to map", "value", "Arc::try_map"),
    rc_arc_doc!("Arc", "Map", "value", "Arc::map"),
    rc_arc_doc!("Arc", "Attempt to map", "slice"),
    rc_arc_doc!("Arc", "Map", "slice"),
}
