use std::borrow::Cow;
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::Arc;
use std::{array, mem};

use super::{FilterMapInPlace, MapInPlace};

#[test]
fn array() {
    let arr = [1i32, 2, 3, 4];

    let mapped = arr.try_map_in_place::<()>(|x| Ok(x as u64 * 10)).unwrap();
    assert_eq!(mapped, [10, 20, 30, 40]);

    let mapped = arr.try_map_in_place::<()>(|x| Ok(format!("x={x}"))).unwrap();
    assert_eq!(mapped, ["x=1", "x=2", "x=3", "x=4"]);

    // the mapper must not be run for zero length arrays
    let _: [usize; 0] = [(); 0].try_map_in_place::<()>(|_| unreachable!()).unwrap();
}

#[test]
fn vec() {
    // simple mapping

    let mut vec = Vec::with_capacity(8);
    vec.extend([1i32, 2, 3, 4]);
    let addr = vec.as_ptr() as usize;
    let cap = vec.capacity();

    let mapped = vec.map_in_place(|x| x as u32 * 10);
    assert_eq!(mapped, [10, 20, 30, 40]);
    assert_eq!(mapped.as_ptr() as usize, addr);
    assert_eq!(mapped.capacity(), cap);

    // different sized types

    let mut vec = Vec::with_capacity(8);
    vec.extend([(1i32, 2), (3, 4), (5, 6), (7, 8)]);
    let addr = vec.as_ptr() as usize;
    let cap = vec.capacity();

    let mapped = vec.map_in_place(|(x, y)| (x + y) as u32);
    assert_eq!(mapped, [3, 7, 11, 15]);
    assert_eq!(mapped.as_ptr() as usize, addr);
    assert_eq!(mapped.capacity(), cap * 2);

    // filter mapping

    let mut vec = Vec::with_capacity(8);
    vec.extend([1i32, 2, 3, 4]);
    let addr = vec.as_ptr() as usize;
    let cap = vec.capacity();

    let mapped = vec.filter_map_in_place(|x| (x % 2 == 0).then_some(x as u32 + 10));
    assert_eq!(mapped, [12, 14]);
    assert_eq!(mapped.as_ptr() as usize, addr);
    assert_eq!(mapped.capacity(), cap);

    // ZST

    let mapped = vec![1, 2, 3, 4].map_in_place(|_| ());
    assert_eq!(mapped.len(), 4);

    let mapped = vec![(); 4].map_in_place(|_| ());
    assert_eq!(mapped.len(), 4);
}

#[test]
fn cow_slice() {
    let arr = [1i32, -2, 3, -4];
    let borrowed: Cow<'_, [i32]> = Cow::Borrowed(&arr);
    let mapped = borrowed.map_in_place(|x| match x {
        Cow::Borrowed(&x) => x.unsigned_abs(),
        Cow::Owned(_) => unreachable!(),
    });
    assert_eq!(mapped, [1, 2, 3, 4]);

    let vec = arr.to_vec();
    let addr = vec.as_ptr() as usize;
    let owned: Cow<'_, [i32]> = Cow::Owned(vec);
    let mapped = owned.map_in_place(|x| match x {
        Cow::Owned(x) => x.unsigned_abs(),
        Cow::Borrowed(_) => unreachable!(),
    });
    assert_eq!(mapped, [1, 2, 3, 4]);
    assert_eq!(mapped.as_ptr() as usize, addr);
}

#[test]
fn boxx() {
    let val = Box::new(String::from("hello"));
    let addr = (&*val) as *const _ as usize;
    let mapped = val.map_in_place(|x| x.to_ascii_uppercase());
    assert_eq!(*mapped, "HELLO");
    assert_eq!((&*mapped) as *const _ as usize, addr);

    let slice = vec![1i32, 2, 3, 4].into_boxed_slice();
    let addr = slice.as_ptr() as usize;
    let mapped = slice.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, [10, 20, 30, 40]);
    assert_eq!(mapped.as_ptr() as usize, addr);
}

#[test]
fn rc_arc_unique() {
    let rc = Rc::new(12i32);
    let weak = Rc::downgrade(&rc);
    let addr = Rc::as_ptr(&rc).cast::<i32>() as usize;
    let mapped = rc.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, 120);
    // when there are no other `Rc`s but `Weak`s, the value will be moved to a new allocation
    assert_ne!(Rc::as_ptr(&mapped) as usize, addr);
    assert!(weak.upgrade().is_none());

    let rc: Rc<[i32]> = Rc::from(vec![1, 2, 3, 4]);
    let addr = Rc::as_ptr(&rc).cast::<i32>() as usize;
    let mapped = rc.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, [10, 20, 30, 40]);
    assert_eq!(Rc::as_ptr(&mapped).cast::<u32>() as usize, addr);

    let arc = Arc::new(12i32);
    let weak = Arc::downgrade(&arc);
    let addr = Arc::as_ptr(&arc).cast::<i32>() as usize;
    let mapped = arc.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, 120);
    // when there are no other `Arc`s but `Weak`s, the value will be moved to a new allocation
    assert_ne!(Arc::as_ptr(&mapped) as usize, addr);
    assert!(weak.upgrade().is_none());

    let arc: Arc<[i32]> = Arc::from(vec![1, 2, 3, 4]);
    let addr = Arc::as_ptr(&arc).cast::<i32>() as usize;
    let mapped = arc.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, [10, 20, 30, 40]);
    assert_eq!(Arc::as_ptr(&mapped).cast::<u32>() as usize, addr);
}

#[test]
fn rc_arc_shared() {
    let rc = Rc::new(12i32);
    let other = Rc::clone(&rc);
    let weak = Rc::downgrade(&rc);
    let addr = Rc::as_ptr(&rc) as usize;
    let mapped = rc.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, 120);
    assert_eq!(*other, 12);
    assert_ne!(Rc::as_ptr(&mapped) as usize, addr);
    assert!(weak.upgrade().is_some());

    let arc = Arc::new(12i32);
    let other = Arc::clone(&arc);
    let weak = Arc::downgrade(&arc);
    let addr = Arc::as_ptr(&arc) as usize;
    let mapped = arc.map_in_place(|x| x as u32 * 10);
    assert_eq!(*mapped, 120);
    assert_eq!(*other, 12);
    assert_ne!(Arc::as_ptr(&mapped) as usize, addr);
    assert!(weak.upgrade().is_some());
}

#[derive(Debug, Clone)]
struct DropProbe {
    drop_count: Rc<Cell<usize>>,
    id: usize,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.drop_count.set(self.drop_count.get() + 1);
    }
}

fn probes<const N: usize>(drop_count: &Rc<Cell<usize>>) -> [DropProbe; N] {
    array::from_fn(|id| DropProbe { drop_count: Rc::clone(drop_count), id })
}

#[test]
fn drop_on_error() {
    let drop_count = Rc::new(Cell::new(0));
    let drop_count_2 = Rc::new(Cell::new(0));
    let mapper = |probe: DropProbe| {
        if probe.id == 6 {
            Err(())
        } else {
            let new = DropProbe { drop_count: drop_count_2.clone(), ..probe };
            mem::forget(probe);
            Ok(new)
        }
    };

    // array
    probes::<10>(&drop_count).try_map_in_place(&mapper).unwrap_err();
    assert_eq!(drop_count.get(), 4); // unconverted
    assert_eq!(drop_count_2.get(), 6); // converted

    // vector
    drop_count.set(0);
    drop_count_2.set(0);
    Vec::from(probes::<10>(&drop_count)).try_map_in_place(&mapper).unwrap_err();
    assert_eq!(drop_count.get(), 4); // unconverted
    assert_eq!(drop_count_2.get(), 6); // converted

    // boxed slice
    drop_count.set(0);
    drop_count_2.set(0);
    Vec::from(probes::<10>(&drop_count)).into_boxed_slice().try_map_in_place(&mapper).unwrap_err();
    assert_eq!(drop_count.get(), 4); // unconverted
    assert_eq!(drop_count_2.get(), 6); // converted
}

#[test]
fn drop_on_panic() {
    fn catch<T>(f: impl FnOnce() -> T) {
        let _ = catch_unwind(AssertUnwindSafe(f));
    }

    let drop_count = Rc::new(Cell::new(0));
    let drop_count_2 = Rc::new(Cell::new(0));
    let mapper = |probe: DropProbe| {
        if probe.id == 6 {
            panic!()
        } else {
            let new = DropProbe { drop_count: drop_count_2.clone(), ..probe };
            mem::forget(probe);
            new
        }
    };

    // array
    catch(|| probes::<10>(&drop_count).map_in_place(&mapper));
    assert_eq!(drop_count.get(), 4); // unconverted
    assert_eq!(drop_count_2.get(), 6); // converted

    // vector
    drop_count.set(0);
    drop_count_2.set(0);
    catch(|| Vec::from(probes::<10>(&drop_count)).map_in_place(&mapper));
    assert_eq!(drop_count.get(), 4); // unconverted
    assert_eq!(drop_count_2.get(), 6); // converted

    // boxed slice
    drop_count.set(0);
    drop_count_2.set(0);
    catch(|| Vec::from(probes::<10>(&drop_count)).into_boxed_slice().map_in_place(&mapper));
    assert_eq!(drop_count.get(), 4); // unconverted
    assert_eq!(drop_count_2.get(), 6); // converted
}
