# in_place_map

This crate provides the functionality of converting between the same container type of different element types, reusing allocation if possible.

The main use case of this crate is converting a collection to that of another element type, reusing its allocation when the layouts are compatible. For example, convert a `Vec<i32>` to a `Vec<u32>`:

```rust
use in_place_map::MapInPlace;
let i_vec = vec![1i32, -2, 3, -4];
let addr = i_vec.as_ptr() as usize;
let u_vec = i_vec.map_in_place(i32::unsigned_abs);
assert_eq!(u_vec, [1u32, 2, 3, 4]);
assert_eq!(u_vec.as_ptr() as usize, addr);
```

The `MapInPlace` trait also aims to be a generalized `map` and `try_map` interface for other container types. Some `std` types already have those methods, but are not yet stabilized, so this crate can also be a stable alternative:

```rust
use in_place_map::MapInPlace;
let i_arr = [1i32, 2, 3, 4];
let u_arr = i_arr.try_map_in_place(u32::try_from);
assert!(u_arr.is_ok());
assert_eq!(u_arr.unwrap(), [1u32, 2, 3, 4]);
```

Another useful trait is `FilterMapInPlace`, which filters the collection while mapping it:

```rust
use in_place_map::FilterMapInPlace;
let i_vec = vec![1i32, 2, 3, 4];
let u_vec = i_vec.filter_map_in_place(|x| (x % 2 == 0).then_some(x as u32));
assert_eq!(u_vec, [2u32, 4]);
```
