use magnus::{
    define_module, function, method, Error, Module, Object, RArray,
    RFloat, Ruby, TryConvert, Value,
};
use magnus::value::ReprValue;


#[derive(Debug)]
#[magnus::wrap(class = "Embedding")]
pub struct Embedding {
    pub data: Vec<f64>, // Stores the embedding vector as a Vec of f64
}

impl Embedding {
    /// Creates a new, empty Embedding.
    fn new() -> Self {
        Self { data: Vec::new() }
    }
}

// No need for Marker trait in Magnus 0.6.4

// Ruby method implementations

/// Ruby 'new' method: creates a new Embedding instance.
fn embedding_new(_ruby: &Ruby) -> Result<Embedding, Error> {
    Ok(Embedding::new())
}

/// Ruby 'initialize' method: initializes Embedding with a Ruby array of numbers.
fn embedding_initialize(rb_self: &mut Embedding, rb_array: RArray) -> Result<(), Error> {
    let mut data = Vec::new();

    // Convert each Ruby array element to f64 and push to data vector
    for i in 0..rb_array.len() {
        let val: Value = rb_array.entry(i as isize)?;

        let float_val = if let Some(f) = RFloat::from_value(val) {
            f.to_f64()
        } else if let Ok(i) = i64::try_convert(val) {
            i as f64
        } else {
            // Raise error if element is not numeric
            return Err(Error::new(
                Ruby::get().unwrap().exception_type_error(),
                "Array elements must be numeric",
            ));
        };

        data.push(float_val);
    }

    rb_self.data = data;
    Ok(())
}

/// Converts the embedding to a Ruby array.
fn embedding_to_a(rb_self: &Embedding) -> Result<RArray, Error> {
    let ruby = Ruby::get().unwrap();
    let arr = RArray::new();

    // Push each f64 value as a Ruby value into the array
    for &value in &rb_self.data {
        arr.push(ruby.into_value(value))?;
    }

    Ok(arr)
}

/// Returns the size (length) of the embedding vector.
fn embedding_size(rb_self: &Embedding) -> usize {
    rb_self.data.len()
}

/// Gets the value at the given index, or None if out of bounds.
fn embedding_get(rb_self: &Embedding, index: isize) -> Result<Option<f64>, Error> {
    if index < 0 {
        return Ok(None);
    }

    let idx = index as usize;
    if idx >= rb_self.data.len() {
        return Ok(None);
    }

    Ok(Some(rb_self.data[idx]))
}

/// Sets the value at the given index, resizing if necessary.
fn embedding_set(rb_self: &mut Embedding, index: isize, value: f64) -> Result<(), Error> {
    if index < 0 {
        return Err(Error::new(
            Ruby::get().unwrap().exception_index_error(),
            "negative index",
        ));
    }

    let idx = index as usize;
    // Resize vector if index is out of bounds, filling with 0.0
    if idx >= rb_self.data.len() {
        rb_self.data.resize(idx + 1, 0.0);
    }

    rb_self.data[idx] = value;
    Ok(())
}

/// Normalizes the embedding vector in-place (L2 norm).
fn embedding_normalize_bang(rb_self: &mut Embedding) -> Result<Value, Error> {
    // Compute the magnitude (L2 norm)
    let magnitude: f64 = rb_self.data.iter().map(|x| x * x).sum::<f64>().sqrt();

    if magnitude == 0.0 {
        // Cannot normalize a zero vector
        return Err(Error::new(
            Ruby::get().unwrap().exception_runtime_error(),
            "Cannot normalize zero vector",
        ));
    }

    // Divide each element by the magnitude
    for value in &mut rb_self.data {
        *value /= magnitude;
    }

    let ruby = Ruby::get().unwrap();
    Ok(ruby.qnil().as_value())
}

/// Computes the dot product with another embedding.
fn embedding_dot_product(rb_self: &Embedding, other: &Embedding) -> Result<f64, Error> {
    if rb_self.data.len() != other.data.len() {
        // Vectors must be the same length
        return Err(Error::new(
            Ruby::get().unwrap().exception_arg_error(),
            "Vectors must have the same dimension",
        ));
    }

    // Sum of element-wise products
    let dot_product = rb_self
        .data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| a * b)
        .sum();

    Ok(dot_product)
}

/// Computes the magnitude (L2 norm) of the embedding.
fn embedding_magnitude(rb_self: &Embedding) -> f64 {
    rb_self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Computes the Euclidean distance to another embedding.
fn embedding_distance(rb_self: &Embedding, other: &Embedding) -> Result<f64, Error> {
    if rb_self.data.len() != other.data.len() {
        // Vectors must be the same length
        return Err(Error::new(
            Ruby::get().unwrap().exception_arg_error(),
            "Vectors must have the same dimension",
        ));
    }

    // Square root of sum of squared differences
    let distance = rb_self
        .data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(distance)
}

/// Computes the cosine similarity with another embedding.
fn embedding_cosine_similarity(rb_self: &Embedding, other: &Embedding) -> Result<f64, Error> {
    let dot_product = embedding_dot_product(rb_self, other)?;
    let magnitude_self = embedding_magnitude(rb_self);
    let magnitude_other = embedding_magnitude(other);

    // If either vector is zero, similarity is 0
    if magnitude_self == 0.0 || magnitude_other == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (magnitude_self * magnitude_other))
}

/// Initializes the Ruby module and class, and defines all methods.
#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = define_module("RagEmbeddings")?;
    let class = module.define_class("Embedding", ruby.class_object())?;

    // Define Ruby methods for the Embedding class
    class.define_singleton_method("new", function!(embedding_new, 0))?;
    class.define_method("initialize", method!(embedding_initialize, 1))?;
    class.define_method("to_a", method!(embedding_to_a, 0))?;
    class.define_method("size", method!(embedding_size, 0))?;
    class.define_method("length", method!(embedding_size, 0))?;
    class.define_method("[]", method!(embedding_get, 1))?;
    class.define_method("[]=", method!(embedding_set, 2))?;
    class.define_method("normalize!", method!(embedding_normalize_bang, 0))?;
    class.define_method("dot_product", method!(embedding_dot_product, 1))?;
    class.define_method("magnitude", method!(embedding_magnitude, 0))?;
    class.define_method("distance", method!(embedding_distance, 1))?;
    class.define_method("cosine_similarity", method!(embedding_cosine_similarity, 1))?;

    Ok(())
}