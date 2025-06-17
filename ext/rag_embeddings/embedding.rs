use magnus::{
    define_module, function, method, Error, Module, Object, RArray,
    RFloat, Ruby, TryConvert, Value,
};
use magnus::value::ReprValue;


#[derive(Debug)]
#[magnus::wrap(class = "Embedding")]
pub struct Embedding {
    pub data: Vec<f64>,
}

impl Embedding {
    fn new() -> Self {
        Self { data: Vec::new() }
    }
}

// No need for Marker trait in Magnus 0.6.4

// Ruby method implementations
fn embedding_new(_ruby: &Ruby) -> Result<Embedding, Error> {
    Ok(Embedding::new())
}

fn embedding_initialize(rb_self: &mut Embedding, rb_array: RArray) -> Result<(), Error> {
    let mut data = Vec::new();

    for i in 0..rb_array.len() {
        let val: Value = rb_array.entry(i as isize)?;

        let float_val = if let Some(f) = RFloat::from_value(val) {
            f.to_f64()
        } else if let Ok(i) = i64::try_convert(val) {
            i as f64
        } else {
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

fn embedding_to_a(rb_self: &Embedding) -> Result<RArray, Error> {
    let ruby = Ruby::get().unwrap();
    let arr = RArray::new();

    for &value in &rb_self.data {
        arr.push(ruby.into_value(value))?;
    }

    Ok(arr)
}

fn embedding_size(rb_self: &Embedding) -> usize {
    rb_self.data.len()
}

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

fn embedding_set(rb_self: &mut Embedding, index: isize, value: f64) -> Result<(), Error> {
    if index < 0 {
        return Err(Error::new(
            Ruby::get().unwrap().exception_index_error(),
            "negative index",
        ));
    }

    let idx = index as usize;
    if idx >= rb_self.data.len() {
        rb_self.data.resize(idx + 1, 0.0);
    }

    rb_self.data[idx] = value;
    Ok(())
}

fn embedding_normalize_bang(rb_self: &mut Embedding) -> Result<Value, Error> {
    let magnitude: f64 = rb_self.data.iter().map(|x| x * x).sum::<f64>().sqrt();

    if magnitude == 0.0 {
        return Err(Error::new(
            Ruby::get().unwrap().exception_runtime_error(),
            "Cannot normalize zero vector",
        ));
    }

    for value in &mut rb_self.data {
        *value /= magnitude;
    }

    let ruby = Ruby::get().unwrap();
    Ok(ruby.qnil().as_value())
}

fn embedding_dot_product(rb_self: &Embedding, other: &Embedding) -> Result<f64, Error> {
    if rb_self.data.len() != other.data.len() {
        return Err(Error::new(
            Ruby::get().unwrap().exception_arg_error(),
            "Vectors must have the same dimension",
        ));
    }

    let dot_product = rb_self
        .data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| a * b)
        .sum();

    Ok(dot_product)
}

fn embedding_magnitude(rb_self: &Embedding) -> f64 {
    rb_self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn embedding_distance(rb_self: &Embedding, other: &Embedding) -> Result<f64, Error> {
    if rb_self.data.len() != other.data.len() {
        return Err(Error::new(
            Ruby::get().unwrap().exception_arg_error(),
            "Vectors must have the same dimension",
        ));
    }

    let distance = rb_self
        .data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(distance)
}

fn embedding_cosine_similarity(rb_self: &Embedding, other: &Embedding) -> Result<f64, Error> {
    let dot_product = embedding_dot_product(rb_self, other)?;
    let magnitude_self = embedding_magnitude(rb_self);
    let magnitude_other = embedding_magnitude(other);

    if magnitude_self == 0.0 || magnitude_other == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (magnitude_self * magnitude_other))
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = define_module("RagEmbeddings")?;
    let class = module.define_class("Embedding", ruby.class_object())?;

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