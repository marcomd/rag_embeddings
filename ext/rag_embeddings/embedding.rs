use magnus::{
    define_class, define_module, function, method, prelude::*, Error, Module, Object, RArray,
    RFloat, Ruby, TypedData, Value,
};
use std::fmt;

/// Main data structure for storing embeddings
/// Uses Vec<f32> for flexibility and Rust's memory safety
#[derive(Debug, Clone)]
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding from a vector of floats
    pub fn new(values: Vec<f32>) -> Result<Self, &'static str> {
        if values.is_empty() {
            return Err("Cannot create embedding from empty array");
        }
        if values.len() > u16::MAX as usize {
            return Err("Array too large: maximum 65535 dimensions allowed");
        }
        Ok(Self { values })
    }

    /// Get the dimension of the embedding
    pub fn dim(&self) -> u16 {
        self.values.len() as u16
    }

    /// Get the values as a slice
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Calculate cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f64, &'static str> {
        if self.values.len() != other.values.len() {
            return Err("Dimension mismatch");
        }

        // Use double precision for intermediate calculations to reduce accumulation errors
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        // Calculate dot product and vector magnitudes in a single loop
        // This is more cache-friendly than separate loops
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let a_f64 = *a as f64;
            let b_f64 = *b as f64;

            dot += a_f64 * b_f64;
            norm_a += a_f64 * a_f64;
            norm_b += b_f64 * b_f64;
        }

        // Check for zero vectors to avoid division by zero
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        // Apply cosine similarity formula: dot(a,b)/(|a|*|b|)
        let magnitude_product = (norm_a * norm_b).sqrt();
        let similarity = dot / magnitude_product;

        // Clamp result to [-1, 1] to handle floating point precision errors
        Ok(similarity.clamp(-1.0, 1.0))
    }

    /// Calculate the magnitude (L2 norm) of the embedding vector
    pub fn magnitude(&self) -> f64 {
        let sum_squares: f64 = self.values.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        sum_squares.sqrt()
    }

    /// Normalize the embedding vector in-place (destructive operation)
    pub fn normalize(&mut self) -> Result<(), &'static str> {
        let magnitude = self.magnitude();

        if magnitude == 0.0 {
            return Err("Cannot normalize zero vector");
        }

        let inv_magnitude = 1.0 / magnitude;
        for value in &mut self.values {
            *value *= inv_magnitude as f32;
        }

        Ok(())
    }
}

impl fmt::Display for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Embedding(dim: {}, values: {:?})", self.dim(), self.values)
    }
}

// Make Embedding work with Ruby's TypedData
unsafe impl TypedData for Embedding {
    fn class() -> magnus::RClass {
        *memoize!(magnus::RClass: {
            let module = define_module("RagEmbeddings").unwrap();
            define_class(module, "Embedding", magnus::class::object()).unwrap()
        })
    }

    fn data_type() -> &'static magnus::rb_data_type_t {
        memoize!(magnus::rb_data_type_t: {
            magnus::rb_data_type_t {
                wrap_struct_name: c_str!("RagEmbeddings::Embedding"),
                function: magnus::typed_data::Functions::new::<Self>(),
                parent: std::ptr::null(),
                data: std::ptr::null_mut(),
                flags: magnus::typed_data::DataTypeFlags::WB_PROTECTED,
            }
        })
    }
}

/// Class method: RagEmbeddings::Embedding.from_array([1.0, 2.0, ...])
/// Creates a new embedding from a Ruby array
fn embedding_from_array(rb_array: RArray) -> Result<Embedding, Error> {
    let array_len = rb_array.len();

    if array_len == 0 {
        return Err(Error::new(
            magnus::exception::arg_error(),
            "Cannot create embedding from empty array",
        ));
    }

    if array_len > u16::MAX as usize {
        return Err(Error::new(
            magnus::exception::arg_error(),
            format!("Array too large: maximum {} dimensions allowed", u16::MAX),
        ));
    }

    let mut values = Vec::with_capacity(array_len);

    // Convert Ruby array elements to f32
    for i in 0..array_len {
        let val: Value = rb_array.entry(i)?;

        // Try to convert to float
        let float_val = if let Ok(f) = RFloat::from_value(val) {
            f.to_f64() as f32
        } else if let Ok(i) = val.try_convert::<i64>() {
            i as f32
        } else {
            return Err(Error::new(
                magnus::exception::type_error(),
                format!("Array element at index {} is not numeric", i),
            ));
        };

        values.push(float_val);
    }

    Embedding::new(values).map_err(|msg| {
        Error::new(magnus::exception::arg_error(), msg)
    })
}

/// Instance method: embedding.dim
/// Returns the dimension of the embedding
fn embedding_dim(rb_self: &Embedding) -> u16 {
    rb_self.dim()
}

/// Instance method: embedding.to_a
/// Converts the embedding back to a Ruby array
fn embedding_to_a(ruby: &Ruby, rb_self: &Embedding) -> Result<RArray, Error> {
    let arr = RArray::with_capacity(rb_self.values.len());

    for &value in &rb_self.values {
        arr.push(ruby.to_value(value as f64))?;
    }

    Ok(arr)
}

/// Instance method: embedding.cosine_similarity(other_embedding)
/// Calculate cosine similarity between two embeddings
fn embedding_cosine_similarity(
    rb_self: &Embedding,
    other: &Embedding,
) -> Result<f64, Error> {
    rb_self.cosine_similarity(other).map_err(|msg| {
        Error::new(magnus::exception::arg_error(), msg)
    })
}

/// Instance method: embedding.magnitude
/// Calculate the magnitude (L2 norm) of the embedding vector
fn embedding_magnitude(rb_self: &Embedding) -> f64 {
    rb_self.magnitude()
}

/// Instance method: embedding.normalize!
/// Normalize the embedding vector in-place (destructive operation)
fn embedding_normalize_bang(rb_self: &mut Embedding) -> Result<Value, Error> {
    rb_self.normalize().map_err(|msg| {
        Error::new(magnus::exception::zero_div_error(), msg)
    })?;

    // Return self for method chaining
    Ok(Ruby::get().unwrap().qself())
}

/// Initialize the Ruby extension
#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    // Define module and class
    let module = define_module("RagEmbeddings")?;
    let class = define_class(module, "Embedding", ruby.class_object())?;

    // Register class methods
    class.define_singleton_method("from_array", function!(embedding_from_array, 1))?;

    // Register instance methods
    class.define_method("dim", method!(embedding_dim, 0))?;
    class.define_method("to_a", method!(embedding_to_a, 0))?;
    class.define_method("cosine_similarity", method!(embedding_cosine_similarity, 1))?;
    class.define_method("magnitude", method!(embedding_magnitude, 0))?;
    class.define_method("normalize!", method!(embedding_normalize_bang, 0))?;

    Ok(())
}