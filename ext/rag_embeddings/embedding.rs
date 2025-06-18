use magnus::{function, method, prelude::*, Error, Ruby, DataTypeFunctions, TypedData};
use std::cell::RefCell;

#[derive(TypedData)]
#[magnus(class = "RagEmbeddings::Embedding", free_immediately)]
struct Embedding {
    values: RefCell<Vec<f32>>,
}

impl DataTypeFunctions for Embedding {
    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.values.borrow().capacity() * std::mem::size_of::<f32>()
    }
}

impl Embedding {
    fn from_array(arr: Vec<f32>) -> Result<Self, Error> {
        if arr.is_empty() {
            return Err(Error::new(
                magnus::exception::arg_error(),
                "Cannot create embedding from empty array",
            ));
        }
        if arr.len() > u16::MAX as usize {
            return Err(Error::new(
                magnus::exception::arg_error(),
                format!(
                    "Array too large: maximum {} dimensions allowed",
                    u16::MAX
                ),
            ));
        }
        Ok(Self {
            values: RefCell::new(arr),
        })
    }

    fn dim(&self) -> usize {
        self.values.borrow().len()
    }

    fn to_a(&self) -> Vec<f32> {
        self.values.borrow().clone()
    }

    fn cosine_similarity(&self, other: &Embedding) -> Result<f64, Error> {
        let a = self.values.borrow();
        let b = other.values.borrow();
        if a.len() != b.len() {
            return Err(Error::new(
                magnus::exception::arg_error(),
                format!("Dimension mismatch: {} vs {}", a.len(), b.len()),
            ));
        }
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (ai, bi) in a.iter().zip(b.iter()) {
            dot += *ai as f64 * *bi as f64;
            norm_a += (*ai as f64) * (*ai as f64);
            norm_b += (*bi as f64) * (*bi as f64);
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        let sim = dot / (norm_a * norm_b).sqrt();
        Ok(sim.clamp(-1.0, 1.0))
    }

    fn magnitude(&self) -> f64 {
        let a = self.values.borrow();
        let mut sum = 0.0f64;
        for v in a.iter() {
            sum += (*v as f64) * (*v as f64);
        }
        sum.sqrt()
    }

    fn normalize_bang(&self) -> Result<(), Error> {
        let mut values = self.values.borrow_mut();
        let mut sum = 0.0f64;
        for v in values.iter() {
            sum += (*v as f64) * (*v as f64);
        }
        let magnitude = sum.sqrt();
        if magnitude == 0.0 {
            return Err(Error::new(
                magnus::exception::zero_div_error(),
                "Cannot normalize zero vector",
            ));
        }
        let inv_mag = 1.0 / magnitude as f32;
        for v in values.iter_mut() {
            *v *= inv_mag;
        }
        Ok(())
    }
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let m_rag = ruby.define_module("RagEmbeddings")?;
    let class = m_rag.define_class("Embedding", ruby.class_object())?;
    class.undef_default_alloc_func();
    class.define_singleton_method("from_array", function!(Embedding::from_array, 1))?;
    class.define_method("dim", method!(Embedding::dim, 0))?;
    class.define_method("to_a", method!(Embedding::to_a, 0))?;
    class.define_method("cosine_similarity", method!(Embedding::cosine_similarity, 1))?;
    class.define_method("magnitude", method!(Embedding::magnitude, 0))?;
    class.define_method("normalize!", method!(Embedding::normalize_bang, 0))?;
    Ok(())
}
