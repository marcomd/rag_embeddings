#include <ruby.h>     // Ruby API
#include <stdint.h>   // For integer types like uint16_t
#include <stdlib.h>   // For memory allocation functions
#include <math.h>     // For math functions like sqrt

// Main data structure for storing embeddings
// Flexible array member (values[]) allows variable length arrays
typedef struct {
  uint16_t dim;       // Dimension of the embedding vector
  float values[];     // Flexible array member to store the actual values
} embedding_t;

// Callback for freeing memory when Ruby's GC collects our object
static void embedding_free(void *ptr) {
  if (ptr) {
    xfree(ptr);       // Ruby's memory free function (with null check)
  }
}

// Callback to report memory usage to Ruby's GC
static size_t embedding_memsize(const void *ptr) {
  const embedding_t *emb = (const embedding_t *)ptr;
  return emb ? sizeof(embedding_t) + emb->dim * sizeof(float) : 0;
}

// Type information for Ruby's GC:
// Tells Ruby how to manage our C data structure
static const rb_data_type_t embedding_type = {
  "RagEmbeddings/Embedding",               // Type name
  {0, embedding_free, embedding_memsize,}, // Functions: mark, free, size
  0, 0,                                    // Parent type, data
  RUBY_TYPED_FREE_IMMEDIATELY              // Flags for immediate cleanup
};

// Class method: RagEmbeddings::Embedding.from_array([1.0, 2.0, ...])
// Creates a new embedding from a Ruby array
static VALUE embedding_from_array(VALUE klass, VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);           // Ensure argument is a Ruby array

  long array_len = RARRAY_LEN(rb_array);

  // Validate array length fits in uint16_t (max 65535 dimensions)
  if (array_len > UINT16_MAX) {
    rb_raise(rb_eArgError, "Array too large: maximum %d dimensions allowed", UINT16_MAX);
  }

  // Prevent zero-length embeddings
  if (array_len == 0) {
    rb_raise(rb_eArgError, "Cannot create embedding from empty array");
  }

  uint16_t dim = (uint16_t)array_len;

  // Allocate memory for struct + array of floats
  embedding_t *ptr = xmalloc(sizeof(embedding_t) + dim * sizeof(float));
  ptr->dim = dim;

  // Copy values from Ruby array to our C array
  // Using RARRAY_CONST_PTR for better performance when available
  const VALUE *array_ptr = RARRAY_CONST_PTR(rb_array);
  for (uint16_t i = 0; i < dim; ++i) {
    VALUE val = array_ptr[i];

    // Ensure the value is numeric
    if (!RB_FLOAT_TYPE_P(val) && !RB_INTEGER_TYPE_P(val)) {
      xfree(ptr);  // Clean up allocated memory before raising exception
      rb_raise(rb_eTypeError, "Array element at index %d is not numeric", i);
    }

    ptr->values[i] = (float)NUM2DBL(val);
  }

  // Wrap our C struct in a Ruby object
  VALUE obj = TypedData_Wrap_Struct(klass, &embedding_type, ptr);
  return obj;
}

// Instance method: embedding.dim
// Returns the dimension of the embedding
static VALUE embedding_dim(VALUE self) {
  embedding_t *ptr;
  // Get the C struct from the Ruby object
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);
  return INT2NUM(ptr->dim);
}

// Instance method: embedding.to_a
// Converts the embedding back to a Ruby array
static VALUE embedding_to_a(VALUE self) {
  embedding_t *ptr;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);

  // Create a new Ruby array with pre-allocated capacity
  VALUE arr = rb_ary_new_capa(ptr->dim);

  // Copy each float value to the Ruby array
  // Using rb_ary_store for better performance than rb_ary_push
  for (uint16_t i = 0; i < ptr->dim; ++i) {
    rb_ary_store(arr, i, DBL2NUM(ptr->values[i]));
  }

  return arr;
}

// Instance method: embedding.cosine_similarity(other_embedding)
// Calculate cosine similarity between two embeddings using optimized algorithm
static VALUE embedding_cosine_similarity(VALUE self, VALUE other) {
  embedding_t *a, *b;
  // Get C structs for both embeddings
  TypedData_Get_Struct(self, embedding_t, &embedding_type, a);
  TypedData_Get_Struct(other, embedding_t, &embedding_type, b);

  // Ensure dimensions match
  if (a->dim != b->dim) {
    rb_raise(rb_eArgError, "Dimension mismatch: %d vs %d", a->dim, b->dim);
  }

  // Use double precision for intermediate calculations to reduce accumulation errors
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;

  // Calculate dot product and vector magnitudes in a single loop
  // This is more cache-friendly than separate loops
  const float *va = a->values;
  const float *vb = b->values;

  for (uint16_t i = 0; i < a->dim; ++i) {
    float ai = va[i];
    float bi = vb[i];

    dot += (double)ai * bi;          // Dot product
    norm_a += (double)ai * ai;       // Square of magnitude for vector a
    norm_b += (double)bi * bi;       // Square of magnitude for vector b
  }

  // Check for zero vectors to avoid division by zero
  if (norm_a == 0.0 || norm_b == 0.0) {
    return DBL2NUM(0.0);  // Return 0 similarity for zero vectors
  }

  // Apply cosine similarity formula: dot(a,b)/(|a|*|b|)
  // Using sqrt for better numerical stability
  double magnitude_product = sqrt(norm_a * norm_b);
  double similarity = dot / magnitude_product;

  // Clamp result to [-1, 1] to handle floating point precision errors
  if (similarity > 1.0) similarity = 1.0;
  if (similarity < -1.0) similarity = -1.0;

  return DBL2NUM(similarity);
}

// Instance method: embedding.magnitude
// Calculate the magnitude (L2 norm) of the embedding vector
static VALUE embedding_magnitude(VALUE self) {
  embedding_t *ptr;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);

  double sum_squares = 0.0;
  const float *values = ptr->values;

  for (uint16_t i = 0; i < ptr->dim; ++i) {
    float val = values[i];
    sum_squares += (double)val * val;
  }

  return DBL2NUM(sqrt(sum_squares));
}

// Instance method: embedding.normalize!
// Normalize the embedding vector in-place (destructive operation)
static VALUE embedding_normalize_bang(VALUE self) {
  embedding_t *ptr;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);

  // Calculate magnitude
  double sum_squares = 0.0;
  float *values = ptr->values;

  for (uint16_t i = 0; i < ptr->dim; ++i) {
    float val = values[i];
    sum_squares += (double)val * val;
  }

  double magnitude = sqrt(sum_squares);

  // Avoid division by zero
  if (magnitude == 0.0) {
    rb_raise(rb_eZeroDivError, "Cannot normalize zero vector");
  }

  // Normalize each component
  float inv_magnitude = (float)(1.0 / magnitude);
  for (uint16_t i = 0; i < ptr->dim; ++i) {
    values[i] *= inv_magnitude;
  }

  return self;  // Return self for method chaining
}

// Ruby extension initialization function
// This function is called when the extension is loaded
void Init_embedding(void) {
  // Define module and class
  VALUE mRag = rb_define_module("RagEmbeddings");
  VALUE cEmbedding = rb_define_class_under(mRag, "Embedding", rb_cObject);

  // IMPORTANT: Undefine the default allocator to prevent the warning
  // This is necessary when using TypedData_Wrap_Struct
  rb_undef_alloc_func(cEmbedding);

  // Register class methods
  rb_define_singleton_method(cEmbedding, "from_array", embedding_from_array, 1);

  // Register instance methods
  rb_define_method(cEmbedding, "dim", embedding_dim, 0);
  rb_define_method(cEmbedding, "to_a", embedding_to_a, 0);
  rb_define_method(cEmbedding, "cosine_similarity", embedding_cosine_similarity, 1);
  rb_define_method(cEmbedding, "magnitude", embedding_magnitude, 0);
  rb_define_method(cEmbedding, "normalize!", embedding_normalize_bang, 0);
}