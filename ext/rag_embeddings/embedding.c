#include <ruby.h>     // Ruby API
#include <stdint.h>   // For integer types like uint16_t
#include <stdlib.h>   // For memory allocation functions
#include <math.h>     // For math functions like sqrt
#include "embedding_config.h" // Import the configuration

// Main data structure for storing embeddings with fixed size
typedef struct {
  uint16_t dim;                           // Actual dimension used (can be <= EMBEDDING_DIMENSION)
  float values[EMBEDDING_DIMENSION];      // Fixed-size array for embedding values
} embedding_t;

// Callback for freeing memory when Ruby's GC collects our object
static void embedding_free(void *ptr) {
  // With RUBY_TYPED_EMBEDDABLE and TypedData_Make_Struct,
  // Ruby handles the deallocation automatically
  // No need to explicitly free the memory
}

// Callback to report memory usage to Ruby's GC
static size_t embedding_memsize(const void *ptr) {
  // With embedded objects, we report the full struct size
  return sizeof(embedding_t);
}

// Type information for Ruby's GC with embedding support
static const rb_data_type_t embedding_type = {
  "RagEmbeddings/Embedding",                    // Type name
  {0, embedding_free, embedding_memsize,},      // Functions: mark, free, size
  0, 0,                                         // Parent type, data
  RUBY_TYPED_FREE_IMMEDIATELY | RUBY_TYPED_EMBEDDABLE  // Flags with embedding!
};

// Class method: RagEmbeddings::Embedding.from_array([1.0, 2.0, ...])
// Creates a new embedding from a Ruby array - NOW FASTER!
static VALUE embedding_from_array(VALUE klass, VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);           // Ensure argument is a Ruby array
  long array_len = RARRAY_LEN(rb_array);

  // Check if the array size exceeds our maximum dimension
  if (array_len > EMBEDDING_DIMENSION) {
    rb_raise(rb_eArgError, "Embedding dimension %ld exceeds maximum %d",
             array_len, EMBEDDING_DIMENSION);
  }

  uint16_t dim = (uint16_t)array_len;

  // With RUBY_TYPED_EMBEDDABLE, use TypedData_Make_Struct
  // This automatically allocates Ruby object + embedded data space
  embedding_t *ptr;
  VALUE obj = TypedData_Make_Struct(klass, embedding_t, &embedding_type, ptr);

  ptr->dim = dim;

  // Copy values from Ruby array to our C array
  for (int i = 0; i < dim; ++i)
    ptr->values[i] = (float)NUM2DBL(rb_ary_entry(rb_array, i));

  // Zero out unused slots for consistency
  for (int i = dim; i < EMBEDDING_DIMENSION; ++i)
    ptr->values[i] = 0.0f;

  return obj;
}

// Class method to get the maximum supported dimension
static VALUE embedding_max_dimension(VALUE klass) {
  return INT2NUM(EMBEDDING_DIMENSION);
}

// Instance method: embedding.dim
// Returns the actual dimension of the embedding
static VALUE embedding_dim(VALUE self) {
  embedding_t *ptr;
  // Get the C struct from the Ruby object - NOW FASTER!
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);
  return INT2NUM(ptr->dim);
}

// Instance method: embedding.to_a
// Converts the embedding back to a Ruby array (only actual dimensions)
static VALUE embedding_to_a(VALUE self) {
  embedding_t *ptr;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);

  // Create a new Ruby array with pre-allocated capacity for actual dimension
  VALUE arr = rb_ary_new2(ptr->dim);

  // Copy only the used float values to the Ruby array - FASTER MEMORY ACCESS!
  for (int i = 0; i < ptr->dim; ++i)
    rb_ary_push(arr, DBL2NUM(ptr->values[i]));

  return arr;
}

// Instance method: embedding.cosine_similarity(other_embedding)
// Calculate cosine similarity - MUCH FASTER with embedded data!
static VALUE embedding_cosine_similarity(VALUE self, VALUE other) {
  embedding_t *a, *b;
  // Get C structs for both embeddings - direct access, no pointer deref!
  TypedData_Get_Struct(self, embedding_t, &embedding_type, a);
  TypedData_Get_Struct(other, embedding_t, &embedding_type, b);

  // Ensure dimensions match
  if (a->dim != b->dim)
    rb_raise(rb_eArgError, "Dimension mismatch: %d vs %d", a->dim, b->dim);

  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

  // Calculate dot product and vector magnitudes
  // Better cache locality = faster calculations!
  for (int i = 0; i < a->dim; ++i) {
    float val_a = a->values[i];
    float val_b = b->values[i];
    dot += val_a * val_b;              // Dot product
    norm_a += val_a * val_a;           // Square of magnitude for vector a
    norm_b += val_b * val_b;           // Square of magnitude for vector b
  }

  // Apply cosine similarity formula: dot(a,b)/(|a|*|b|)
  // Small epsilon (1e-8) added to prevent division by zero
  float magnitude_product = sqrt(norm_a) * sqrt(norm_b);
  return DBL2NUM(dot / (magnitude_product + 1e-8f));
}

// Ruby extension initialization function
// This function is called when the extension is loaded
void Init_embedding(void) {
  // Define module and class
  VALUE mRag = rb_define_module("RagEmbeddings");
  VALUE cEmbedding = rb_define_class_under(mRag, "Embedding", rb_cObject);

  // Register class methods
  rb_define_singleton_method(cEmbedding, "from_array", embedding_from_array, 1);
  rb_define_singleton_method(cEmbedding, "max_dimension", embedding_max_dimension, 0);

  // Register instance methods
  rb_define_method(cEmbedding, "dim", embedding_dim, 0);
  rb_define_method(cEmbedding, "to_a", embedding_to_a, 0);
  rb_define_method(cEmbedding, "cosine_similarity", embedding_cosine_similarity, 1);
}