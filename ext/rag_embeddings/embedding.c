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
  xfree(ptr);         // Ruby's memory free function
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
  RUBY_TYPED_FREE_IMMEDIATELY              // Flags
};

// Class method: RagEmbeddings::Embedding.from_array([1.0, 2.0, ...])
// Creates a new embedding from a Ruby array
static VALUE embedding_from_array(VALUE klass, VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);           // Ensure argument is a Ruby array
  uint16_t dim = (uint16_t)RARRAY_LEN(rb_array);

  // Allocate memory for struct + array of floats
  embedding_t *ptr = xmalloc(sizeof(embedding_t) + dim * sizeof(float));
  ptr->dim = dim;

  // Copy values from Ruby array to our C array
  for (int i = 0; i < dim; ++i)
    ptr->values[i] = (float)NUM2DBL(rb_ary_entry(rb_array, i));

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
  VALUE arr = rb_ary_new2(ptr->dim);

  // Copy each float value to the Ruby array
  for (int i = 0; i < ptr->dim; ++i)
    rb_ary_push(arr, DBL2NUM(ptr->values[i]));

  return arr;
}

// Instance method: embedding.cosine_similarity(other_embedding)
// Calculate cosine similarity between two embeddings
static VALUE embedding_cosine_similarity(VALUE self, VALUE other) {
  embedding_t *a, *b;
  // Get C structs for both embeddings
  TypedData_Get_Struct(self, embedding_t, &embedding_type, a);
  TypedData_Get_Struct(other, embedding_t, &embedding_type, b);

  // Ensure dimensions match
  if (a->dim != b->dim)
    rb_raise(rb_eArgError, "Dimension mismatch");

  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

  // Calculate dot product and vector magnitudes
  for (int i = 0; i < a->dim; ++i) {
    dot += a->values[i] * b->values[i];      // Dot product
    norm_a += a->values[i] * a->values[i];   // Square of magnitude for vector a
    norm_b += b->values[i] * b->values[i];   // Square of magnitude for vector b
  }

  // Apply cosine similarity formula: dot(a,b)/(|a|*|b|)
  // Small epsilon (1e-8) added to prevent division by zero
  return DBL2NUM(dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8));
}

// Ruby extension initialization function
// This function is called when the extension is loaded
void Init_embedding(void) {
  // Define module and class
  VALUE mRag = rb_define_module("RagEmbeddings");
  VALUE cEmbedding = rb_define_class_under(mRag, "Embedding", rb_cObject);

  // Register class methods
  rb_define_singleton_method(cEmbedding, "from_array", embedding_from_array, 1);

  // Register instance methods
  rb_define_method(cEmbedding, "dim", embedding_dim, 0);
  rb_define_method(cEmbedding, "to_a", embedding_to_a, 0);
  rb_define_method(cEmbedding, "cosine_similarity", embedding_cosine_similarity, 1);
}