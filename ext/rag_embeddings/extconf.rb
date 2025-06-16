# frozen_string_literal: true

require 'mkmf'

# Check if Rust toolchain is available
def rust_available?
  system('cargo --version > /dev/null 2>&1')
end

# Build Rust extension
def build_rust_extension
  puts "Building Rust extension..."

  # Set target directory
  target_dir = File.join(__dir__, 'target', 'release')

  # Build the Rust library
  unless system('cargo build --release')
    puts "Failed to build Rust extension"
    return false
  end

  # Find the generated library
  rust_lib = if RUBY_PLATFORM =~ /darwin/
               File.join(target_dir, 'librag_embeddings.dylib')
             elsif RUBY_PLATFORM =~ /linux/
               File.join(target_dir, 'librag_embeddings.so')
             elsif RUBY_PLATFORM =~ /mswin|mingw/
               File.join(target_dir, 'rag_embeddings.dll')
             else
               puts "Unsupported platform: #{RUBY_PLATFORM}"
               return false
             end

  unless File.exist?(rust_lib)
    puts "Rust library not found at: #{rust_lib}"
    return false
  end

  # Copy the library to the expected location
  extension_name = "embedding.#{RbConfig::CONFIG['DLEXT']}"
  FileUtils.cp(rust_lib, extension_name)

  puts "Rust extension built successfully"
  true
end

# Fallback to C extension
def build_c_extension
  puts "Building C extension as fallback..."

  # Original C extension configuration
  extension_name = 'embedding'
  dir_config(extension_name)

  # Check for required headers and functions
  have_header('ruby.h') or abort('ruby.h not found')
  have_header('stdint.h') or abort('stdint.h not found')
  have_header('stdlib.h') or abort('stdlib.h not found')
  have_header('math.h') or abort('math.h not found')

  # Check for math library
  have_library('m', 'sqrt') or abort('math library not found')

  create_makefile(extension_name)
end

# Main build logic
if rust_available?
  puts "Rust toolchain detected, building Rust extension..."
  unless build_rust_extension
    puts "Rust build failed, falling back to C extension..."
    build_c_extension
  end
else
  puts "Rust toolchain not found, building C extension..."
  build_c_extension
end