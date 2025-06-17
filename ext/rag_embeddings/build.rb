#!/usr/bin/env ruby
# frozen_string_literal: true

# Build script for rag_embeddings gem
# Supports both Rust and C implementations

require 'fileutils'
require 'rbconfig'

class EmbeddingBuilder
  def initialize
    @platform = RUBY_PLATFORM
    @dlext = RbConfig::CONFIG['DLEXT']
  end

  def rust_available?
    system('cargo --version > /dev/null 2>&1')
  end

  def build_rust
    puts "🦀 Building Rust extension..."

    # Clean previous builds
    system('cargo clean') if File.exist?('Cargo.lock')

    # Build in release mode for performance
    success = system('cargo build --release')

    unless success
      puts "❌ Rust build failed"
      return false
    end

    # Copy the built library to the Ruby extension location
    rust_lib_path = find_rust_library
    return false unless rust_lib_path

    target_path = "embedding.#{@dlext}"
    FileUtils.cp(rust_lib_path, target_path)

    puts "✅ Rust extension built successfully: #{target_path}"
    true
  end

  def build_c
    puts "🔧 Building C extension..."

    # Use the traditional Ruby extension build process
    success = system('ruby extconf.rb && make clean && make')

    if success
      puts "✅ C extension built successfully"
    else
      puts "❌ C build failed"
    end

    success
  end

  def clean
    puts "🧹 Cleaning Rust artifacts..."
    system('cargo clean') if File.exist?('Cargo.toml')

    # Clean C artifacts
    puts "🧹 Cleaning C artifacts..."
    FileUtils.rm_f(['Makefile', 'mkmf.log'])
    FileUtils.rm_rf(Dir.glob('*.{o,so,dylib,dll,bundle,bundle.*}'))
    FileUtils.rm_rf(['target/'])

    puts "✅ Cleaned"
  end

  def build
    case ARGV[0]
    when 'rust'
      build_rust || exit(1)
    when 'c'
      build_c || exit(1)
    when 'clean'
      clean
    else
      # Auto-detect and build
      if rust_available?
        puts "🚀 Auto-detected Rust toolchain, building Rust extension..."
        unless build_rust
          puts "⚠️  Rust build failed, falling back to C..."
          build_c || exit(1)
        end
      else
        puts "📦 Rust not available, building C extension..."
        build_c || exit(1)
      end
    end
  end

  private

  def find_rust_library
    target_dir = File.join('target', 'release')

    library_name = case @platform
                   when /darwin/
                     'librag_embeddings.dylib'
                   when /linux/
                     'librag_embeddings.so'
                   when /mswin|mingw/
                     'rag_embeddings.dll'
                   else
                     puts "❌ Unsupported platform: #{@platform}"
                     return nil
                   end

    library_path = File.join(target_dir, library_name)

    unless File.exist?(library_path)
      puts "❌ Rust library not found at: #{library_path}"
      return nil
    end

    library_path
  end
end

# Usage information
if ARGV.include?('--help') || ARGV.include?('-h')
  puts <<~HELP
    Usage: ruby build.rb [COMMAND]
    
    Commands:
      rust    - Build Rust extension only
      c       - Build C extension only  
      clean   - Clean all build artifacts
      (none)  - Auto-detect and build (Rust preferred)
    
    Examples:
      ruby build.rb          # Auto-build (prefers Rust)
      ruby build.rb rust     # Force Rust build
      ruby build.rb c        # Force C build
      ruby build.rb clean    # Clean artifacts
  HELP
  exit 0
end

# Run the builder
EmbeddingBuilder.new.build