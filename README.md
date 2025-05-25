# C++ Parallels POC

A proof-of-concept implementation that brings C#-style parallel processing to C++ using the Windows Thread Pool API.

## 🎯 Project Overview

This POC demonstrates how to implement parallel processing functionality similar to C#'s `Parallel` static class in C++. The implementation leverages the Windows Thread Pool API to provide efficient task distribution and thread management, making parallel programming more accessible and performant in C++ applications.

## ✨ Key Features

- **C#-Style API**: Familiar `Parallel.For` and `Parallel.ForEach` syntax adapted for C++
- **Windows Thread Pool Integration**: Utilizes native Windows Thread Pool API for optimal performance
- **Template-Based Design**: Supports any callable objects (functions, lambdas, functors)
- **Concept-Driven Safety**: Uses C++20 concepts to ensure type safety at compile time
- **Exception Handling**: Built-in exception propagation from worker threads

## 🏗️ Architecture

### Design Principles

- **Performance-First**: Limited to `logical_processors / 2` to prevent CPU lock-up
- **Type Safety**: C++20 concepts ensure compile-time validation
- **Exception Safety**: Promise/future mechanism for proper exception propagation
- **ThreadPoolEnvironment**: Manages the Windows Thread Pool lifecycle and work submission
- **WorkItem**: Template-based work wrapper with exception handling
- **Parallel Functions**: High-level APIs (`For` and `ForEach`) for parallel execution

## 🚀 Usage Examples

### Parallel For Loop
```cpp
#include "CppParallelImp.cpp"

// Process indices 0-9 in parallel
Parallels::For(0, 10, [](int i) {
    std::println("Processing index: {}", i);
});
```

### Parallel Computation
```cpp
std::vector<int> numbers(1000000);
std::vector<int> results(1000000);

// Initialize data
std::iota(numbers.begin(), numbers.end(), 0);

// Parallel computation
Parallels::For(size_t(0), numbers.size(), [&](size_t i) {
    results[i] = numbers[i] * numbers[i]; // Square each number
});

// Parallel Ranged for computation
Parallels::For(numbers.begin(), numbers.end(), [&](std::vector<int>::iterator& it) {
auto itVar = *it;
size_t index = std::distance(numbers.begin(), it);
results[index] =itVar * itVar; // Square each number
});
```

### Parallel ForEach with Containers
```cpp
std::vector<std::string> words = {"hello", "world", "parallel", "processing"};
std::vector<size_t> lengths(words.size());

Parallels::ForEach(words, [&](const std::string& word) {
    // Process each word in parallel
    auto it = std::find(words.begin(), words.end(), word);
    if (it != words.end()) {
        size_t index = std::distance(words.begin(), it);
        lengths[index] = word.length();
    }
});
```

## 📋 Requirements

- **Operating System**: Windows (uses Windows Thread Pool API)
- **Compiler**: C++20 compatible compiler (MSVC recommended)
- **Dependencies**: 
  - `<Windows.h>` - Windows Thread Pool API
  - `<thread>`, `<future>`, `<chrono>` - Standard C++ threading
  - `<print>` - C++23 print library (or substitute with iostream)

## ⚡ Performance Characteristics

- **Thread Count**: Automatically configured to `logical_processors / 2`
- **Work Distribution**: Configurable chunk sizes for optimal load balancing
- **Overhead**: Minimal per-task overhead through efficient work item pooling
- **Scalability**: Scales effectively with available hardware threads

## 🎛️ Configuration Options

### Custom Chunk Sizes
```cpp
// Auto-calculated chunk size (default)
Parallels::For(0, 1000, func);

// Custom chunk size for fine-tuning
Parallels::For(0, 1000, func, 50);
```

### Thread Pool Status
```cpp
ThreadPoolEnvironment pool;
if (pool.IsParallelsReady()) {
    // Pool is ready for work submission
}
```

## ⚠️ Limitations & Considerations

- **Platform Specific**: Currently Windows-only implementation
- **Exception Handling**: Exceptions in worker threads are captured but may impact performance
- **Memory Management**: Work items are dynamically allocated (consider object pooling for high-frequency scenarios)
- **Thread Safety**: User code must handle shared resource synchronization


## 📝 Notes

This is a **proof-of-concept** implementation designed to explore C++ parallel programming patterns. For production use, consider additional features like:
- Comprehensive error handling
- Logging and diagnostics
- Performance monitoring
- Cross-platform compatibility
- Integration with existing C++ parallel libraries

## 👨‍💻 Author

**IceCoaled** - *Initial POC Development* - 2025-05-24

---

*This POC demonstrates the feasibility of bringing high-level parallel programming constructs to C++ while maintaining performance and type safety.*
