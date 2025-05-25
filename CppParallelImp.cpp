#include <Windows.h>
#include <memory>
#include <thread>
#include <future>
#include <chrono>
#include <print>
#include <numeric>




/**
* @namespace Parallels
* @brief Quick and dirty POC for C#-style parallel processing using Windows Thread Pool API
*
* @warning This is a proof-of-concept implementation specifically for Windows.
* For multiplatform support, a custom thread pool implementation would be required.
* This namespace provides functionality similar to C#'s Parallel static class.
*
* @note Parallelism is intentionally limited to logical processors / 2 to prevent CPU lock-up.
*
* @author IceCoaled
* @date 2025-05-24
*/
namespace Parallels
{
	/**
	* @class ThreadPoolEnvironment
	* @brief Manages Windows Thread Pool API for parallel task execution
	*
	* This class encapsulates the Windows Thread Pool API and provides a simplified
	* interface for submitting work items. It automatically manages thread creation,
	* cleanup, and synchronization.
	*/
	class ThreadPoolEnvironment
	{
	private:
		// Handle to the Windows thread pool
		PTP_POOL mPoolHandle;
		// Thread pool callback environment
		PTP_CALLBACK_ENVIRON mCallbackEnviro;
		// Cleanup group for thread pool work items
		PTP_CLEANUP_GROUP mCleanUp;

		// Number of logical processors available
		uint32_t mProcessorCount;
		// Thread-safe ready state indicator
		std::atomic_bool mReady = false;

		/**
		* @interface IWorkItem
		* @brief Abstract base class(Interface) for work items that can be executed by the thread pool
		*/
		class IWorkItem
		{
		public:
			/**
			* @brief Virtual destructor for proper cleanup of derived classes
			*/
			virtual ~IWorkItem() = default;
			/**
			* @brief Pure virtual function that executes the work item's task
			*/
			virtual auto Execute() -> void = 0;
		};


		/**
		* @class WorkItem
		* @brief Template implementation of IWorkItem that wraps callable objects
		*
		*
		* @details This class captures callable objects and their arguments, providing exception
		* safety through std::promise/std::future mechanism. This is where we can add a template
		* parameter for return type and add in return functionality
		*/
		class WorkItem: public IWorkItem
		{
		private:
			// Promise for exception handling and completion tracking
			std::promise<void> mPromise;
			// Wrapped callable with bound arguments
			std::function<void()> mFunction;

		public:
			/**
			* @brief Constructs a WorkItem from a callable and its arguments
			*
			* @tparam Callable Type of the callable object (function, lambda, etc.)
			* @tparam Args Variadic template for argument types
			* @param Pred The callable object to execute
			* @param args Arguments to bind to the callable
			*
			* Uses perfect forwarding and std::apply to bind arguments at construction time.
			*/
			template <typename Callable, typename... Args>
			WorkItem( Callable&& Pred, Args&&... args )
			{
				// Recreate the lambda and bind our arguments to it
				mFunction = [callable = std::forward<Callable>( Pred ),
					argsTuple = std::make_tuple( std::forward<Args>( args )... )]()
					{
						// This is like invoke but it unpacks the arguments
						std::apply( callable, argsTuple );
					};

			}

			/**
			* @brief Executes the work item's function with exception handling
			*
			* This method is called by the thread pool worker threads.
			* It executes the stored function and captures any exceptions
			* for later retrieval via the future.
			*/
			auto Execute() -> void override
			{
				try
				{
					// Call function
					mFunction();
					// Set value void for this
					mPromise.set_value();
				} catch ( ... )
				{
					// Else capture any exceptions thrown
					mPromise.set_exception( std::current_exception() );
				}
			}

			/**
			* @brief Returns a future for tracking completion and exceptions
			*
			* @return std::future<void> Future that becomes ready when work completes
			*
			* @note This is still used here as it can be used to get
			* the state of the result and captured exception. Also we
			* could easily make the return type a template parameter and
			* allow returns.
			*/
			[[nodiscard]] auto GetFuture() -> std::future<void>
			{
				return mPromise.get_future();
			}
		};


		/**
		* @brief Static callback function called by Windows Thread Pool
		*
		* @param instance Thread pool callback instance (unused)
		* @param parameter Pointer to the work item to execute
		* @param work Thread pool work object (unused)
		*
		* This is the entry point for all thread pool work. It casts the parameter
		* to a work item and executes it, then cleans up the memory.
		*
		* @warning This function must not throw exceptions as it could terminate the thread pool
		*/
		static VOID CALLBACK WorkCallback( PTP_CALLBACK_INSTANCE instance, PVOID parameter, PTP_WORK work )
		{
			UNREFERENCED_PARAMETER( instance );
			UNREFERENCED_PARAMETER( work );
			// Capture our work item class
			auto* workItem = static_cast< IWorkItem* >( parameter );
			try
			{
				// Each thread gets different Parameter
				workItem->Execute();
			} catch ( ... )
			{
				// Note: Rethrowing here could terminate the thread pool
				// Consider logging instead in production code
				throw std::current_exception();
			}

			// Each thread deletes their own object
			delete workItem;
		}


	private:

		/**
		* @brief Initializes the Windows Thread Pool and associated structures
		*
		* Sets up the thread pool with processor count / 2 threads, creates
		* the callback environment, and initializes the cleanup group.
		* Updates mReady state based on success/failure.
		*/
		auto SetupThreadPool() -> void
		{
			// Create the pool
			this->mPoolHandle = CreateThreadpool( nullptr );
			if ( this->mPoolHandle == nullptr )
			{
				return;//Error handling
			}
			// We statically Set min max thread group size to processor count / 2
			SetThreadpoolThreadMinimum( this->mPoolHandle, this->mProcessorCount >> 1 );
			SetThreadpoolThreadMaximum( this->mPoolHandle, this->mProcessorCount >> 1 );

			// Create our environment struct
			this->mCallbackEnviro = ( ( PTP_CALLBACK_ENVIRON )malloc( sizeof( TP_CALLBACK_ENVIRON ) ) );
			if ( this->mCallbackEnviro == nullptr )
			{
				return;//Error handling
			}

			// Initialize the environment and check its V3
			InitializeThreadpoolEnvironment( this->mCallbackEnviro );
			if ( this->mCallbackEnviro->Version != 0x03 )
			{
				return;//Error handling
			} else
			{
				// Set our pool priority to high
				this->mCallbackEnviro->CallbackPriority = TP_CALLBACK_PRIORITY_HIGH;
			}
			// Now bind the pool and environment
			SetThreadpoolCallbackPool( this->mCallbackEnviro, this->mPoolHandle );

			// Create and bind the cleanup group
			this->mCleanUp = CreateThreadpoolCleanupGroup();
			if ( this->mCleanUp == nullptr )
			{
				return;//Error handling
			}

			SetThreadpoolCallbackCleanupGroup( this->mCallbackEnviro, this->mCleanUp, nullptr );

			return;
		}

	public:
		/**
		* @brief Constructs and initializes the thread pool environment
		*
		* Automatically detects processor count and sets up the thread pool
		* with half the available logical processors.
		*/
		ThreadPoolEnvironment()
		{
			// Get processor count
			this->mProcessorCount = GetProcessorCount();
			// Main init function
			SetupThreadPool();
			// TODO: Add error checking for SetupThreadPool return value}
			this->mReady = true;
		}

		/**
		* @brief Destroys the thread pool and cleans up all resources
		*
		* Waits for all pending work to complete before cleanup.
		*/
		~ThreadPoolEnvironment()
		{
			CloseThreadpoolCleanupGroupMembers( this->mCleanUp, false, nullptr );
			CloseThreadpoolCleanupGroup( this->mCleanUp );
			CloseThreadpool( this->mPoolHandle );
			free( this->mCallbackEnviro );
			this->mReady = false;
		}

		/**
		* @brief Submits a callable object to the thread pool for execution
		*
		* @tparam Callable Type of the callable object
		* @param workCallback The function/lambda to execute asynchronously
		*
		* This method waits up to 20 milliseconds for the thread pool to be ready,
		* then submits the work. The work item is automatically cleaned up after execution.
		*/
		template<typename Callable>
		auto PostWork( Callable&& Pred ) -> void
		{
			using Timer = typename std::conditional<std::chrono::high_resolution_clock::is_steady,
				std::chrono::high_resolution_clock, std::chrono::steady_clock>;
			using Microseconds = std::chrono::duration<double, std::micro>;

			// We sleep the calling thread till our pool is ready
			// This times our after 20 milliseconds
			auto start = Timer::type::now();
			while ( !this->mReady && ( Timer::type::now() - start ).count() < 20LLU )
			{
				std::this_thread::sleep_for( Microseconds( 1 ) );
			}
			if ( !this->mReady )
			{
				// Error handling - consider throwing or logging
				return;
			} else
			{
				// Create our work item from the provided lambda and check if its null
				// Note: the work items memory ends in the callback as we delete it there before returning

				if ( auto* workItem = new WorkItem( std::forward<Callable>( Pred ) ); workItem == nullptr )
				{
					// Error handling - consider throwing or logging
				} else
				{
					// Create our work object and check null
					if ( auto work = CreateThreadpoolWork( WorkCallback, workItem, mCallbackEnviro ); work != nullptr )
					{
						// Post the work for the pool
						SubmitThreadpoolWork( work );
						// Release the work object, this is like
						// Closing a handle
						CloseThreadpoolWork( work );
					} else
					{
						delete workItem;
						//TODO: consider throwing or logging
					}

				}
			}
		}

		/**
		* @brief Waits for all submitted work items to complete
		*
		* This method blocks until all work items in the current cleanup group
		* have finished executing. It then recreates the cleanup group for future work.
		*/
		auto WaitAll() -> void
		{

			if ( this->mCleanUp )
			{
				this->mReady = false;
				CloseThreadpoolCleanupGroupMembers( this->mCleanUp, false, nullptr );

				this->mCleanUp = CreateThreadpoolCleanupGroup();
				if ( this->mCleanUp == nullptr )
				{
					// Error handling - consider throwing or logging
					return;
				}

				SetThreadpoolCallbackCleanupGroup( this->mCallbackEnviro, this->mCleanUp, nullptr );
				this->mReady = true;
			}
		}

		/**
		* @brief Gets the number of logical processors available
		*
		* @return const uint32_t Number of hardware threads available
		*/
		[[nodiscard]] static __inline auto GetProcessorCount() -> const uint32_t
		{
			return std::thread::hardware_concurrency();
		}

		/**
		* @brief Checks if the thread pool is ready to accept work
		*
		* @return const bool True if ready, false otherwise
		*/
		[[nodiscard]] auto IsParallelsReady() const-> const bool
		{
			return this->mReady.load();
		}
	};


	/**
	* @concept IsRangedContainer
	* @brief Concept to validate that a type supports range-based operations
	*
	* @tparam Container The container type to validate
	*
	* @details Ensures the container has begin(), end(), cbegin(), cend(), and empty() methods
	* with appropriate return types for iteration.
	*/
	template< typename Container>
	concept IsRangedContainer = requires( Container c )
	{
		{ c.begin() } -> std::same_as< typename Container::iterator>;
		{ c.end() }	-> std::same_as< typename Container::iterator>;
		{ c.cbegin() } -> std::same_as< typename Container::const_iterator>;
		{ c.cend() } -> std::same_as< typename Container::const_iterator>;
		{ c.empty() } -> std::convertible_to<bool>;
	};


	/**
	* @concept IsProperRange
	* @brief Concept to validate types suitable for range-based parallel operations
	*
	* @tparam Indexer The indexer type to validate
	*
	* @details Accepts either input iterators or integral types for indexing operations.
	*/
	template< typename Indexer>
	concept IsProperRange =
		std::input_iterator<Indexer> ||
		std::is_integral_v<Indexer>;


	/**
	* @brief Parallel for loop implementation using Windows Thread Pool
	*
	* @tparam indexer Type used for loop indexing (must satisfy IsProperRange)
	* @tparam Pred Type of the predicate function
	* @param start Starting index for the loop
	* @param end Ending index for the loop (exclusive)
	* @param func Function to execute for each index
	* @param szChunk Optional chunk size for work distribution (0 = auto-calculate)
	*
	* @details Distributes loop iterations across available threads. Each thread processes
	* a chunk of iterations to minimize overhead while maximizing parallelism.
	*
	* @note Currently creates one work item per iteration, which may be inefficient
	* for loops with small per-iteration work. Consider using larger chunk sizes.
	*/
	template<IsProperRange indexer, typename Pred>
	constexpr auto For( indexer start, indexer end, Pred&& func, size_t szChunk = 0 ) -> void
	{
		using TPE = ThreadPoolEnvironment;
		// Get processor count for initialization and math
		// Calculate the total range and the chunk size for each thread
		const auto totalRange = static_cast< size_t >( end - start );
		if ( szChunk == 0 )
		{
			const auto pCount = TPE::GetProcessorCount();
			szChunk = std::max< size_t >( 1ULL, totalRange / pCount );
		}

		// Create our environment
		auto pool = TPE();



		// Post the work with the lambda function
		// This is our main callable
		for ( indexer i = start; i < end; i += static_cast< indexer >( szChunk ) )
		{
			indexer chunkEnd = std::min< indexer >( i + static_cast< indexer >( szChunk ), end );
			pool.PostWork( [i, chunkEnd, &func]()
						   {
							   for ( indexer j = i; j < chunkEnd; ++j )
							   {
								   func( j );
							   }
						   } );
		}

		// Wait for the pool to
		// finish and Clean up from work
		pool.WaitAll();
	}

	/**
	* @brief Parallel for-each implementation for containers
	*
	* @tparam Container Type of the container (must satisfy IsRangedContainer)
	* @tparam Pred Type of the predicate function
	* @param c The container to iterate over
	* @param func Function to execute for each container element
	* @param szChunk Optional chunk size for work distribution (0 = auto-calculate)
	*
	* @details Distributes container elements across available threads for parallel processing.
	* Uses iterator advancement to ensure proper traversal of all container types.
	*/
	template<IsRangedContainer Container, typename Pred>
	constexpr auto ForEach( const Container& c, Pred&& func, size_t szChunk = 0 ) -> void
	{
		if ( c.empty() )
		{
			return;
		}

		using TPE = ThreadPoolEnvironment;

		// Calculate the total range and the chunk size for each thread
		const auto totalRange = static_cast< size_t >( std::distance( c.begin(), c.end() ) );
		if ( szChunk == 0 )
		{
			const auto pCount = TPE::GetProcessorCount();
			szChunk = std::max< size_t >( 1ULL, totalRange / pCount );
		}

		// Create our environment
		auto pool = TPE();

		// Get our beginning and end of our container
		auto start = c.begin();
		auto end = c.end();

		// Loop through container and post the work for each one
		// The lambda here is our main callable
		for ( auto it = start; it < end; )
		{
			auto chunkEnd = it;
			// We use advance as this solidifies we 
			// Properly advance through ranges 
			std::advance( chunkEnd, std::min< size_t >( szChunk, static_cast< size_t >( std::distance( it, end ) ) ) );

			pool.PostWork( [it, chunkEnd, &func]()
						   {
							   for ( auto current = it; current != chunkEnd; ++current )
							   {
								   func( *current );
							   }
						   } );
					   // Advance the main iterator to the end of the current chunk
			it = chunkEnd;
		}
		// Wait for the pool to
		// finish and Clean up from work
		pool.WaitAll();
	}

};//!Parallels








int main()
{
	using Timer = typename std::conditional<std::chrono::high_resolution_clock::is_steady,
		std::chrono::high_resolution_clock, std::chrono::steady_clock>;
	using Microseconds = std::chrono::duration<double, std::micro>;


	// Example 1: Simple parallel for loop
	std::println( "Showing all the threads involved:" );

	std::mutex printMutex;
	Parallels::For( 0, 10, [&]( int i )
					{
						std::lock_guard<std::mutex> lock( printMutex );
						std::println( "Processing index: {}, on thread: {}", i, GetCurrentThreadId() );
					} );

										// Example 2: Parallel computation


	constexpr size_t arraySize = 1000000;
	std::vector<int> numbers( arraySize );
	std::vector<int> results( arraySize );

	std::println( "\nParallel Computation Example Using {} digits:", arraySize );

	// Initialize input
	std::iota( numbers.begin(), numbers.end(), 0 );

	auto start = Timer::type::now();

	for ( int z = 0; z < arraySize; ++z )
	{
		results[ z ] = numbers[ z ] * numbers[ z ]; // Square each number
	}

	auto end = Timer::type::now();
	auto duration = std::chrono::duration_cast< std::chrono::milliseconds >( end - start );

	std::println( "Non Parallel computation took: {}ms ", duration.count() );

	std::iota( numbers.begin(), numbers.end(), 0 );

	start = Timer::type::now();

	// Parallel computation
	Parallels::For( size_t( 0 ), arraySize, [&]( size_t i )
					{
						results[ i ] = numbers[ i ] * numbers[ i ]; // Square each number
					} );

	end = Timer::type::now();
	duration = std::chrono::duration_cast< std::chrono::milliseconds >( end - start );

	std::println( "Parallel computation took: {}ms ", duration.count() );

	// Example 3: Parallel for each with container
	std::println( "\nParallel ForEach Example:" );

	std::vector<std::string> words = { "hello", "world", "parallel", "processing", "rocks" };
	std::vector<size_t> lengths( words.size() );

	start = Timer::type::now();

	Parallels::ForEach( words, [&]( const std::string& word )
						{
							// Find the index of this word (for demonstration)
							auto it = std::find( words.begin(), words.end(), word );
							if ( it != words.end() )
							{
								size_t index = std::distance( words.begin(), it );
								lengths[ index ] = word.length();
							}
						} );

	end = Timer::type::now();
	duration = std::chrono::duration_cast< std::chrono::milliseconds >( end - start );

	std::println( "Parallel For each computation took: {}ms ", duration.count() );

	for ( size_t i = 0; i < words.size(); ++i )
	{
		std::println( "'{}' has: {} characters", words[ i ], lengths[ i ] );
	}

	return 0;
}