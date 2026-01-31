import Google from "../assets/Google.png";
import Amazon from "../assets/Amazon.webp";
import Meta from "../assets/Meta.jpg";
import Adobe from "../assets/Adobe.png";
import Apple from "../assets/apple.webp";
import DE from "../assets/DE.png";
import Flipkart from "../assets/Flipkart.webp";
import IBM from "../assets/IBM.jpg";
import JP from "../assets/JP.webp";
import Juspay from "../assets/Juspay.avif";
import LinkedIn from "../assets/LinkedIn.jpg";
import Microsoft from "../assets/Microsoft.webp";
import Netflix from "../assets/Netflix.png";
import Nvidia from "../assets/Nvidia.webp";
import Oracle from "../assets/Oracle.webp";
import PayPal from "../assets/PayPal.png";
import Paytm from "../assets/Paytm.jpg";
import Uber from "../assets/Uber.webp";


export const cheatsheetData = {
  bigO: {
  title: "Big O Complexity Analysis",
  icon: "Activity",
  description: "Understanding algorithm efficiency through time and space complexity",
  hero: {
    title: "Master Big O Complexity",
    subtitle: "Learn how to analyze and optimize algorithm performance",
    image: "/mnt/user-data/uploads/Big_O.jpg"
  },
  timeComplexities: [
    {
      notation: "O(1)",
      name: "Constant Time",
      description: "Best possible time complexity. The algorithm takes the same amount of time regardless of input size. Operations complete in fixed time.",
      color: "#10b981",
      maxN: "> 10⁹",
      operations: "1 operation",
      performance: "Excellent",
      examples: [
        "Array access by index: arr[5]",
        "Hash table lookup: map.get(key)",
        "Stack push/pop operations",
        "Getting array length",
        "Mathematical calculations"
      ],
      code: `// O(1) - Constant Time Example
function getFirstElement(arr) {
    return arr[0];  // Single operation, always
}

function hashLookup(map, key) {
    return map.get(key);  // Direct access
}

// Even with multiple operations, still O(1)
function constantExample(x, y) {
    const a = x + y;
    const b = a * 2;
    const c = b - 10;
    return c;
}`
    },
    {
      notation: "O(log n)",
      name: "Logarithmic Time",
      description: "Excellent complexity. The algorithm divides the problem in half with each step. log(1,000,000) is only about 20 operations!",
      color: "#3b82f6",
      maxN: "> 10⁸",
      operations: "~20 ops for 1M",
      performance: "Great",
      examples: [
        "Binary search on sorted array",
        "Balanced Binary Search Tree operations",
        "Finding element in sorted data",
        "Divide and conquer algorithms",
        "Processing digits of a number"
      ],
      code: `// O(log n) - Binary Search
function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (arr[mid] === target) {
            return mid;
        }
        
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}`
    },
    {
      notation: "O(n)",
      name: "Linear Time",
      description: "Good complexity. The algorithm processes each element once. Time grows proportionally with input size. Most optimal for problems requiring inspection of all elements.",
      color: "#8b5cf6",
      maxN: "≤ 10⁶",
      operations: "1M ops",
      performance: "Good",
      examples: [
        "Single loop through array",
        "Finding max/min element",
        "Two pointers technique",
        "Linear search",
        "Counting occurrences"
      ],
      code: `// O(n) - Linear Time
function findMax(arr) {
    let max = arr[0];
    
    // Process each element once
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    
    return max;
}

// Still O(n) even with constants
function linearExample(arr) {
    // 3 * n operations, but still O(n)
    for (let i = 0; i < arr.length; i++) {
        // operation 1
    }
    for (let i = 0; i < arr.length; i++) {
        // operation 2
    }
    for (let i = 0; i < arr.length; i++) {
        // operation 3
    }
}`
    },
    {
      notation: "O(n log n)",
      name: "Linearithmic Time",
      description: "Fair complexity and optimal for comparison-based sorting. The best we can do for general sorting problems. Common in divide-and-conquer algorithms.",
      color: "#f59e0b",
      maxN: "≤ 10⁶",
      operations: "1M ops",
      performance: "Fair",
      examples: [
        "Merge Sort",
        "Quick Sort (average case)",
        "Heap Sort",
        "Efficient sorting algorithms",
        "Divide and conquer with merge"
      ],
      code: `// O(n log n) - Merge Sort
function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    
    return merge(left, right);
}

function merge(left, right) {
    const result = [];
    let i = 0, j = 0;
    
    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            result.push(left[i++]);
        } else {
            result.push(right[j++]);
        }
    }
    
    return result.concat(left.slice(i), right.slice(j));
}`
    },
    {
      notation: "O(n²)",
      name: "Quadratic Time",
      description: "Moderate complexity. Acceptable only for small inputs (n ≤ 3000). Common in brute force solutions with nested loops. Always try to optimize if possible.",
      color: "#ef4444",
      maxN: "≤ 3,000",
      operations: "9M ops",
      performance: "Moderate",
      examples: [
        "Nested loops (checking all pairs)",
        "Bubble Sort",
        "Selection Sort",
        "Insertion Sort",
        "Comparing every element with every other"
      ],
      code: `// O(n²) - Quadratic Time
function bubbleSort(arr) {
    const n = arr.length;
    
    // Nested loops = n * n = n²
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    
    return arr;
}

// Finding all pairs
function findAllPairs(arr) {
    const pairs = [];
    
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            pairs.push([arr[i], arr[j]]);
        }
    }
    
    return pairs;
}`
    },
    {
      notation: "O(2ⁿ)",
      name: "Exponential Time",
      description: "Poor complexity. Only feasible for very small inputs (n ≤ 20). Grows extremely rapidly. Often requires memoization to optimize. Common in recursive solutions without caching.",
      color: "#dc2626",
      maxN: "≤ 20",
      operations: "1M ops at n=20",
      performance: "Poor",
      examples: [
        "Recursive Fibonacci (naive)",
        "Generating all subsets",
        "Solving Tower of Hanoi",
        "Backtracking without pruning",
        "Recursive tree traversal (all paths)"
      ],
      code: `// O(2ⁿ) - Exponential Time
// WARNING: Very slow for n > 30!
function fibonacci(n) {
    if (n <= 1) return n;
    
    // Each call creates 2 more calls!
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Generating all subsets
function generateSubsets(arr) {
    const result = [];
    
    function backtrack(start, current) {
        result.push([...current]);
        
        for (let i = start; i < arr.length; i++) {
            current.push(arr[i]);
            backtrack(i + 1, current);
            current.pop();
        }
    }
    
    backtrack(0, []);
    return result;
}`
    },
    {
      notation: "O(n!)",
      name: "Factorial Time",
      description: "Very poor complexity. Only works for tiny inputs (n ≤ 12). Grows astronomically fast. Common in generating all permutations. Almost always needs optimization.",
      color: "#991b1b",
      maxN: "≤ 12",
      operations: "479M ops at n=12",
      performance: "Very Poor",
      examples: [
        "Generating all permutations",
        "Traveling Salesman (brute force)",
        "Solving N-Queens (naive)",
        "All possible arrangements",
        "Exhaustive search problems"
      ],
      code: `// O(n!) - Factorial Time
// WARNING: Only works for n ≤ 10!
function generatePermutations(arr) {
    if (arr.length <= 1) return [arr];
    
    const result = [];
    
    for (let i = 0; i < arr.length; i++) {
        const current = arr[i];
        const remaining = [...arr.slice(0, i), ...arr.slice(i + 1)];
        const perms = generatePermutations(remaining);
        
        for (const perm of perms) {
            result.push([current, ...perm]);
        }
    }
    
    return result;
}

// n! grows: 5! = 120, 10! = 3.6M, 12! = 479M`
    }
  ],
  spaceComplexities: [
    {
      notation: "O(1)",
      name: "Constant Space",
      description: "Uses fixed amount of memory regardless of input size",
      color: "#10b981",
      examples: [
        "Few variables",
        "In-place algorithms",
        "Two pointers technique",
        "Swapping elements"
      ]
    },
    {
      notation: "O(n)",
      name: "Linear Space",
      description: "Memory usage grows proportionally with input size",
      color: "#8b5cf6",
      examples: [
        "Creating new array of size n",
        "Hash map with n entries",
        "Recursion call stack",
        "Storing all elements"
      ]
    },
    {
      notation: "O(n²)",
      name: "Quadratic Space",
      description: "Memory usage grows quadratically with input",
      color: "#ef4444",
      examples: [
        "2D matrix of size n×n",
        "Graph adjacency matrix",
        "Dynamic programming table",
        "Storing all pairs"
      ]
    }
  ],
  visualizations: {
    complexityChart: {
      image: "/mnt/user-data/uploads/Big_O.jpg",
      caption: "Visual representation of how different time complexities scale with input size"
    },
    dataStructures: {
      image: "/mnt/user-data/uploads/Data.png",
      caption: "Time and space complexity of common data structures"
    },
    dataOperations: {
      image: "/mnt/user-data/uploads/DataOperation.jpg",
      caption: "Complexity of operations on various data structures"
    },
    algorithms: {
      image: "/mnt/user-data/uploads/Algorithms.png",
      caption: "Time and space complexity of sorting algorithms"
    }
  },
  keyInsights: [
    {
      title: "Drop Constants & Lower Terms",
      description: "O(2n) = O(n), O(n² + n) = O(n²). Focus on the dominant term.",
      icon: "info"
    },
    {
      title: "Logarithmic is Extremely Fast",
      description: "log(1,000,000) ≈ 20. Binary search on a million items takes only ~20 steps!",
      icon: "trending-up"
    },
    {
      title: "Know Your Limits",
      description: "O(n²) works for n ≤ 3K, O(n) for n ≤ 1M, O(2ⁿ) for n ≤ 20. Plan accordingly.",
      icon: "alert-circle"
    },
    {
      title: "Space-Time Tradeoff",
      description: "Often you can trade memory for speed. Hash tables use O(n) space for O(1) lookups.",
      icon: "zap"
    }
  ],
  interviewTips: [
    {
      number: 1,
      title: "Always Analyze Both Time and Space",
      description: "Interviewers want to see you consider memory usage, not just runtime. Mention both complexities."
    },
    {
      number: 2,
      title: "Explain Your Reasoning",
      description: "Walk through why your solution has a certain complexity. Count loops, recursive calls, etc."
    },
    {
      number: 3,
      title: "Consider Amortized Complexity",
      description: "Some operations are expensive occasionally but average out. Dynamic array resize is O(1) amortized."
    },
    {
      number: 4,
      title: "Optimize When Necessary",
      description: "Start with working solution, then optimize. Explain trade-offs between different approaches."
    }
  ]
},

  dataStructures: {
    title: "Data Structures",
    icon: "Database",
    sections: [
      {
        title: "Arrays & Strings",
        complexity: {
          access: "O(1)",
          search: "O(n)",
          insertion: "O(n)",
          deletion: "O(n)",
        },
        techniques: [
          "Two pointers",
          "Sliding window",
          "Prefix sum",
          "Kadane's algorithm",
        ],
        code: {
          javascript: `// Two Pointers
const twoSum = (arr, target) => {
  let left = 0, right = arr.length - 1;
  while (left < right) {
    const sum = arr[left] + arr[right];
    if (sum === target) return [left, right];
    sum < target ? left++ : right--;
  }
  return [-1, -1];
};

// Sliding Window
const maxSubarraySum = (arr, k) => {
  let maxSum = 0, windowSum = 0;
  for (let i = 0; i < k; i++) windowSum += arr[i];
  maxSum = windowSum;
  
  for (let i = k; i < arr.length; i++) {
    windowSum += arr[i] - arr[i - k];
    maxSum = Math.max(maxSum, windowSum);
  }
  return maxSum;
};`,
          python: `# Two Pointers
def two_sum(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]

# Sliding Window
def max_subarray_sum(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum`,
          java: `// Two Pointers
public int[] twoSum(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) return new int[]{left, right};
        if (sum < target) left++;
        else right--;
    }
    return new int[]{-1, -1};
}`,
        },
      },
      {
        title: "Linked Lists",
        complexity: {
          access: "O(n)",
          search: "O(n)",
          insertion: "O(1)",
          deletion: "O(1)",
        },
        techniques: ["Fast & slow pointers", "Dummy node", "Reverse"],
        code: {
          javascript: `class ListNode {
  constructor(val = 0, next = null) {
    this.val = val;
    this.next = next;
  }
}

// Detect cycle (Fast & Slow)
const hasCycle = (head) => {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
};

// Reverse linked list
const reverse = (head) => {
  let prev = null, curr = head;
  while (curr) {
    const next = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
  }
  return prev;
};`,
          python: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Detect cycle
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Reverse linked list
def reverse(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev`,
        },
      },
      {
        title: "Stacks & Queues",
        complexity: {
          push: "O(1)",
          pop: "O(1)",
          peek: "O(1)",
        },
        techniques: ["Monotonic stack", "BFS with queue"],
        code: {
          javascript: `// Stack - Valid Parentheses
const isValid = (s) => {
  const stack = [];
  const pairs = { '(': ')', '{': '}', '[': ']' };
  
  for (const char of s) {
    if (char in pairs) {
      stack.push(char);
    } else {
      const last = stack.pop();
      if (pairs[last] !== char) return false;
    }
  }
  return stack.length === 0;
};

// Queue - BFS
const levelOrder = (root) => {
  if (!root) return [];
  const result = [], queue = [root];
  
  while (queue.length) {
    const level = [];
    const size = queue.length;
    
    for (let i = 0; i < size; i++) {
      const node = queue.shift();
      level.push(node.val);
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
    result.push(level);
  }
  return result;
};`,
        },
      },
      {
        title: "Hash Tables",
        complexity: {
          search: "O(1) avg",
          insert: "O(1) avg",
          delete: "O(1) avg",
        },
        techniques: [
          "Frequency counting",
          "Two sum pattern",
          "Anagram detection",
        ],
        code: {
          javascript: `// Frequency Counter
const charCount = (str) => {
  const freq = new Map();
  for (const char of str) {
    freq.set(char, (freq.get(char) || 0) + 1);
  }
  return freq;
};

// Two Sum
const twoSum = (nums, target) => {
  const map = new Map();
  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];
    if (map.has(complement)) {
      return [map.get(complement), i];
    }
    map.set(nums[i], i);
  }
  return [];
};`,
          python: `# Frequency Counter
def char_count(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Two Sum
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`,
        },
      },
      {
        title: "Trees & Graphs",
        complexity: {
          search: "O(log n) - O(n)",
          insert: "O(log n) - O(n)",
          delete: "O(log n) - O(n)",
        },
        techniques: ["DFS (Pre/In/Post)", "BFS", "Backtracking"],
        code: {
          javascript: `class TreeNode {
  constructor(val = 0, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

// DFS - Inorder Traversal (Iterative)
const inorder = (root) => {
  const result = [], stack = [];
  let curr = root;
  
  while (curr || stack.length) {
    while (curr) {
      stack.push(curr);
      curr = curr.left;
    }
    curr = stack.pop();
    result.push(curr.val);
    curr = curr.right;
  }
  return result;
};

// Max Depth
const maxDepth = (root) => {
  if (!root) return 0;
  return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
};

// Validate BST
const isValidBST = (root, min = -Infinity, max = Infinity) => {
  if (!root) return true;
  if (root.val <= min || root.val >= max) return false;
  return isValidBST(root.left, min, root.val) && 
         isValidBST(root.right, root.val, max);
};`,
        },
      },
      {
        title: "Heaps (Priority Queue)",
        complexity: {
          insert: "O(log n)",
          deleteMin: "O(log n)",
          peek: "O(1)",
        },
        techniques: ["Top K elements", "Merge K sorted"],
        code: {
          javascript: `// Min Heap implementation
class MinHeap {
  constructor() {
    this.heap = [];
  }
  
  push(val) {
    this.heap.push(val);
    this.bubbleUp(this.heap.length - 1);
  }
  
  pop() {
    if (this.heap.length === 0) return null;
    const min = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.bubbleDown(0);
    }
    return min;
  }
  
  peek() {
    return this.heap[0];
  }
  
  bubbleUp(idx) {
    while (idx > 0) {
      const parent = Math.floor((idx - 1) / 2);
      if (this.heap[parent] <= this.heap[idx]) break;
      [this.heap[parent], this.heap[idx]] = [this.heap[idx], this.heap[parent]];
      idx = parent;
    }
  }
  
  bubbleDown(idx) {
    while (true) {
      let smallest = idx;
      const left = 2 * idx + 1;
      const right = 2 * idx + 2;
      
      if (left < this.heap.length && this.heap[left] < this.heap[smallest]) {
        smallest = left;
      }
      if (right < this.heap.length && this.heap[right] < this.heap[smallest]) {
        smallest = right;
      }
      if (smallest === idx) break;
      
      [this.heap[idx], this.heap[smallest]] = [this.heap[smallest], this.heap[idx]];
      idx = smallest;
    }
  }
}`,
        },
      },
      {
        title: "Tries",
        complexity: {
          search: "O(m)",
          insert: "O(m)",
          delete: "O(m)",
        },
        techniques: ["Prefix search", "Word dictionary"],
        code: {
          javascript: `class TrieNode {
  constructor() {
    this.children = {};
    this.isEnd = false;
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
  }
  
  insert(word) {
    let node = this.root;
    for (const char of word) {
      if (!node.children[char]) {
        node.children[char] = new TrieNode();
      }
      node = node.children[char];
    }
    node.isEnd = true;
  }
  
  search(word) {
    let node = this.root;
    for (const char of word) {
      if (!node.children[char]) return false;
      node = node.children[char];
    }
    return node.isEnd;
  }
  
  startsWith(prefix) {
    let node = this.root;
    for (const char of prefix) {
      if (!node.children[char]) return false;
      node = node.children[char];
    }
    return true;
  }
}`,
        },
      },
    ],
  },

  algorithms: {
    title: "Algorithms",
    icon: "Code2",
    sections: [
      {
        title: "Sorting",
        items: [
          {
            name: "Quick Sort",
            time: "O(n log n) avg, O(n²) worst",
            space: "O(log n)",
            stable: false,
            code: {
              javascript: `const quickSort = (arr, low = 0, high = arr.length - 1) => {
  if (low < high) {
    const pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
  return arr;
};

const partition = (arr, low, high) => {
  const pivot = arr[high];
  let i = low - 1;
  
  for (let j = low; j < high; j++) {
    if (arr[j] < pivot) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
  return i + 1;
};`,
            },
          },
          {
            name: "Merge Sort",
            time: "O(n log n)",
            space: "O(n)",
            stable: true,
            code: {
              javascript: `const mergeSort = (arr) => {
  if (arr.length <= 1) return arr;
  
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  
  return merge(left, right);
};

const merge = (left, right) => {
  const result = [];
  let i = 0, j = 0;
  
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }
  
  return result.concat(left.slice(i), right.slice(j));
};`,
            },
          },
          {
            name: "Heap Sort",
            time: "O(n log n)",
            space: "O(1)",
            stable: false,
            code: {
              javascript: `const heapSort = (arr) => {
  const n = arr.length;
  
  // Build max heap
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    heapify(arr, n, i);
  }
  
  // Extract elements from heap
  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]];
    heapify(arr, i, 0);
  }
  return arr;
};

const heapify = (arr, n, i) => {
  let largest = i;
  const left = 2 * i + 1;
  const right = 2 * i + 2;
  
  if (left < n && arr[left] > arr[largest]) largest = left;
  if (right < n && arr[right] > arr[largest]) largest = right;
  
  if (largest !== i) {
    [arr[i], arr[largest]] = [arr[largest], arr[i]];
    heapify(arr, n, largest);
  }
};`,
            },
          },
        ],
      },
      {
        title: "Searching",
        items: [
          {
            name: "Binary Search",
            time: "O(log n)",
            space: "O(1)",
            code: {
              javascript: `const binarySearch = (arr, target) => {
  let left = 0, right = arr.length - 1;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
};

// Find first occurrence
const firstOccurrence = (arr, target) => {
  let left = 0, right = arr.length - 1;
  let result = -1;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    
    if (arr[mid] === target) {
      result = mid;
      right = mid - 1; // Continue searching left
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return result;
};`,
            },
          },
          {
            name: "DFS (Graph)",
            time: "O(V + E)",
            space: "O(V)",
            code: {
              javascript: `// Recursive DFS
const dfs = (graph, start, visited = new Set()) => {
  visited.add(start);
  console.log(start);
  
  for (const neighbor of graph[start] || []) {
    if (!visited.has(neighbor)) {
      dfs(graph, neighbor, visited);
    }
  }
  return visited;
};

// Iterative DFS
const dfsIterative = (graph, start) => {
  const visited = new Set();
  const stack = [start];
  
  while (stack.length) {
    const node = stack.pop();
    
    if (!visited.has(node)) {
      visited.add(node);
      console.log(node);
      
      for (const neighbor of graph[node] || []) {
        if (!visited.has(neighbor)) {
          stack.push(neighbor);
        }
      }
    }
  }
  return visited;
};`,
            },
          },
          {
            name: "BFS (Graph)",
            time: "O(V + E)",
            space: "O(V)",
            code: {
              javascript: `const bfs = (graph, start) => {
  const visited = new Set([start]);
  const queue = [start];
  
  while (queue.length) {
    const node = queue.shift();
    console.log(node);
    
    for (const neighbor of graph[node] || []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  return visited;
};

// Level-order with distance tracking
const bfsWithDistance = (graph, start) => {
  const visited = new Set([start]);
  const queue = [[start, 0]];
  const distances = { [start]: 0 };
  
  while (queue.length) {
    const [node, dist] = queue.shift();
    
    for (const neighbor of graph[node] || []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        distances[neighbor] = dist + 1;
        queue.push([neighbor, dist + 1]);
      }
    }
  }
  return distances;
};`,
            },
          },
        ],
      },
      {
        title: "Dynamic Programming",
        items: [
          {
            name: "Fibonacci (Memoization)",
            time: "O(n)",
            space: "O(n)",
            code: {
              javascript: `// Top-down (Memoization)
const fib = (n, memo = {}) => {
  if (n <= 1) return n;
  if (memo[n]) return memo[n];
  
  memo[n] = fib(n - 1, memo) + fib(n - 2, memo);
  return memo[n];
};

// Bottom-up (Tabulation)
const fibTabulation = (n) => {
  if (n <= 1) return n;
  
  const dp = [0, 1];
  for (let i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2];
  }
  return dp[n];
};

// Space optimized
const fibOptimized = (n) => {
  if (n <= 1) return n;
  
  let prev = 0, curr = 1;
  for (let i = 2; i <= n; i++) {
    [prev, curr] = [curr, prev + curr];
  }
  return curr;
};`,
            },
          },
          {
            name: "0/1 Knapsack",
            time: "O(n·W)",
            space: "O(n·W)",
            code: {
              javascript: `const knapsack = (weights, values, capacity) => {
  const n = weights.length;
  const dp = Array(n + 1).fill(0).map(() => Array(capacity + 1).fill(0));
  
  for (let i = 1; i <= n; i++) {
    for (let w = 1; w <= capacity; w++) {
      if (weights[i - 1] <= w) {
        dp[i][w] = Math.max(
          dp[i - 1][w],
          values[i - 1] + dp[i - 1][w - weights[i - 1]]
        );
      } else {
        dp[i][w] = dp[i - 1][w];
      }
    }
  }
  return dp[n][capacity];
};`,
            },
          },
          {
            name: "Longest Common Subsequence",
            time: "O(m·n)",
            space: "O(m·n)",
            code: {
              javascript: `const lcs = (text1, text2) => {
  const m = text1.length, n = text2.length;
  const dp = Array(m + 1).fill(0).map(() => Array(n + 1).fill(0));
  
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (text1[i - 1] === text2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  return dp[m][n];
};`,
            },
          },
        ],
      },
      {
        title: "Backtracking",
        items: [
          {
            name: "Permutations",
            time: "O(n!)",
            space: "O(n)",
            code: {
              javascript: `const permute = (nums) => {
  const result = [];
  
  const backtrack = (path, remaining) => {
    if (remaining.length === 0) {
      result.push([...path]);
      return;
    }
    
    for (let i = 0; i < remaining.length; i++) {
      path.push(remaining[i]);
      const newRemaining = [...remaining.slice(0, i), ...remaining.slice(i + 1)];
      backtrack(path, newRemaining);
      path.pop();
    }
  };
  
  backtrack([], nums);
  return result;
};`,
            },
          },
          {
            name: "Subsets",
            time: "O(2ⁿ)",
            space: "O(n)",
            code: {
              javascript: `const subsets = (nums) => {
  const result = [];
  
  const backtrack = (start, path) => {
    result.push([...path]);
    
    for (let i = start; i < nums.length; i++) {
      path.push(nums[i]);
      backtrack(i + 1, path);
      path.pop();
    }
  };
  
  backtrack(0, []);
  return result;
};`,
            },
          },
          {
            name: "N-Queens",
            time: "O(n!)",
            space: "O(n²)",
            code: {
              javascript: `const solveNQueens = (n) => {
  const result = [];
  const board = Array(n).fill(0).map(() => Array(n).fill('.'));
  
  const isValid = (row, col) => {
    // Check column
    for (let i = 0; i < row; i++) {
      if (board[i][col] === 'Q') return false;
    }
    
    // Check diagonal
    for (let i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
      if (board[i][j] === 'Q') return false;
    }
    
    // Check anti-diagonal
    for (let i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
      if (board[i][j] === 'Q') return false;
    }
    
    return true;
  };
  
  const backtrack = (row) => {
    if (row === n) {
      result.push(board.map(r => r.join('')));
      return;
    }
    
    for (let col = 0; col < n; col++) {
      if (isValid(row, col)) {
        board[row][col] = 'Q';
        backtrack(row + 1);
        board[row][col] = '.';
      }
    }
  };
  
  backtrack(0);
  return result;
};`,
            },
          },
        ],
      },
    ],
  },

  patterns: {
    title: "DSA Patterns & Roadmap",
    icon: "Lightbulb",
    sections: [
      {
        title: "1. Fast and Slow Pointer",
        description:
          "Use two pointers moving at different speeds to detect cycles or find middle elements",
        useCases: [
          "Detecting cycles in linked lists",
          "Finding the middle of a linked list",
          "Finding duplicate numbers in arrays",
          "Checking if a linked list is a palindrome",
        ],
        code: {
          javascript: `function fastSlowPointer(head) {
    let slow = head;
    let fast = head;
    let ans = 0;
    
    while (fast && fast.next) {
        // Do logic
        slow = slow.next;
        fast = fast.next.next;
    }
    
    return ans;
}`,
          python: `def fast_slow_pointer(head):
    slow = head
    fast = head
    ans = 0
    
    while fast and fast.next:
        # Do logic
        slow = slow.next
        fast = fast.next.next
    
    return ans`,
          java: `public int fastSlowPointer(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
    int ans = 0;
    
    while (fast != null && fast.next != null) {
        // do logic
        slow = slow.next;
        fast = fast.next.next;
    }
    
    return ans;
}`,
          cpp: `int fastSlowPointer(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    int ans = 0;
    
    while (fast && fast->next) {
        // do logic
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return ans;
}`,
        },
        problems: [
          "Linked List Cycle II",
          "Remove nth Node from the End of List",
          "Find the Duplicate Number",
          "Palindrome Linked List",
        ],
      },
      {
        title: "2. Two Pointers: One Input, Opposite Ends",
        description: "Start from both ends of array and move towards center",
        useCases: [
          "Finding pairs in sorted arrays",
          "Palindrome checking",
          "Container with most water problems",
          "Trapping rain water",
        ],
        code: {
          javascript: `function twoPointers(arr) {
    let left = 0;
    let right = arr.length - 1;
    let ans = 0;
    
    while (left < right) {
        // Do some logic here with left and right
        if (CONDITION) {
            left++;
        } else {
            right--;
        }
    }
    
    return ans;
}`,
          python: `def two_pointers(arr):
    left = 0
    right = len(arr) - 1
    ans = 0
    
    while left < right:
        # Do some logic here with left and right
        if CONDITION:
            left += 1
        else:
            right -= 1
    
    return ans`,
          java: `public int twoPointers(int[] arr) {
    int left = 0;
    int right = arr.length - 1;
    int ans = 0;
    
    while (left < right) {
        // do some logic here with left and right
        if (CONDITION) {
            left++;
        } else {
            right--;
        }
    }
    
    return ans;
}`,
          cpp: `int twoPointers(vector<int>& arr) {
    int left = 0;
    int right = arr.size() - 1;
    int ans = 0;
    
    while (left < right) {
        // do some logic here with left and right
        if (CONDITION) {
            left++;
        } else {
            right--;
        }
    }
    
    return ans;
}`,
        },
        problems: [
          "Two Sum II - Input Array is Sorted",
          "Dutch National Flag: Sort Colors",
          "Next Permutation",
          "Bag of Tokens",
          "Container with most water",
          "Trapping Rain Water",
        ],
      },
      {
        title: "3. Two Pointers: Two Inputs, Exhaust Both",
        description: "Merge or compare two sorted arrays/lists efficiently",
        code: {
          javascript: `function twoPointers(arr1, arr2) {
    let i = 0, j = 0, ans = 0;
    
    while (i < arr1.length && j < arr2.length) {
        // Do some logic here
        if (CONDITION) {
            i++;
        } else {
            j++;
        }
    }
    
    while (i < arr1.length) {
        // Do logic
        i++;
    }
    
    while (j < arr2.length) {
        // Do logic
        j++;
    }
    
    return ans;
}`,
          python: `def two_pointers(arr1, arr2):
    i = j = ans = 0
    
    while i < len(arr1) and j < len(arr2):
        # Do some logic here
        if CONDITION:
            i += 1
        else:
            j += 1
    
    while i < len(arr1):
        # Do logic
        i += 1
    
    while j < len(arr2):
        # Do logic
        j += 1
    
    return ans`,
          java: `public int twoPointers(int[] arr1, int[] arr2) {
    int i = 0, j = 0, ans = 0;
    
    while (i < arr1.length && j < arr2.length) {
        // do some logic here
        if (CONDITION) {
            i++;
        } else {
            j++;
        }
    }
    
    while (i < arr1.length) {
        // do logic
        i++;
    }
    
    while (j < arr2.length) {
        // do logic
        j++;
    }
    
    return ans;
}`,
          cpp: `int twoPointers(vector<int>& arr1, vector<int>& arr2) {
    int i = 0, j = 0, ans = 0;
    
    while (i < arr1.size() && j < arr2.size()) {
        // do some logic here
        if (CONDITION) {
            i++;
        } else {
            j++;
        }
    }
    
    while (i < arr1.size()) {
        // do logic
        i++;
    }
    
    while (j < arr2.size()) {
        // do logic
        j++;
    }
    
    return ans;
}`,
        },
        problems: [
          "Merge Sorted Array",
          "Intersection of Two Arrays",
          "Intersection of Two Arrays II",
        ],
      },
      {
        title: "4. Sliding Window",
        description: "Fixed or dynamic window that slides through array",
        categories: [
          {
            category: "Fixed Size",
            problems: [
              "Maximum Sum Subarray of Size K",
              "Number of Subarrays having Average Greater or Equal to Threshold",
              "Repeated DNA sequences",
              "Permutation in String",
              "Sliding Subarray Beauty",
              "Sliding Window Maximum",
            ],
          },
          {
            category: "Variable Size",
            problems: [
              "Longest Substring Without Repeating Characters",
              "Minimum Size Subarray Sum",
              "Subarray Product Less Than K",
              "Max Consecutive Ones",
              "Fruits Into Baskets",
              "Count Number of Nice Subarrays",
              "Minimum Window Substring",
            ],
          },
        ],
        code: {
          javascript: `function slidingWindow(arr) {
    let left = 0, ans = 0, curr = 0;
    
    for (let right = 0; right < arr.length; right++) {
        // Do logic here to add arr[right] to curr
        
        while (WINDOW_CONDITION_BROKEN) {
            // Remove arr[left] from curr
            left++;
        }
        
        // Update ans
    }
    
    return ans;
}`,
          python: `def sliding_window(arr):
    left = ans = curr = 0
    
    for right in range(len(arr)):
        # do logic here to add arr[right] to curr
        
        while WINDOW_CONDITION_BROKEN:
            # remove arr[left] from curr
            left += 1
        
        # update ans
    
    return ans`,
          java: `public int slidingWindow(int[] arr) {
    int left = 0, ans = 0, curr = 0;
    
    for (int right = 0; right < arr.length; right++) {
        // do logic here to add arr[right] to curr
        
        while (WINDOW_CONDITION_BROKEN) {
            // remove arr[left] from curr
            left++;
        }
        
        // update ans
    }
    
    return ans;
}`,
          cpp: `int slidingWindow(vector<int>& arr) {
    int left = 0, ans = 0, curr = 0;
    
    for (int right = 0; right < arr.size(); right++) {
        // do logic here to add arr[right] to curr
        
        while (WINDOW_CONDITION_BROKEN) {
            // remove arr[left] from curr
            left++;
        }
        
        // update ans
    }
    
    return ans;
}`,
        },
      },
      {
        title: "5. Prefix Sum",
        description: "Precompute cumulative sums for range queries",
        code: {
          javascript: `function buildPrefixSum(arr) {
    let prefix = [arr[0]];
    
    for (let i = 1; i < arr.length; i++) {
        prefix.push(arr[i] + prefix[prefix.length - 1]);
    }
    
    return prefix;
}`,
          python: `def build_prefix_sum(arr):
    prefix = [arr[0]]
    
    for i in range(1, len(arr)):
        prefix.append(arr[i] + prefix[-1])
    
    return prefix`,
          java: `public int[] buildPrefixSum(int[] arr) {
    int[] prefix = new int[arr.length];
    prefix[0] = arr[0];
    
    for (int i = 1; i < arr.length; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }
    
    return prefix;
}`,
          cpp: `vector<int> buildPrefixSum(vector<int>& arr) {
    vector<int> prefix(arr.size());
    prefix[0] = arr[0];
    
    for (int i = 1; i < arr.size(); i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }
    
    return prefix;
}`,
        },
        problems: [
          "Find the middle index in array",
          "Product of array except self",
          "Maximum product subarray",
          "Number of ways to split array",
          "Range Sum Query 2D",
        ],
      },
      {
        title: "6. Overlapping Intervals",
        description: "Sort intervals and merge overlapping ones",
        problems: [
          "Merge Intervals",
          "Insert Interval",
          "My Calendar II",
          "Minimum Number of Arrows to Burst Balloons",
          "Non-overlapping Intervals",
        ],
      },
      {
        title: "7. Cyclic Sort (Index-Based)",
        description: "Sort array by placing each element at its correct index",
        useCases: [
          "Finding missing numbers in arrays",
          "Finding duplicate numbers",
          "Array elements are in range [1, n]",
          "Problems involving indices matching values",
        ],
        code: {
          javascript: `function cyclicSort(arr) {
    let i = 0;
    
    while (i < arr.length) {
        const correctIdx = arr[i] - 1;
        if (arr[i] !== arr[correctIdx]) {
            // Swap
            [arr[i], arr[correctIdx]] = [arr[correctIdx], arr[i]];
        } else {
            i++;
        }
    }
    
    return arr;
}`,
          python: `def cyclic_sort(arr):
    i = 0
    
    while i < len(arr):
        correct_idx = arr[i] - 1
        if arr[i] != arr[correct_idx]:
            # Swap
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    
    return arr`,
          java: `public void cyclicSort(int[] arr) {
    int i = 0;
    
    while (i < arr.length) {
        int correctIdx = arr[i] - 1;
        if (arr[i] != arr[correctIdx]) {
            // Swap
            int temp = arr[i];
            arr[i] = arr[correctIdx];
            arr[correctIdx] = temp;
        } else {
            i++;
        }
    }
}`,
          cpp: `void cyclicSort(vector<int>& arr) {
    int i = 0;
    
    while (i < arr.size()) {
        int correctIdx = arr[i] - 1;
        if (arr[i] != arr[correctIdx]) {
            // Swap
            swap(arr[i], arr[correctIdx]);
        } else {
            i++;
        }
    }
}`,
        },
        problems: [
          "Missing Number",
          "Find Missing Numbers",
          "Set Mismatch",
          "First Missing Positive",
        ],
      },
      {
        title: "8. Reversal of Linked List (In-place)",
        description: "Reverse pointers in linked list without extra space",
        useCases: [
          "Reversing entire linked list",
          "Reversing k-group nodes",
          "Reversing sub-list between positions",
          "Palindrome checking in linked lists",
        ],
        code: {
          javascript: `function reverseLinkedList(head) {
    let prev = null;
    let curr = head;
    
    while (curr) {
        const nextNode = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextNode;
    }
    
    return prev;
}`,
          python: `def reverse_linked_list(head):
    prev = None
    curr = head
    
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    return prev`,
          java: `public ListNode reverseLinkedList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    
    while (curr != null) {
        ListNode nextNode = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextNode;
    }
    
    return prev;
}`,
          cpp: `ListNode* reverseLinkedList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr) {
        ListNode* nextNode = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nextNode;
    }
    
    return prev;
}`,
        },
        problems: [
          "Reverse Linked List",
          "Reverse Nodes in k-Group",
          "Swap Nodes in Pairs",
        ],
      },
      {
        title: "9. Matrix Manipulation",
        description:
          "In-place matrix rotation, spiral traversal, and modifications",
        useCases: [
          "Rotating matrix 90/180/270 degrees",
          "Spiral order traversal",
          "Setting rows/columns to zero",
          "Matrix searching and modification",
        ],
        code: {
          javascript: `// Rotate 90 degrees clockwise
function rotateMatrix(matrix) {
    const n = matrix.length;
    
    // Transpose
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
        }
    }
    
    // Reverse each row
    for (let i = 0; i < n; i++) {
        matrix[i].reverse();
    }
}`,
          python: `def rotate_matrix(matrix):
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()`,
          java: `public void rotateMatrix(int[][] matrix) {
    int n = matrix.length;
    
    // Transpose
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
    
    // Reverse each row
    for (int i = 0; i < n; i++) {
        int left = 0, right = n - 1;
        while (left < right) {
            int temp = matrix[i][left];
            matrix[i][left] = matrix[i][right];
            matrix[i][right] = temp;
            left++;
            right--;
        }
    }
}`,
          cpp: `void rotateMatrix(vector<vector<int>>& matrix) {
    int n = matrix.size();
    
    // Transpose
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
    
    // Reverse each row
    for (int i = 0; i < n; i++) {
        reverse(matrix[i].begin(), matrix[i].end());
    }
}`,
        },
        problems: [
          "Rotate Image",
          "Spiral Matrix",
          "Set Matrix Zeroes",
          "Game of Life",
        ],
      },
      {
        title: "10. Breadth First Search (BFS)",
        description: "Level-order traversal using queue",
        useCases: [
          "Shortest path in unweighted graphs",
          "Level-order tree traversal",
          "Finding all nodes at distance K",
          "Multi-source BFS problems",
        ],
        code: {
          javascript: `function bfs(graph, start) {
    const queue = [start];
    const seen = new Set([start]);
    let ans = 0;
    
    while (queue.length) {
        const node = queue.shift();
        // Do some logic
        
        for (const neighbor of graph[node] || []) {
            if (!seen.has(neighbor)) {
                seen.add(neighbor);
                queue.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
          python: `from collections import deque

def bfs(graph, start):
    queue = deque([start])
    seen = {start}
    ans = 0
    
    while queue:
        node = queue.popleft()
        # Do some logic
        
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    
    return ans`,
          java: `public int bfs(Map<Integer, List<Integer>> graph, int start) {
    Queue<Integer> queue = new LinkedList<>();
    Set<Integer> seen = new HashSet<>();
    queue.add(start);
    seen.add(start);
    int ans = 0;
    
    while (!queue.isEmpty()) {
        int node = queue.remove();
        // do some logic
        
        for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
            if (!seen.contains(neighbor)) {
                seen.add(neighbor);
                queue.add(neighbor);
            }
        }
    }
    
    return ans;
}`,
          cpp: `int bfs(unordered_map<int, vector<int>>& graph, int start) {
    queue<int> q;
    unordered_set<int> seen;
    q.push(start);
    seen.insert(start);
    int ans = 0;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        // do some logic
        
        for (int neighbor : graph[node]) {
            if (seen.find(neighbor) == seen.end()) {
                seen.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
        },
        problems: [
          "Shortest Path in Binary Matrix",
          "Rotten Oranges",
          "As Far From Land as Possible",
          "Word Ladder",
        ],
      },
      {
        title: "11. Depth First Search (DFS)",
        description:
          "Explore as far as possible along each branch before backtracking",
        useCases: [
          "Finding connected components",
          "Detecting cycles in graphs",
          "Path finding problems",
          "Tree traversals (preorder, inorder, postorder)",
        ],
        code: {
          javascript: `// Recursive DFS
function dfs(node, graph, seen) {
    let ans = 0;
    
    for (const neighbor of graph[node] || []) {
        if (!seen.has(neighbor)) {
            seen.add(neighbor);
            ans += dfs(neighbor, graph, seen);
        }
    }
    
    return ans;
}

// Iterative DFS
function dfsIterative(graph, start) {
    const stack = [start];
    const seen = new Set([start]);
    let ans = 0;
    
    while (stack.length) {
        const node = stack.pop();
        // Do some logic
        
        for (const neighbor of graph[node] || []) {
            if (!seen.has(neighbor)) {
                seen.add(neighbor);
                stack.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
          python: `def dfs(node, graph, seen):
    ans = 0
    
    for neighbor in graph.get(node, []):
        if neighbor not in seen:
            seen.add(neighbor)
            ans += dfs(neighbor, graph, seen)
    
    return ans

def dfs_iterative(graph, start):
    stack = [start]
    seen = {start}
    ans = 0
    
    while stack:
        node = stack.pop()
        # Do some logic
        
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    
    return ans`,
          java: `Set<Integer> seen = new HashSet<>();

public int dfs(int node, Map<Integer, List<Integer>> graph) {
    int ans = 0;
    
    for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
        if (!seen.contains(neighbor)) {
            seen.add(neighbor);
            ans += dfs(neighbor, graph);
        }
    }
    
    return ans;
}

public int dfsIterative(Map<Integer, List<Integer>> graph, int start) {
    Stack<Integer> stack = new Stack<>();
    Set<Integer> seen = new HashSet<>();
    stack.push(start);
    seen.add(start);
    int ans = 0;
    
    while (!stack.empty()) {
        int node = stack.pop();
        // do some logic
        
        for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
            if (!seen.contains(neighbor)) {
                seen.add(neighbor);
                stack.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
          cpp: `unordered_set<int> seen;

int dfs(int node, unordered_map<int, vector<int>>& graph) {
    int ans = 0;
    
    for (int neighbor : graph[node]) {
        if (seen.find(neighbor) == seen.end()) {
            seen.insert(neighbor);
            ans += dfs(neighbor, graph);
        }
    }
    
    return ans;
}

int dfsIterative(unordered_map<int, vector<int>>& graph, int start) {
    stack<int> stk;
    unordered_set<int> seen;
    stk.push(start);
    seen.insert(start);
    int ans = 0;
    
    while (!stk.empty()) {
        int node = stk.top();
        stk.pop();
        // do some logic
        
        for (int neighbor : graph[node]) {
            if (seen.find(neighbor) == seen.end()) {
                seen.insert(neighbor);
                stk.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
        },
        problems: [
          "Number of Closed Islands",
          "Coloring a Border",
          "Number of Enclaves",
          "Time Needed to Inform all Employees",
          "Find Eventual Safe States",
        ],
      },
      {
        title: "12. Backtracking",
        description:
          "Build solution incrementally and backtrack when constraint violated",
        useCases: [
          "Generating all permutations/combinations",
          "N-Queens problem",
          "Sudoku solver",
          "Word search in matrix",
          "Subset generation",
        ],
        code: {
          javascript: `function backtrack(curr, ...otherArgs) {
    if (BASE_CASE) {
        // Modify the answer
        return 0;
    }
    
    let ans = 0;
    for (const option of OPTIONS) {
        // Modify the current state
        curr.push(option);
        
        ans += backtrack(curr, ...otherArgs);
        
        // Undo the modification (backtrack)
        curr.pop();
    }
    
    return ans;
}`,
          python: `def backtrack(curr, *other_args):
    if BASE_CASE:
        # Modify the answer
        return 0
    
    ans = 0
    for option in OPTIONS:
        # Modify the current state
        curr.append(option)
        
        ans += backtrack(curr, *other_args)
        
        # Undo the modification (backtrack)
        curr.pop()
    
    return ans`,
          java: `public int backtrack(List<Integer> curr, OTHER_ARGUMENTS) {
    if (BASE_CASE) {
        // modify the answer
        return 0;
    }
    
    int ans = 0;
    for (ITERATE_OVER_INPUT) {
        // modify the current state
        curr.add(ELEMENT);
        
        ans += backtrack(curr, OTHER_ARGUMENTS);
        
        // undo the modification of the current state
        curr.remove(curr.size() - 1);
    }
    
    return ans;
}`,
          cpp: `int backtrack(vector<int>& curr, OTHER_ARGUMENTS) {
    if (BASE_CASE) {
        // modify the answer
        return 0;
    }
    
    int ans = 0;
    for (ITERATE_OVER_INPUT) {
        // modify the current state
        curr.push_back(element);
        
        ans += backtrack(curr, OTHER_ARGUMENTS);
        
        // undo the modification of the current state
        curr.pop_back();
    }
    
    return ans;
}`,
        },
        problems: [
          "Permutation II",
          "Combination Sum",
          "Generate Parenthesis",
          "N-Queens",
          "Sudoku Solver",
          "Palindrome Partitioning",
          "Word Search",
        ],
      },
      {
        title: "13. Modified Binary Search",
        description:
          "Binary search on rotated arrays, finding boundaries, or answer space",
        useCases: [
          "Search in rotated sorted array",
          "Finding peak element",
          "Finding minimum in rotated array",
          "Finding square root",
          "Capacity/speed optimization problems",
        ],
        categories: [
          {
            category: "Standard Binary Search",
            problems: [
              "Search in Rotated Sorted Array",
              "Find Minimum in Rotated Sorted Array",
              "Find Peak Element",
              "Single element in a sorted array",
            ],
          },
          {
            category: "Binary Search on Answer",
            problems: [
              "Minimum Time to Arrive on Time",
              "Capacity to Ship Packages within 'd' Days",
              "Koko Eating Bananas",
              "Find in Mountain Array",
              "Median of Two Sorted Arrays",
            ],
          },
        ],
        code: {
          javascript: `// Standard Binary Search
function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (arr[mid] === target) {
            return mid;
        }
        
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left; // insertion point
}

// Binary Search for Minimum (Greedy)
function binarySearchMin(arr) {
    let left = MIN_ANSWER;
    let right = MAX_ANSWER;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (check(mid)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

// Binary Search for Maximum (Greedy)
function binarySearchMax(arr) {
    let left = MIN_ANSWER;
    let right = MAX_ANSWER;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (check(mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return right;
}`,
          python: `def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        
        if arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    return left

def binary_search_min(arr):
    left = MIN_ANSWER
    right = MAX_ANSWER
    
    while left <= right:
        mid = (left + right) // 2
        
        if check(mid):
            right = mid - 1
        else:
            left = mid + 1
    
    return left

def binary_search_max(arr):
    left = MIN_ANSWER
    right = MAX_ANSWER
    
    while left <= right:
        mid = (left + right) // 2
        
        if check(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return right`,
          java: `public int binarySearch(int[] arr, int target) {
    int left = 0;
    int right = arr.length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        }
        
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

public int binarySearchMin(int[] arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (check(mid)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

public int binarySearchMax(int[] arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (check(mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return right;
}`,
          cpp: `int binarySearch(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        }
        
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

int binarySearchMin(vector<int>& arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (check(mid)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

int binarySearchMax(vector<int>& arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (check(mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return right;
}`,
        },
      },
      {
        title: "14. Bitwise XOR",
        description: "Use XOR properties to find unique elements",
        useCases: [
          "Finding missing number",
          "Finding single number in array",
          "XOR queries on subarrays",
          "Detecting duplicate numbers",
        ],
        code: {
          javascript: `// Find Missing Number
function findMissing(arr) {
    let xor = 0;
    
    // XOR all array elements
    for (const num of arr) {
        xor ^= num;
    }
    
    // XOR with all numbers from 0 to n
    for (let i = 0; i <= arr.length; i++) {
        xor ^= i;
    }
    
    return xor;
}

// Single Number
function singleNumber(arr) {
    let xor = 0;
    for (const num of arr) {
        xor ^= num;
    }
    return xor;
}`,
          python: `def find_missing(arr):
    xor = 0
    
    # XOR all array elements
    for num in arr:
        xor ^= num
    
    # XOR with all numbers from 0 to n
    for i in range(len(arr) + 1):
        xor ^= i
    
    return xor

def single_number(arr):
    xor = 0
    for num in arr:
        xor ^= num
    return xor`,
          java: `public int findMissing(int[] arr) {
    int xor = 0;
    
    // XOR all array elements
    for (int num : arr) {
        xor ^= num;
    }
    
    // XOR with all numbers from 0 to n
    for (int i = 0; i <= arr.length; i++) {
        xor ^= i;
    }
    
    return xor;
}

public int singleNumber(int[] arr) {
    int xor = 0;
    for (int num : arr) {
        xor ^= num;
    }
    return xor;
}`,
          cpp: `int findMissing(vector<int>& arr) {
    int xor_val = 0;
    
    // XOR all array elements
    for (int num : arr) {
        xor_val ^= num;
    }
    
    // XOR with all numbers from 0 to n
    for (int i = 0; i <= arr.size(); i++) {
        xor_val ^= i;
    }
    
    return xor_val;
}

int singleNumber(vector<int>& arr) {
    int xor_val = 0;
    for (int num : arr) {
        xor_val ^= num;
    }
    return xor_val;
}`,
        },
        problems: [
          "Missing Number",
          "Single Number II",
          "Single Number III",
          "Find the Original array of Prefix XOR",
          "XOR Queries of a Subarray",
        ],
      },
      {
        title: "15. Top 'K' Elements",
        description: "Use heap to efficiently find K largest/smallest elements",
        useCases: [
          "Finding K largest/smallest elements",
          "K closest points to origin",
          "Top K frequent elements",
          "Kth largest element in stream",
        ],
        code: {
          javascript: `function topKElements(arr, k) {
    const heap = new MinHeap(); // or MaxHeap based on problem
    
    for (const num of arr) {
        heap.push(num);
        
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    return heap.toArray();
}

// Using built-in sort (less optimal)
function topKSimple(arr, k) {
    return arr.sort((a, b) => b - a).slice(0, k);
}`,
          python: `import heapq

def top_k_elements(arr, k):
    heap = []
    
    for num in arr:
        heapq.heappush(heap, num)
        
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap

# Using nlargest/nsmallest
def top_k_simple(arr, k):
    return heapq.nlargest(k, arr)`,
          java: `public int[] topKElements(int[] arr, int k) {
    PriorityQueue<Integer> heap = new PriorityQueue<>();
    
    for (int num : arr) {
        heap.add(num);
        
        if (heap.size() > k) {
            heap.remove();
        }
    }
    
    int[] ans = new int[k];
    for (int i = 0; i < k; i++) {
        ans[i] = heap.remove();
    }
    
    return ans;
}`,
          cpp: `vector<int> topKElements(vector<int>& arr, int k) {
    priority_queue<int, vector<int>, greater<int>> heap;
    
    for (int num : arr) {
        heap.push(num);
        
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    vector<int> ans;
    while (!heap.empty()) {
        ans.push_back(heap.top());
        heap.pop();
    }
    
    return ans;
}`,
        },
        problems: [
          "Top K Frequent Elements",
          "Kth Largest Element",
          "Ugly Number II",
          "K Closest Points to Origin",
        ],
      },
      {
        title: "16. K-way Merge",
        description: "Merge K sorted lists using heap",
        useCases: [
          "Merging K sorted arrays/lists",
          "Finding smallest range covering K lists",
          "Kth smallest in sorted matrix",
          "Finding K pairs with smallest sums",
        ],
        code: {
          javascript: `function mergeKLists(lists) {
    const heap = new MinHeap();
    
    // Add first element from each list
    for (let i = 0; i < lists.length; i++) {
        if (lists[i]) {
            heap.push({ val: lists[i].val, listIdx: i, node: lists[i] });
        }
    }
    
    const dummy = new ListNode(0);
    let curr = dummy;
    
    while (heap.size() > 0) {
        const { node, listIdx } = heap.pop();
        curr.next = node;
        curr = curr.next;
        
        if (node.next) {
            heap.push({ val: node.next.val, listIdx, node: node.next });
        }
    }
    
    return dummy.next;
}`,
          python: `import heapq

def merge_k_lists(lists):
    heap = []
    
    # Add first element from each list
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap, (lists[i].val, i, lists[i]))
    
    dummy = ListNode(0)
    curr = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next`,
          java: `public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> heap = new PriorityQueue<>((a, b) -> a.val - b.val);
    
    // Add first element from each list
    for (ListNode node : lists) {
        if (node != null) {
            heap.add(node);
        }
    }
    
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    
    while (!heap.isEmpty()) {
        ListNode node = heap.remove();
        curr.next = node;
        curr = curr.next;
        
        if (node.next != null) {
            heap.add(node.next);
        }
    }
    
    return dummy.next;
}`,
          cpp: `ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> heap(cmp);
    
    // Add first element from each list
    for (ListNode* node : lists) {
        if (node) {
            heap.push(node);
        }
    }
    
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (!heap.empty()) {
        ListNode* node = heap.top();
        heap.pop();
        curr->next = node;
        curr = curr->next;
        
        if (node->next) {
            heap.push(node->next);
        }
    }
    
    return dummy->next;
}`,
        },
        problems: [
          "Find K Pairs with Smallest Sums",
          "Kth Smallest Element in a Sorted Matrix",
          "Merge K Sorted Lists",
          "Smallest Range Covering Elements from K Lists",
        ],
      },
      {
        title: "17. Two Heaps",
        description: "Use min heap and max heap to track median or split data",
        useCases: [
          "Finding median in data stream",
          "Sliding window median",
          "IPO problem",
          "Split data into two halves dynamically",
        ],
        code: {
          javascript: `class MedianFinder {
    constructor() {
        this.maxHeap = new MaxHeap(); // left half
        this.minHeap = new MinHeap(); // right half
    }
    
    addNum(num) {
        // Add to max heap first
        this.maxHeap.push(num);
        
        // Balance: move largest from max to min
        this.minHeap.push(this.maxHeap.pop());
        
        // If min heap is larger, rebalance
        if (this.minHeap.size() > this.maxHeap.size()) {
            this.maxHeap.push(this.minHeap.pop());
        }
    }
    
    findMedian() {
        if (this.maxHeap.size() > this.minHeap.size()) {
            return this.maxHeap.peek();
        }
        return (this.maxHeap.peek() + this.minHeap.peek()) / 2;
    }
}`,
          python: `import heapq

class MedianFinder:
    def __init__(self):
        self.max_heap = []  # left half (invert for max heap)
        self.min_heap = []  # right half
    
    def addNum(self, num):
        # Add to max heap first (negate for max heap behavior)
        heapq.heappush(self.max_heap, -num)
        
        # Balance: move largest from max to min
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        # If min heap is larger, rebalance
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2`,
          java: `class MedianFinder {
    PriorityQueue<Integer> maxHeap; // left half
    PriorityQueue<Integer> minHeap; // right half
    
    public MedianFinder() {
        maxHeap = new PriorityQueue<>((a, b) -> b - a);
        minHeap = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        // Add to max heap first
        maxHeap.add(num);
        
        // Balance: move largest from max to min
        minHeap.add(maxHeap.remove());
        
        // If min heap is larger, rebalance
        if (minHeap.size() > maxHeap.size()) {
            maxHeap.add(minHeap.remove());
        }
    }
    
    public double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.peek();
        }
        return (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
}`,
          cpp: `class MedianFinder {
    priority_queue<int> maxHeap; // left half
    priority_queue<int, vector<int>, greater<int>> minHeap; // right half
    
public:
    MedianFinder() {}
    
    void addNum(int num) {
        // Add to max heap first
        maxHeap.push(num);
        
        // Balance: move largest from max to min
        minHeap.push(maxHeap.top());
        maxHeap.pop();
        
        // If min heap is larger, rebalance
        if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};`,
        },
        problems: [
          "Find Median from Data Stream",
          "Sliding Window Median",
          "IPO",
        ],
      },
      {
        title: "18. Monotonic Stack",
        description: "Stack maintaining monotonic increasing/decreasing order",
        useCases: [
          "Next greater/smaller element",
          "Daily temperatures",
          "Largest rectangle in histogram",
          "Stock span problems",
        ],
        code: {
          javascript: `function monotonicStack(arr) {
    const stack = [];
    const ans = new Array(arr.length);
    
    for (let i = 0; i < arr.length; i++) {
        // For monotonic decreasing, flip > to 
        while (stack.length && arr[stack[stack.length - 1]] > arr[i]) {
            const idx = stack.pop();
            // Do logic with idx
            ans[idx] = i - idx;
        }
        stack.push(i);
    }
    
    return ans;
}`,
          python: `def monotonic_stack(arr):
    stack = []
    ans = [0] * len(arr)
    
    for i in range(len(arr)):
        # For monotonic decreasing, flip > to 
        while stack and arr[stack[-1]] > arr[i]:
            idx = stack.pop()
            # Do logic with idx
            ans[idx] = i - idx
        stack.append(i)
    
    return ans`,
          java: `public int[] monotonicStack(int[] arr) {
    Stack<Integer> stack = new Stack<>();
    int[] ans = new int[arr.length];
    
    for (int i = 0; i < arr.length; i++) {
        // For monotonic decreasing, just flip the > to 
        while (!stack.empty() && arr[stack.peek()] > arr[i]) {
            int idx = stack.pop();
            // do logic with idx
            ans[idx] = i - idx;
        }
        stack.push(i);
    }
    
    return ans;
}`,
          cpp: `vector<int> monotonicStack(vector<int>& arr) {
    stack<int> stk;
    vector<int> ans(arr.size());
    
    for (int i = 0; i < arr.size(); i++) {
        // For monotonic decreasing, just flip the > to 
        while (!stk.empty() && arr[stk.top()] > arr[i]) {
            int idx = stk.top();
            stk.pop();
            // do logic with idx
            ans[idx] = i - idx;
        }
        stk.push(i);
    }
    
    return ans;
}`,
        },
        problems: [
          "Next Greater Element II",
          "Next Greater Node in Linked List",
          "Daily Temperatures",
          "Online Stock Span",
          "Maximum Width Ramp",
          "Largest Rectangle in Histogram",
        ],
      },
      {
        title: "19. Binary Tree Traversals",
        description: "DFS and BFS approaches for tree problems",
        categories: [
          {
            category: "Level Order Traversal (BFS)",
            problems: [
              "Level order Traversal",
              "Zigzag Level order Traversal",
              "Even Odd Tree",
              "Reverse odd Levels",
              "Deepest Leaves Sum",
              "Add one row to Tree",
              "Maximum width of Binary Tree",
              "All Nodes Distance K in Binary tree",
            ],
          },
          {
            category: "Tree Construction",
            problems: [
              "Construct BT from Preorder and Inorder",
              "Construct BT from Postorder and Inorder",
              "Maximum Binary Tree",
              "Construct BST from Preorder",
            ],
          },
          {
            category: "Height Related",
            problems: [
              "Maximum Depth of BT",
              "Balanced Binary Tree",
              "Diameter of Binary Tree",
              "Minimum Depth of BT",
            ],
          },
          {
            category: "Root to Leaf Path",
            problems: [
              "Binary Tree Paths",
              "Path Sum II",
              "Sum Root to Leaf numbers",
              "Smallest string starting from Leaf",
              "Insufficient nodes in root to Leaf",
              "Pseudo-Palindromic Paths in a Binary Tree",
              "Binary Tree Maximum Path Sum",
            ],
          },
          {
            category: "Ancestor Problems",
            problems: [
              "LCA of Binary Tree",
              "Maximum difference between node and ancestor",
              "LCA of deepest leaves",
              "Kth Ancestor of a Tree Node",
            ],
          },
          {
            category: "Binary Search Tree",
            problems: [
              "Validate BST",
              "Range Sum of BST",
              "Minimum Absolute Difference in BST",
              "Insert into a BST",
              "LCA of BST",
            ],
          },
        ],
        code: {
          javascript: `// DFS - Recursive
function dfs(root) {
    if (!root) return 0;
    
    let ans = 0;
    // Do logic
    dfs(root.left);
    dfs(root.right);
    
    return ans;
}

// DFS - Iterative
function dfsIterative(root) {
    const stack = [root];
    let ans = 0;
    
    while (stack.length) {
        const node = stack.pop();
        // Do logic
        
        if (node.right) stack.push(node.right);
        if (node.left) stack.push(node.left);
    }
    
    return ans;
}

// BFS - Level Order
function bfs(root) {
    const queue = [root];
    let ans = 0;
    
    while (queue.length) {
        const levelSize = queue.length;
        
        for (let i = 0; i < levelSize; i++) {
            const node = queue.shift();
            // Do logic
            
            if (node.left) queue.push(node.left);
            if (node.right) queue.push(node.right);
        }
    }
    
    return ans;
}`,
          python: `def dfs(root):
    if not root:
        return 0
    
    ans = 0
    # Do logic
    dfs(root.left)
    dfs(root.right)
    
    return ans

def dfs_iterative(root):
    stack = [root]
    ans = 0
    
    while stack:
        node = stack.pop()
        # Do logic
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return ans

from collections import deque

def bfs(root):
    queue = deque([root])
    ans = 0
    
    while queue:
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            # Do logic
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return ans`,
          java: `public int dfs(TreeNode root) {
    if (root == null) {
        return 0;
    }
    
    int ans = 0;
    // do logic
    dfs(root.left);
    dfs(root.right);
    
    return ans;
}

public int dfsIterative(TreeNode root) {
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    int ans = 0;
    
    while (!stack.empty()) {
        TreeNode node = stack.pop();
        // do logic
        
        if (node.right != null) {
            stack.push(node.right);
        }
        if (node.left != null) {
            stack.push(node.left);
        }
    }
    
    return ans;
}

public int bfs(TreeNode root) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    int ans = 0;
    
    while (!queue.isEmpty()) {
        int currentLength = queue.size();
        
        for (int i = 0; i < currentLength; i++) {
            TreeNode node = queue.remove();
            // do logic
            
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
    }
    
    return ans;
}`,
          cpp: `int dfs(TreeNode* root) {
    if (!root) {
        return 0;
    }
    
    int ans = 0;
    // do logic
    dfs(root->left);
    dfs(root->right);
    
    return ans;
}

int dfsIterative(TreeNode* root) {
    stack<TreeNode*> stk;
    stk.push(root);
    int ans = 0;
    
    while (!stk.empty()) {
        TreeNode* node = stk.top();
        stk.pop();
        // do logic
        
        if (node->right) {
            stk.push(node->right);
        }
        if (node->left) {
            stk.push(node->left);
        }
    }
    
    return ans;
}

int bfs(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    int ans = 0;
    
    while (!q.empty()) {
        int currentLength = q.size();
        
        for (int i = 0; i < currentLength; i++) {
            TreeNode* node = q.front();
            q.pop();
            // do logic
            
            if (node->left) {
                q.push(node->left);
            }
            if (node->right) {
                q.push(node->right);
            }
        }
    }
    
    return ans;
}`,
        },
      },
      {
        title: "20. Dynamic Programming",
        description: "Break down problems into overlapping subproblems",
        categories: [
          {
            category: "0/1 Knapsack (Take/Not Take)",
            problems: [
              "House Robber II",
              "Target Sum",
              "Partition Equal Subset Sum",
              "Ones and Zeroes",
              "Last Stone Weight II",
            ],
          },
          {
            category: "Infinite Supply (Unbounded Knapsack)",
            problems: [
              "Coin Change",
              "Coin Change II",
              "Perfect Squares",
              "Minimum Cost For Tickets",
            ],
          },
          {
            category: "Longest Increasing Subsequence",
            problems: [
              "Longest Increasing Subsequence",
              "Largest Divisible Subset",
              "Maximum Length of Pair Chain",
              "Number of LIS",
              "Longest String Chain",
            ],
          },
          {
            category: "DP on Grids",
            problems: [
              "Unique Paths II",
              "Minimum Path Sum",
              "Triangle",
              "Minimum Falling Path Sum",
              "Maximal Square",
              "Cherry Pickup",
              "Dungeon Game",
            ],
          },
          {
            category: "DP on Strings",
            problems: [
              "Longest Common Subsequence",
              "Longest Palindromic Subsequence",
              "Palindromic Substrings",
              "Longest Palindromic Substring",
              "Edit Distance",
              "Minimum ASCII Delete Sum for Two Strings",
              "Distinct Subsequences",
              "Shortest Common Supersequence",
              "Wildcard Matching",
            ],
          },
          {
            category: "DP on Stocks",
            problems: [
              "Buy and Sell Stocks II",
              "Buy and Sell Stocks III",
              "Buy and Sell Stocks IV",
              "Buy and Sell Stocks with Cooldown",
              "Buy and Sell Stocks with Transaction Fee",
            ],
          },
        ],
        code: {
          javascript: `// Top-Down Memoization
function dp(state, memo = new Map()) {
    if (BASE_CASE) {
        return 0;
    }
    
    const key = JSON.stringify(state);
    if (memo.has(key)) {
        return memo.get(key);
    }
    
    const ans = RECURRENCE_RELATION(state);
    memo.set(key, ans);
    
    return ans;
}

// Bottom-Up Tabulation
function dpTabulation(n) {
    const dp = new Array(n + 1).fill(0);
    dp[0] = BASE_CASE;
    
    for (let i = 1; i <= n; i++) {
        dp[i] = RECURRENCE_RELATION(dp, i);
    }
    
    return dp[n];
}`,
          python: `def dp(state, memo={}):
    if BASE_CASE:
        return 0
    
    if state in memo:
        return memo[state]
    
    ans = RECURRENCE_RELATION(state)
    memo[state] = ans
    
    return ans

def dp_tabulation(n):
    dp = [0] * (n + 1)
    dp[0] = BASE_CASE
    
    for i in range(1, n + 1):
        dp[i] = RECURRENCE_RELATION(dp, i)
    
    return dp[n]`,
          java: `Map<String, Integer> memo = new HashMap<>();

public int dp(STATE state) {
    if (BASE_CASE) {
        return 0;
    }
    
    String key = state.toString();
    if (memo.containsKey(key)) {
        return memo.get(key);
    }
    
    int ans = RECURRENCE_RELATION(state);
    memo.put(key, ans);
    
    return ans;
}

public int dpTabulation(int n) {
    int[] dp = new int[n + 1];
    dp[0] = BASE_CASE;
    
    for (int i = 1; i <= n; i++) {
        dp[i] = RECURRENCE_RELATION(dp, i);
    }
    
    return dp[n];
}`,
          cpp: `unordered_map<string, int> memo;

int dp(STATE state) {
    if (BASE_CASE) {
        return 0;
    }
    
    string key = to_string(state);
    if (memo.find(key) != memo.end()) {
        return memo[key];
    }
    
    int ans = RECURRENCE_RELATION(state);
    memo[key] = ans;
    
    return ans;
}

int dpTabulation(int n) {
    vector<int> dp(n + 1, 0);
    dp[0] = BASE_CASE;
    
    for (int i = 1; i <= n; i++) {
        dp[i] = RECURRENCE_RELATION(dp, i);
    }
    
    return dp[n];
}`,
        },
      },
      {
        title: "21. Graph Algorithms",
        description: "Advanced graph traversal and topological sorting",
        categories: [
          {
            category: "Topological Sort",
            problems: [
              "Course Schedule",
              "Course Schedule II",
              "Strange Printer II",
              "Sequence Reconstruction",
              "Alien Dictionary",
            ],
          },
          {
            category: "Union-Find",
            problems: [
              "Number of Operations to Make Network Connected",
              "Redundant Connection",
              "Accounts Merge",
              "Satisfiability of Equality Equations",
            ],
          },
        ],
        code: {
          javascript: `// Union-Find (Disjoint Set)
class UnionFind {
    constructor(n) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = new Array(n).fill(1);
    }
    
    find(x) {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]); // Path compression
        }
        return this.parent[x];
    }
    
    union(x, y) {
        const rootX = this.find(x);
        const rootY = this.find(y);
        
        if (rootX === rootY) return false;
        
        // Union by rank
        if (this.rank[rootX] > this.rank[rootY]) {
            this.parent[rootY] = rootX;
        } else if (this.rank[rootX] < this.rank[rootY]) {
            this.parent[rootX] = rootY;
        } else {
            this.parent[rootY] = rootX;
            this.rank[rootX]++;
        }
        
        return true;
    }
}`,
          python: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True`,
          java: `class UnionFind {
    private int[] parent;
    private int[] rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 1;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    public boolean union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return false;
        
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        
        return true;
    }
}`,
          cpp: `class UnionFind {
    vector<int> parent;
    vector<int> rank;
    
public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return false;
        
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        
        return true;
    }
};`,
        },
      },
      {
        title: "22. Greedy Algorithms",
        description: "Make locally optimal choice at each step",
        useCases: [
          "Jump game problems",
          "Activity selection",
          "Fractional knapsack",
          "Huffman coding",
        ],
        code: {
          javascript: `// Jump Game II - Greedy
function jumpGame(nums) {
    let jumps = 0;
    let currentEnd = 0;
    let farthest = 0;
    
    for (let i = 0; i < nums.length - 1; i++) {
        farthest = Math.max(farthest, i + nums[i]);
        
        if (i === currentEnd) {
            jumps++;
            currentEnd = farthest;
        }
    }
    
    return jumps;
}`,
          python: `def jump_game(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps`,
          java: `public int jumpGame(int[] nums) {
    int jumps = 0;
    int currentEnd = 0;
    int farthest = 0;
    
    for (int i = 0; i < nums.length - 1; i++) {
        farthest = Math.max(farthest, i + nums[i]);
        
        if (i == currentEnd) {
            jumps++;
            currentEnd = farthest;
        }
    }
    
    return jumps;
}`,
          cpp: `int jumpGame(vector<int>& nums) {
    int jumps = 0;
    int currentEnd = 0;
    int farthest = 0;
    
    for (int i = 0; i < nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        
        if (i == currentEnd) {
            jumps++;
            currentEnd = farthest;
        }
    }
    
    return jumps;
}`,
        },
        problems: [
          "Jump Game II",
          "Gas Station",
          "Bag of Tokens",
          "Boats to Save People",
          "Wiggle Subsequence",
          "Car Pooling",
          "Candy",
        ],
      },
      {
        title: "23. Trie (Prefix Tree)",
        description: "Tree structure for efficient string operations",
        useCases: [
          "Autocomplete systems",
          "Spell checkers",
          "IP routing tables",
          "Word search problems",
        ],
        code: {
          javascript: `class TrieNode {
    constructor() {
        this.children = {};
        this.isEnd = false;
    }
}

class Trie {
    constructor() {
        this.root = new TrieNode();
    }
    
    insert(word) {
        let node = this.root;
        for (const char of word) {
            if (!node.children[char]) {
                node.children[char] = new TrieNode();
            }
            node = node.children[char];
        }
        node.isEnd = true;
    }
    
    search(word) {
        let node = this.root;
        for (const char of word) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return node.isEnd;
    }
    
    startsWith(prefix) {
        let node = this.root;
        for (const char of prefix) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return true;
    }
}`,
          python: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True`,
          java: `class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    boolean isEnd = false;
}

class Trie {
    TrieNode root = new TrieNode();
    
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (!node.children.containsKey(c)) {
                node.children.put(c, new TrieNode());
            }
            node = node.children.get(c);
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (!node.children.containsKey(c)) {
                return false;
            }
            node = node.children.get(c);
        }
        return node.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            if (!node.children.containsKey(c)) {
                return false;
            }
            node = node.children.get(c);
        }
        return true;
    }
}`,
          cpp: `struct TrieNode {
    unordered_map<char, TrieNode*> children;
    bool isEnd = false;
};

class Trie {
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return true;
    }
};`,
        },
        problems: [
          "Implement Trie",
          "Word Search II",
          "Design Add and Search Words Data Structure",
          "Replace Words",
        ],
      },
      {
        title: "24. Dijkstra's Algorithm",
        description: "Shortest path in weighted graphs",
        useCases: [
          "Shortest path from source to all nodes",
          "Network routing protocols",
          "GPS navigation systems",
          "Flight routing",
        ],
        code: {
          javascript: `function dijkstra(graph, source, n) {
    const distances = new Array(n).fill(Infinity);
    distances[source] = 0;
    
    const heap = new MinHeap();
    heap.push([0, source]); // [distance, node]
    
    while (heap.size() > 0) {
        const [currDist, node] = heap.pop();
        
        if (currDist > distances[node]) {
            continue;
        }
        
        for (const [neighbor, weight] of graph[node] || []) {
            const dist = currDist + weight;
            
            if (dist < distances[neighbor]) {
                distances[neighbor] = dist;
                heap.push([dist, neighbor]);
            }
        }
    }
    
    return distances;
}`,
          python: `import heapq

def dijkstra(graph, source, n):
    distances = [float('inf')] * n
    distances[source] = 0
    
    heap = [(0, source)]
    
    while heap:
        curr_dist, node = heapq.heappop(heap)
        
        if curr_dist > distances[node]:
            continue
        
        for neighbor, weight in graph.get(node, []):
            dist = curr_dist + weight
            
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(heap, (dist, neighbor))
    
    return distances`,
          java: `public int[] dijkstra(Map<Integer, List<int[]>> graph, int source, int n) {
    int[] distances = new int[n];
    Arrays.fill(distances, Integer.MAX_VALUE);
    distances[source] = 0;
    
    PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
    heap.add(new int[]{0, source});
    
    while (!heap.isEmpty()) {
        int[] curr = heap.remove();
        int currDist = curr[0];
        int node = curr[1];
        
        if (currDist > distances[node]) {
            continue;
        }
        
        for (int[] edge : graph.getOrDefault(node, new ArrayList<>())) {
            int neighbor = edge[0];
            int weight = edge[1];
            int dist = currDist + weight;
            
            if (dist < distances[neighbor]) {
                distances[neighbor] = dist;
                heap.add(new int[]{dist, neighbor});
            }
        }
    }
    
    return distances;
}`,
          cpp: `vector<int> dijkstra(unordered_map<int, vector<pair<int, int>>>& graph, int source, int n) {
    vector<int> distances(n, INT_MAX);
    distances[source] = 0;
    
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> heap;
    heap.push({0, source});
    
    while (!heap.empty()) {
        auto [currDist, node] = heap.top();
        heap.pop();
        
        if (currDist > distances[node]) {
            continue;
        }
        
        for (auto [neighbor, weight] : graph[node]) {
            int dist = currDist + weight;
            
            if (dist < distances[neighbor]) {
                distances[neighbor] = dist;
                heap.push({dist, neighbor});
            }
        }
    }
    
    return distances;
}`,
        },
        problems: [
          "Network Delay Time",
          "Cheapest Flights Within K Stops",
          "Path with Maximum Minimum Value",
          "Swim in Rising Water",
        ],
      },
      {
        title: "25. Design Data Structures",
        description:
          "Implement custom data structures with specific operations",
        useCases: [
          "LRU/LFU Cache implementation",
          "Twitter feed design",
          "Browser history",
          "Snapshot arrays",
        ],
        problems: [
          "Design Twitter",
          "Design Browser History",
          "Design Circular Deque",
          "Snapshot Array",
          "LRU Cache",
          "LFU Cache",
        ],
      },
    ],
  },

  codeTemplates: {
    title: "Code Templates",
    icon: "FileCode",
    templates: [
      {
        title: "Two Pointers: One Input, Opposite Ends",
        description: "Start from both ends and move towards each other",
        code: {
          javascript: `function twoPointers(arr) {
    let left = 0;
    let right = arr.length - 1;
    let ans = 0;
    
    while (left < right) {
        // Do some logic here with left and right
        if (CONDITION) {
            left++;
        } else {
            right--;
        }
    }
    
    return ans;
}`,
          python: `def two_pointers(arr):
    left = 0
    right = len(arr) - 1
    ans = 0
    
    while left < right:
        # Do some logic here with left and right
        if CONDITION:
            left += 1
        else:
            right -= 1
    
    return ans`,
          java: `public int twoPointers(int[] arr) {
    int left = 0;
    int right = arr.length - 1;
    int ans = 0;
    
    while (left < right) {
        // Do some logic here with left and right
        if (CONDITION) {
            left++;
        } else {
            right--;
        }
    }
    
    return ans;
}`,
          cpp: `int twoPointers(vector<int>& arr) {
    int left = 0;
    int right = arr.size() - 1;
    int ans = 0;
    
    while (left < right) {
        // Do some logic here with left and right
        if (CONDITION) {
            left++;
        } else {
            right--;
        }
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Two Pointers: Two Inputs, Exhaust Both",
        description: "Merge or compare two sorted arrays/lists",
        code: {
          javascript: `function twoPointers(arr1, arr2) {
    let i = 0, j = 0;
    let ans = 0;
    
    while (i < arr1.length && j < arr2.length) {
        // Do some logic here
        if (CONDITION) {
            i++;
        } else {
            j++;
        }
    }
    
    while (i < arr1.length) {
        // Do logic for remaining elements in arr1
        i++;
    }
    
    while (j < arr2.length) {
        // Do logic for remaining elements in arr2
        j++;
    }
    
    return ans;
}`,
          python: `def two_pointers(arr1, arr2):
    i = j = 0
    ans = 0
    
    while i < len(arr1) and j < len(arr2):
        # Do some logic here
        if CONDITION:
            i += 1
        else:
            j += 1
    
    while i < len(arr1):
        # Do logic for remaining elements in arr1
        i += 1
    
    while j < len(arr2):
        # Do logic for remaining elements in arr2
        j += 1
    
    return ans`,
          java: `public int twoPointers(int[] arr1, int[] arr2) {
    int i = 0, j = 0;
    int ans = 0;
    
    while (i < arr1.length && j < arr2.length) {
        // Do some logic here
        if (CONDITION) {
            i++;
        } else {
            j++;
        }
    }
    
    while (i < arr1.length) {
        // Do logic for remaining elements in arr1
        i++;
    }
    
    while (j < arr2.length) {
        // Do logic for remaining elements in arr2
        j++;
    }
    
    return ans;
}`,
          cpp: `int twoPointers(vector<int>& arr1, vector<int>& arr2) {
    int i = 0, j = 0;
    int ans = 0;
    
    while (i < arr1.size() && j < arr2.size()) {
        // Do some logic here
        if (CONDITION) {
            i++;
        } else {
            j++;
        }
    }
    
    while (i < arr1.size()) {
        // Do logic for remaining elements in arr1
        i++;
    }
    
    while (j < arr2.size()) {
        // Do logic for remaining elements in arr2
        j++;
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Sliding Window",
        description: "Fixed or dynamic window that slides through array",
        code: {
          javascript: `function slidingWindow(arr) {
    let left = 0, ans = 0, curr = 0;
    
    for (let right = 0; right < arr.length; right++) {
        // Add arr[right] to curr
        
        while (WINDOW_CONDITION_BROKEN) {
            // Remove arr[left] from curr
            left++;
        }
        
        // Update ans
    }
    
    return ans;
}`,
          python: `def sliding_window(arr):
    left = ans = curr = 0
    
    for right in range(len(arr)):
        # Add arr[right] to curr
        
        while WINDOW_CONDITION_BROKEN:
            # Remove arr[left] from curr
            left += 1
        
        # Update ans
    
    return ans`,
          java: `public int slidingWindow(int[] arr) {
    int left = 0, ans = 0, curr = 0;
    
    for (int right = 0; right < arr.length; right++) {
        // Add arr[right] to curr
        
        while (WINDOW_CONDITION_BROKEN) {
            // Remove arr[left] from curr
            left++;
        }
        
        // Update ans
    }
    
    return ans;
}`,
          cpp: `int slidingWindow(vector<int>& arr) {
    int left = 0, ans = 0, curr = 0;
    
    for (int right = 0; right < arr.size(); right++) {
        // Add arr[right] to curr
        
        while (WINDOW_CONDITION_BROKEN) {
            // Remove arr[left] from curr
            left++;
        }
        
        // Update ans
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Build a Prefix Sum",
        description: "Precompute cumulative sums for range queries",
        code: {
          javascript: `function buildPrefixSum(arr) {
    let prefix = [arr[0]];
    
    for (let i = 1; i < arr.length; i++) {
        prefix.push(arr[i] + prefix[prefix.length - 1]);
    }
    
    return prefix;
}`,
          python: `def build_prefix_sum(arr):
    prefix = [arr[0]]
    
    for i in range(1, len(arr)):
        prefix.append(arr[i] + prefix[-1])
    
    return prefix`,
          java: `public int[] buildPrefixSum(int[] arr) {
    int[] prefix = new int[arr.length];
    prefix[0] = arr[0];
    
    for (int i = 1; i < arr.length; i++) {
        prefix[i] = arr[i] + prefix[i - 1];
    }
    
    return prefix;
}`,
          cpp: `vector<int> buildPrefixSum(vector<int>& arr) {
    vector<int> prefix(arr.size());
    prefix[0] = arr[0];
    
    for (int i = 1; i < arr.size(); i++) {
        prefix[i] = arr[i] + prefix[i - 1];
    }
    
    return prefix;
}`,
        },
      },
      {
        title: "Linked List: Fast and Slow Pointer",
        description: "Detect cycles or find middle element",
        code: {
          javascript: `function fastSlowPointer(head) {
    let slow = head;
    let fast = head;
    let ans = 0;
    
    while (fast && fast.next) {
        // Do logic
        slow = slow.next;
        fast = fast.next.next;
    }
    
    return ans;
}`,
          python: `def fast_slow_pointer(head):
    slow = head
    fast = head
    ans = 0
    
    while fast and fast.next:
        # Do logic
        slow = slow.next
        fast = fast.next.next
    
    return ans`,
          java: `public int fastSlowPointer(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
    int ans = 0;
    
    while (fast != null && fast.next != null) {
        // Do logic
        slow = slow.next;
        fast = fast.next.next;
    }
    
    return ans;
}`,
          cpp: `int fastSlowPointer(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    int ans = 0;
    
    while (fast && fast->next) {
        // Do logic
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Reversing a Linked List",
        description: "Reverse pointers in-place",
        code: {
          javascript: `function reverseLinkedList(head) {
    let prev = null;
    let curr = head;
    
    while (curr) {
        let nextNode = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextNode;
    }
    
    return prev;
}`,
          python: `def reverse_linked_list(head):
    prev = None
    curr = head
    
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    return prev`,
          java: `public ListNode reverseLinkedList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    
    while (curr != null) {
        ListNode nextNode = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextNode;
    }
    
    return prev;
}`,
          cpp: `ListNode* reverseLinkedList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr) {
        ListNode* nextNode = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nextNode;
    }
    
    return prev;
}`,
        },
      },
      {
        title: "Find Number of Subarrays that Fit Exact Criteria",
        description: "Count subarrays using HashMap",
        code: {
          javascript: `function numSubarrays(arr, k) {
    const counts = new Map();
    counts.set(0, 1);
    let ans = 0, curr = 0;
    
    for (const num of arr) {
        // Do logic to change curr
        ans += counts.get(curr - k) || 0;
        counts.set(curr, (counts.get(curr) || 0) + 1);
    }
    
    return ans;
}`,
          python: `def num_subarrays(arr, k):
    counts = {0: 1}
    ans = curr = 0
    
    for num in arr:
        # Do logic to change curr
        ans += counts.get(curr - k, 0)
        counts[curr] = counts.get(curr, 0) + 1
    
    return ans`,
          java: `public int numSubarrays(int[] arr, int k) {
    Map<Integer, Integer> counts = new HashMap<>();
    counts.put(0, 1);
    int ans = 0, curr = 0;
    
    for (int num : arr) {
        // Do logic to change curr
        ans += counts.getOrDefault(curr - k, 0);
        counts.put(curr, counts.getOrDefault(curr, 0) + 1);
    }
    
    return ans;
}`,
          cpp: `int numSubarrays(vector<int>& arr, int k) {
    unordered_map<int, int> counts;
    counts[0] = 1;
    int ans = 0, curr = 0;
    
    for (int num : arr) {
        // Do logic to change curr
        ans += counts[curr - k];
        counts[curr]++;
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Monotonic Increasing Stack",
        description: "Maintain elements in increasing order",
        code: {
          javascript: `function monotonicStack(arr) {
    const stack = [];
    let ans = 0;
    
    for (const num of arr) {
        // For monotonic decreasing, just flip the > to <
        while (stack.length && stack[stack.length - 1] > num) {
            // Do logic
            stack.pop();
        }
        stack.push(num);
    }
    
    return ans;
}`,
          python: `def monotonic_stack(arr):
    stack = []
    ans = 0
    
    for num in arr:
        # For monotonic decreasing, just flip the > to <
        while stack and stack[-1] > num:
            # Do logic
            stack.pop()
        stack.append(num)
    
    return ans`,
          java: `public int monotonicStack(int[] arr) {
    Stack<Integer> stack = new Stack<>();
    int ans = 0;
    
    for (int num : arr) {
        // For monotonic decreasing, just flip the > to <
        while (!stack.isEmpty() && stack.peek() > num) {
            // Do logic
            stack.pop();
        }
        stack.push(num);
    }
    
    return ans;
}`,
          cpp: `int monotonicStack(vector<int>& arr) {
    stack<int> stk;
    int ans = 0;
    
    for (int num : arr) {
        // For monotonic decreasing, just flip the > to <
        while (!stk.empty() && stk.top() > num) {
            // Do logic
            stk.pop();
        }
        stk.push(num);
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Binary Tree: DFS (Recursive)",
        description: "Recursive tree traversal",
        code: {
          javascript: `function dfs(root) {
    if (!root) {
        return;
    }
    
    let ans = 0;
    
    // Do logic
    dfs(root.left);
    dfs(root.right);
    return ans;
}`,
          python: `def dfs(root):
    if not root:
        return
    
    ans = 0
    
    # Do logic
    dfs(root.left)
    dfs(root.right)
    return ans`,
          java: `public int dfs(TreeNode root) {
    if (root == null) {
        return 0;
    }
    
    int ans = 0;
    
    // Do logic
    dfs(root.left);
    dfs(root.right);
    return ans;
}`,
          cpp: `int dfs(TreeNode* root) {
    if (!root) {
        return 0;
    }
    
    int ans = 0;
    
    // Do logic
    dfs(root->left);
    dfs(root->right);
    return ans;
}`,
        },
      },
      {
        title: "Binary Tree: DFS (Iterative)",
        description: "Iterative tree traversal using stack",
        code: {
          javascript: `function dfs(root) {
    const stack = [root];
    let ans = 0;
    
    while (stack.length) {
        const node = stack.pop();
        // Do logic
        if (node.left) {
            stack.push(node.left);
        }
        if (node.right) {
            stack.push(node.right);
        }
    }
    
    return ans;
}`,
          python: `def dfs(root):
    stack = [root]
    ans = 0
    
    while stack:
        node = stack.pop()
        # Do logic
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    return ans`,
          java: `public int dfs(TreeNode root) {
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    int ans = 0;
    
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        // Do logic
        if (node.left != null) {
            stack.push(node.left);
        }
        if (node.right != null) {
            stack.push(node.right);
        }
    }
    
    return ans;
}`,
          cpp: `int dfs(TreeNode* root) {
    stack<TreeNode*> stk;
    stk.push(root);
    int ans = 0;
    
    while (!stk.empty()) {
        TreeNode* node = stk.top();
        stk.pop();
        // Do logic
        if (node->left) {
            stk.push(node->left);
        }
        if (node->right) {
            stk.push(node->right);
        }
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Binary Tree: BFS",
        description: "Level-order traversal using queue",
        code: {
          javascript: `function bfs(root) {
    const queue = [root];
    let ans = 0;
    
    while (queue.length) {
        const currentLength = queue.length;
        // Do logic for current level
        
        for (let i = 0; i < currentLength; i++) {
            const node = queue.shift();
            // Do logic
            if (node.left) {
                queue.push(node.left);
            }
            if (node.right) {
                queue.push(node.right);
            }
        }
    }
    
    return ans;
}`,
          python: `from collections import deque

def bfs(root):
    queue = deque([root])
    ans = 0
    
    while queue:
        current_length = len(queue)
        # Do logic for current level
        
        for _ in range(current_length):
            node = queue.popleft()
            # Do logic
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return ans`,
          java: `public int bfs(TreeNode root) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    int ans = 0;
    
    while (!queue.isEmpty()) {
        int currentLength = queue.size();
        // Do logic for current level
        
        for (int i = 0; i < currentLength; i++) {
            TreeNode node = queue.remove();
            // Do logic
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
    }
    
    return ans;
}`,
          cpp: `int bfs(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    int ans = 0;
    
    while (!q.empty()) {
        int currentLength = q.size();
        // Do logic for current level
        
        for (int i = 0; i < currentLength; i++) {
            TreeNode* node = q.front();
            q.pop();
            // Do logic
            if (node->left) {
                q.push(node->left);
            }
            if (node->right) {
                q.push(node->right);
            }
        }
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Graph: DFS (Recursive)",
        description: "Recursive graph traversal",
        code: {
          javascript: `function dfs(graph, node, seen) {
    let ans = 0;
    
    for (const neighbor of graph[node]) {
        if (!seen.has(neighbor)) {
            seen.add(neighbor);
            ans += dfs(graph, neighbor, seen);
        }
    }
    
    return ans;
}

// Main function
function solve(graph) {
    const seen = new Set([START_NODE]);
    return dfs(graph, START_NODE, seen);
}`,
          python: `def dfs(graph, node, seen):
    ans = 0
    
    for neighbor in graph[node]:
        if neighbor not in seen:
            seen.add(neighbor)
            ans += dfs(graph, neighbor, seen)
    
    return ans

# Main function
def solve(graph):
    seen = {START_NODE}
    return dfs(graph, START_NODE, seen)`,
          java: `public int dfs(Map<Integer, List<Integer>> graph, int node, Set<Integer> seen) {
    int ans = 0;
    
    for (int neighbor : graph.get(node)) {
        if (!seen.contains(neighbor)) {
            seen.add(neighbor);
            ans += dfs(graph, neighbor, seen);
        }
    }
    
    return ans;
}

// Main function
public int solve(Map<Integer, List<Integer>> graph) {
    Set<Integer> seen = new HashSet<>();
    seen.add(START_NODE);
    return dfs(graph, START_NODE, seen);
}`,
          cpp: `int dfs(unordered_map<int, vector<int>>& graph, int node, unordered_set<int>& seen) {
    int ans = 0;
    
    for (int neighbor : graph[node]) {
        if (seen.find(neighbor) == seen.end()) {
            seen.insert(neighbor);
            ans += dfs(graph, neighbor, seen);
        }
    }
    
    return ans;
}

// Main function
int solve(unordered_map<int, vector<int>>& graph) {
    unordered_set<int> seen = {START_NODE};
    return dfs(graph, START_NODE, seen);
}`,
        },
      },
      {
        title: "Graph: DFS (Iterative)",
        description: "Iterative graph traversal using stack",
        code: {
          javascript: `function dfs(graph) {
    const stack = [START_NODE];
    const seen = new Set([START_NODE]);
    let ans = 0;
    
    while (stack.length) {
        const node = stack.pop();
        // Do some logic
        for (const neighbor of graph[node]) {
            if (!seen.has(neighbor)) {
                seen.add(neighbor);
                stack.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
          python: `def dfs(graph):
    stack = [START_NODE]
    seen = {START_NODE}
    ans = 0
    
    while stack:
        node = stack.pop()
        # Do some logic
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    
    return ans`,
          java: `public int dfs(Map<Integer, List<Integer>> graph) {
    Stack<Integer> stack = new Stack<>();
    stack.push(START_NODE);
    Set<Integer> seen = new HashSet<>();
    seen.add(START_NODE);
    int ans = 0;
    
    while (!stack.isEmpty()) {
        int node = stack.pop();
        // Do some logic
        for (int neighbor : graph.get(node)) {
            if (!seen.contains(neighbor)) {
                seen.add(neighbor);
                stack.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
          cpp: `int dfs(unordered_map<int, vector<int>>& graph) {
    stack<int> stk;
    stk.push(START_NODE);
    unordered_set<int> seen = {START_NODE};
    int ans = 0;
    
    while (!stk.empty()) {
        int node = stk.top();
        stk.pop();
        // Do some logic
        for (int neighbor : graph[node]) {
            if (seen.find(neighbor) == seen.end()) {
                seen.insert(neighbor);
                stk.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Graph: BFS",
        description: "Level-order graph traversal",
        code: {
          javascript: `function bfs(graph) {
    const queue = [START_NODE];
    const seen = new Set([START_NODE]);
    let ans = 0;
    
    while (queue.length) {
        const node = queue.shift();
        // Do some logic
        for (const neighbor of graph[node]) {
            if (!seen.has(neighbor)) {
                seen.add(neighbor);
                queue.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
          python: `from collections import deque

def bfs(graph):
    queue = deque([START_NODE])
    seen = {START_NODE}
    ans = 0
    
    while queue:
        node = queue.popleft()
        # Do some logic
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    
    return ans`,
          java: `public int bfs(Map<Integer, List<Integer>> graph) {
    Queue<Integer> queue = new LinkedList<>();
    queue.add(START_NODE);
    Set<Integer> seen = new HashSet<>();
    seen.add(START_NODE);
    int ans = 0;
    
    while (!queue.isEmpty()) {
        int node = queue.remove();
        // Do some logic
        for (int neighbor : graph.get(node)) {
            if (!seen.contains(neighbor)) {
                seen.add(neighbor);
                queue.add(neighbor);
            }
        }
    }
    
    return ans;
}`,
          cpp: `int bfs(unordered_map<int, vector<int>>& graph) {
    queue<int> q;
    q.push(START_NODE);
    unordered_set<int> seen = {START_NODE};
    int ans = 0;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        // Do some logic
        for (int neighbor : graph[node]) {
            if (seen.find(neighbor) == seen.end()) {
                seen.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Find Top K Elements with Heap",
        description: "Use min/max heap to track top K elements",
        code: {
          javascript: `function topK(arr, k) {
    const heap = new MinHeap();
    
    for (const num of arr) {
        heap.push(num);
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    return heap.toArray();
}`,
          python: `import heapq

def top_k(arr, k):
    heap = []
    
    for num in arr:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap`,
          java: `public int[] topK(int[] arr, int k) {
    PriorityQueue<Integer> heap = new PriorityQueue<>();
    
    for (int num : arr) {
        heap.add(num);
        if (heap.size() > k) {
            heap.remove();
        }
    }
    
    int[] ans = new int[k];
    for (int i = 0; i < k; i++) {
        ans[i] = heap.remove();
    }
    return ans;
}`,
          cpp: `vector<int> topK(vector<int>& arr, int k) {
    priority_queue<int, vector<int>, greater<int>> heap;
    
    for (int num : arr) {
        heap.push(num);
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    vector<int> ans;
    while (!heap.empty()) {
        ans.push_back(heap.top());
        heap.pop();
    }
    return ans;
}`,
        },
      },
      {
        title: "Binary Search",
        description: "Standard binary search on sorted array",
        code: {
          javascript: `function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) {
            // Do something
            return mid;
        }
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return -1;
}`,
          python: `def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            # Do something
            return mid
        if arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1`,
          java: `public int binarySearch(int[] arr, int target) {
    int left = 0;
    int right = arr.length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            // Do something
            return mid;
        }
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return -1;
}`,
          cpp: `int binarySearch(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            // Do something
            return mid;
        }
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return -1;
}`,
        },
      },
      {
        title: "Binary Search: Left-Most Insertion Point",
        description: "Find first position where element can be inserted",
        code: {
          javascript: `function binarySearchLeftmost(arr, target) {
    let left = 0;
    let right = arr.length;
    
    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] >= target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}`,
          python: `def binary_search_leftmost(arr, target):
    left = 0
    right = len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            right = mid
        else:
            left = mid + 1
    
    return left`,
          java: `public int binarySearchLeftmost(int[] arr, int target) {
    int left = 0;
    int right = arr.length;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] >= target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}`,
          cpp: `int binarySearchLeftmost(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size();
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] >= target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}`,
        },
      },
      {
        title: "Binary Search: Right-Most Insertion Point",
        description: "Find last position where element can be inserted",
        code: {
          javascript: `function binarySearchRightmost(arr, target) {
    let left = 0;
    let right = arr.length;
    
    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] > target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}`,
          python: `def binary_search_rightmost(arr, target):
    left = 0
    right = len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > target:
            right = mid
        else:
            left = mid + 1
    
    return left`,
          java: `public int binarySearchRightmost(int[] arr, int target) {
    int left = 0;
    int right = arr.length;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] > target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}`,
          cpp: `int binarySearchRightmost(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size();
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] > target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}`,
        },
      },
      {
        title: "Binary Search: Greedy (Minimum)",
        description: "Find minimum value that satisfies condition",
        code: {
          javascript: `function binarySearchGreedy(arr) {
    let left = MINIMUM_POSSIBLE_ANSWER;
    let right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (check(mid)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

function check(x) {
    // Return true if x is feasible
    return false;
}`,
          python: `def binary_search_greedy(arr):
    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            right = mid - 1
        else:
            left = mid + 1
    
    return left

def check(x):
    # Return true if x is feasible
    return False`,
          java: `public int binarySearchGreedy(int[] arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (check(mid)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

private boolean check(int x) {
    // Return true if x is feasible
    return false;
}`,
          cpp: `int binarySearchGreedy(vector<int>& arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (check(mid)) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}

bool check(int x) {
    // Return true if x is feasible
    return false;
}`,
        },
      },
      {
        title: "Binary Search: Greedy (Maximum)",
        description: "Find maximum value that satisfies condition",
        code: {
          javascript: `function binarySearchGreedy(arr) {
    let left = MINIMUM_POSSIBLE_ANSWER;
    let right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (check(mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return right;
}

function check(x) {
    // Return true if x is feasible
    return false;
}`,
          python: `def binary_search_greedy(arr):
    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return right

def check(x):
    # Return true if x is feasible
    return False`,
          java: `public int binarySearchGreedy(int[] arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (check(mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return right;
}

private boolean check(int x) {
    // Return true if x is feasible
    return false;
}`,
          cpp: `int binarySearchGreedy(vector<int>& arr) {
    int left = MINIMUM_POSSIBLE_ANSWER;
    int right = MAXIMUM_POSSIBLE_ANSWER;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (check(mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return right;
}

bool check(int x) {
    // Return true if x is feasible
    return false;
}`,
        },
      },
      {
        title: "Backtracking",
        description: "Generate all combinations/permutations",
        code: {
          javascript: `function backtrack(curr, OTHER_ARGUMENTS...) {
    if (BASE_CASE) {
        // Modify the answer
        return;
    }
    
    let ans = 0;
    for (const ITERATE_OVER_INPUT) {
        // Modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS...);
        // Undo the modification of the current state
    }
    
    return ans;
}`,
          python: `def backtrack(curr, OTHER_ARGUMENTS):
    if BASE_CASE:
        # Modify the answer
        return
    
    ans = 0
    for ITERATE_OVER_INPUT:
        # Modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS)
        # Undo the modification of the current state
    
    return ans`,
          java: `public int backtrack(List<Integer> curr, OTHER_ARGUMENTS) {
    if (BASE_CASE) {
        // Modify the answer
        return 0;
    }
    
    int ans = 0;
    for (ITERATE_OVER_INPUT) {
        // Modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS);
        // Undo the modification of the current state
    }
    
    return ans;
}`,
          cpp: `int backtrack(vector<int>& curr, OTHER_ARGUMENTS) {
    if (BASE_CASE) {
        // Modify the answer
        return 0;
    }
    
    int ans = 0;
    for (ITERATE_OVER_INPUT) {
        // Modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS);
        // Undo the modification of the current state
    }
    
    return ans;
}`,
        },
      },
      {
        title: "Dynamic Programming: Top-Down Memoization",
        description: "Recursive DP with caching",
        code: {
          javascript: `function dp(STATE, memo = new Map()) {
    if (BASE_CASE) {
        return 0;
    }
    
    const key = JSON.stringify(STATE);
    if (memo.has(key)) {
        return memo.get(key);
    }
    
    let ans = RECURRENCE_RELATION(STATE);
    memo.set(key, ans);
    return ans;
}`,
          python: `def dp(STATE, memo={}):
    if BASE_CASE:
        return 0
    
    if STATE in memo:
        return memo[STATE]
    
    ans = RECURRENCE_RELATION(STATE)
    memo[STATE] = ans
    return ans`,
          java: `private Map<String, Integer> memo = new HashMap<>();

public int dp(STATE) {
    if (BASE_CASE) {
        return 0;
    }
    
    String key = STATE.toString();
    if (memo.containsKey(key)) {
        return memo.get(key);
    }
    
    int ans = RECURRENCE_RELATION(STATE);
    memo.put(key, ans);
    return ans;
}`,
          cpp: `unordered_map<string, int> memo;

int dp(STATE) {
    if (BASE_CASE) {
        return 0;
    }
    
    string key = to_string(STATE);
    if (memo.find(key) != memo.end()) {
        return memo[key];
    }
    
    int ans = RECURRENCE_RELATION(STATE);
    memo[key] = ans;
    return ans;
}`,
        },
      },
      {
        title: "Build a Trie",
        description: "Prefix tree for string operations",
        code: {
          javascript: `class TrieNode {
    constructor() {
        this.children = {};
        this.isEnd = false;
    }
}

class Trie {
    constructor() {
        this.root = new TrieNode();
    }
    
    insert(word) {
        let node = this.root;
        for (const char of word) {
            if (!node.children[char]) {
                node.children[char] = new TrieNode();
            }
            node = node.children[char];
        }
        node.isEnd = true;
    }
    
    search(word) {
        let node = this.root;
        for (const char of word) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return node.isEnd;
    }
}`,
          python: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end`,
          java: `class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    boolean isEnd = false;
}

class Trie {
    TrieNode root = new TrieNode();
    
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (!node.children.containsKey(c)) {
                node.children.put(c, new TrieNode());
            }
            node = node.children.get(c);
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (!node.children.containsKey(c)) {
                return false;
            }
            node = node.children.get(c);
        }
        return node.isEnd;
    }
}`,
          cpp: `struct TrieNode {
    unordered_map<char, TrieNode*> children;
    bool isEnd = false;
};

class Trie {
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return node->isEnd;
    }
};`,
        },
      },
      {
        title: "Dijkstra's Algorithm",
        description: "Shortest path in weighted graph",
        code: {
          javascript: `function dijkstra(graph, start) {
    const distances = new Map();
    const heap = new MinHeap();
    heap.push([0, start]);
    
    while (heap.size()) {
        const [currDist, node] = heap.pop();
        
        if (distances.has(node)) {
            continue;
        }
        
        distances.set(node, currDist);
        
        for (const [neighbor, weight] of graph[node]) {
            if (!distances.has(neighbor)) {
                heap.push([currDist + weight, neighbor]);
            }
        }
    }
    
    return distances;
}`,
          python: `import heapq

def dijkstra(graph, start):
    distances = {}
    heap = [(0, start)]
    
    while heap:
        curr_dist, node = heapq.heappop(heap)
        
        if node in distances:
            continue
        
        distances[node] = curr_dist
        
        for neighbor, weight in graph[node]:
            if neighbor not in distances:
                heapq.heappush(heap, (curr_dist + weight, neighbor))
    
    return distances`,
          java: `public Map<Integer, Integer> dijkstra(Map<Integer, List<int[]>> graph, int start) {
    Map<Integer, Integer> distances = new HashMap<>();
    PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
    heap.add(new int[]{0, start});
    
    while (!heap.isEmpty()) {
        int[] curr = heap.remove();
        int currDist = curr[0];
        int node = curr[1];
        
        if (distances.containsKey(node)) {
            continue;
        }
        
        distances.put(node, currDist);
        
        for (int[] edge : graph.get(node)) {
            int neighbor = edge[0];
            int weight = edge[1];
            if (!distances.containsKey(neighbor)) {
                heap.add(new int[]{currDist + weight, neighbor});
            }
        }
    }
    
    return distances;
}`,
          cpp: `unordered_map<int, int> dijkstra(unordered_map<int, vector<pair<int, int>>>& graph, int start) {
    unordered_map<int, int> distances;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> heap;
    heap.push({0, start});
    
    while (!heap.empty()) {
        auto [currDist, node] = heap.top();
        heap.pop();
        
        if (distances.find(node) != distances.end()) {
            continue;
        }
        
        distances[node] = currDist;
        
        for (auto [neighbor, weight] : graph[node]) {
            if (distances.find(neighbor) == distances.end()) {
                heap.push({currDist + weight, neighbor});
            }
        }
    }
    
    return distances;
}`,
        },
      },
    ],
  },

  tips: {
    title: "Stages of an interview & Tips",
    icon: "Brain",
    sections: [
      {
        title: "1. Introductions (First 2-3 minutes)",
        items: [
          "Prepare a 30-60 second introduction summarizing your education, work experience, and interests",
          "Smile and speak with a confident voice to make a strong first impression",
          "Pay attention when the interviewer talks about their work - use it for questions later",
          "Mention shared interests if the interviewer mentions any hobbies or projects you're passionate about",
          "Make eye contact and show enthusiasm about the opportunity",
        ],
      },
      {
        title: "2. Understanding the Problem (5-10 minutes)",
        items: [
          "Confirm understanding by paraphrasing the problem back to the interviewer",
          "Ask clarifying questions: Will input have only integers? Could it be empty? Sorted or unsorted?",
          "Ask about expected input size - it can hint at the solution (small n → backtracking, n~1000 → O(n²), large n → O(n) or better)",
          "Walk through example test cases to verify your understanding",
          "Clarify edge cases: What should happen with invalid input? Empty arrays? Negative numbers?",
          "Ask 'What should I optimize for - time or space?' if not specified",
        ],
      },
      {
        title: "3. Brainstorming Solution (10-15 minutes)",
        items: [
          "Think out loud! Explain your thought process to show problem-solving skills",
          "Identify patterns: Does this involve subarrays? (sliding window) Sorted data? (binary search)",
          "Break down what the problem needs and match it to known data structures/algorithms",
          "Discuss trade-offs between different approaches before committing to one",
          "Be receptive to hints - the interviewer knows the optimal solution and wants you to succeed",
          "Explain your algorithm's rough steps before coding and get interviewer buy-in",
          "If stuck, start with brute force and discuss how to optimize it",
        ],
      },
      {
        title: "4. Implementation (15-20 minutes)",
        items: [
          "Ask if you can use specific libraries (e.g., Python's collections, Java's PriorityQueue)",
          "Explain your decisions as you code (e.g., 'Using a set here to track visited nodes')",
          "Write clean code following language conventions (spacing, naming, indentation)",
          "Avoid duplicated code - use loops, helper functions, or direction arrays",
          "Use descriptive variable names (goodNames instead of x, y, z)",
          "Don't be afraid of helper functions - they make code modular and professional",
          "If stuck, communicate with your interviewer - struggling in silence is worse",
          "Consider implementing brute force first, then optimize the slow parts",
        ],
      },
      {
        title: "5. Testing & Debugging (5-10 minutes)",
        items: [
          "Test with variety: small inputs, large inputs, edge cases, invalid inputs",
          "If code runs: Add print statements to debug, trace variables through execution",
          "If code doesn't run: Walk through manually with small test case, track variable values",
          "Don't panic if there's a bug - explain your debugging process out loud",
          "Test edge cases: empty input, single element, all elements same, duplicates",
          "Verify your solution handles the examples given in the problem statement",
          "If manually testing, condense trivial parts to save time",
        ],
      },
      {
        title: "6. Analysis & Follow-ups (5 minutes)",
        items: [
          "Be ready to explain time and space complexity (worst-case and average-case)",
          "Justify your choice of data structures and algorithms",
          "Discuss whether the solution can be optimized further",
          "If asked 'Can this be improved?' - answer is usually yes unless you're certain it's optimal",
          "Don't be confidently wrong - it's ok to be uncertain about optimality",
          "Be prepared for follow-ups: new constraints, better space complexity, handling new edge cases",
        ],
      },
      {
        title: "7. Asking Questions (Last 5 minutes)",
        items: [
          "Prepare 3-5 thoughtful questions before the interview",
          "Ask about day-to-day work: 'What does an average day look like?'",
          "Show interest: 'Why did you join this company?' 'What's your favorite project?'",
          "Ask about growth: 'What kind of work can I expect?' 'Mentorship opportunities?'",
          "Read the company's tech blog and ask about their technical decisions",
          "Listen actively and ask follow-up questions to show engagement",
          "Stay interested and enthusiastic - first impressions matter, but so do last ones",
        ],
      },
      {
        title: "Before Coding - Problem-Solving Checklist",
        items: [
          "Restate the problem in your own words",
          "Identify inputs, outputs, and constraints",
          "Work through examples manually to find patterns",
          "Consider edge cases before proposing a solution",
          "Discuss time/space complexity goals before implementing",
          "Get explicit approval from interviewer before writing code",
        ],
      },
      {
        title: "During Coding - Best Practices",
        items: [
          "Start with a plan - outline your approach in comments first",
          "Use meaningful variable names that explain their purpose",
          "Add comments for complex logic, but don't over-comment obvious code",
          "Write modular code with helper functions for readability",
          "Handle edge cases explicitly in your code",
          "Explain your code as you write to demonstrate understanding",
        ],
      },
      {
        title: "Optimization Strategies",
        items: [
          "Can you use a hash table to trade space for O(1) lookups?",
          "Can you sort the input to enable binary search or two pointers?",
          "Can you use two pointers instead of nested loops?",
          "Is there redundant computation you can cache with memoization?",
          "Can you reduce space by reusing the input array or using constant space?",
          "Look for bottlenecks - which part of your code is slowest?",
          "Consider preprocessing the input for faster repeated operations",
        ],
      },
      {
        title: "Common Edge Cases to Consider",
        items: [
          "Empty input: [], '', null, undefined",
          "Single element: [1], 'a', single node",
          "Two elements: minimum for two pointers problems",
          "All elements identical: [5,5,5,5]",
          "Already sorted vs unsorted input",
          "Negative numbers, zero, very large numbers",
          "Integer overflow for products/sums of large numbers",
          "Duplicate elements when uniqueness is expected",
          "Cycles in linked lists or graphs",
          "Disconnected components in graphs",
        ],
      },
      {
        title: "Time Management (45-60 min interview)",
        items: [
          "Introductions: 2-3 minutes - be concise and confident",
          "Problem understanding: 5-10 minutes - clarify everything upfront",
          "Solution design: 10-15 minutes - think thoroughly before coding",
          "Implementation: 15-20 minutes - write clean, working code",
          "Testing: 5-10 minutes - catch bugs before interviewer does",
          "Discussion: 5 minutes - explain complexity and trade-offs",
          "Your questions: 5 minutes - show genuine interest in the role",
          "If running late, communicate! Ask: 'Should I implement this or discuss the approach?'",
        ],
      },
      {
        title: "Communication Tips",
        items: [
          "Think out loud - silence makes it hard for interviewer to help you",
          "Explain your reasoning: 'I'm using a HashMap because we need O(1) lookups'",
          "If stuck, say so: 'I'm considering X and Y, which direction should I explore?'",
          "Acknowledge hints: 'That's a great point, let me think about that approach'",
          "Admit when you don't know something rather than making up answers",
          "Ask for help when needed: 'Can I get a hint on this part?'",
          "Be enthusiastic and positive, even if the problem is challenging",
        ],
      },
      {
        title: "What Interviewers Are Looking For",
        items: [
          "Problem-solving ability - can you break down complex problems?",
          "Communication skills - can you explain your thought process clearly?",
          "Code quality - clean, readable, maintainable code",
          "Testing mindset - do you think about edge cases and validation?",
          "Learning ability - do you take feedback and adapt?",
          "Culture fit - would you be a good teammate?",
          "Technical knowledge - do you understand DSA fundamentals?",
          "Debugging skills - can you identify and fix issues systematically?",
        ],
      },
      {
        title: "Red Flags to Avoid",
        items: [
          "Starting to code immediately without understanding the problem",
          "Struggling in silence instead of thinking out loud",
          "Ignoring hints or suggestions from the interviewer",
          "Being defensive when your approach is questioned",
          "Writing messy, unreadable code with poor naming",
          "Not testing your code or only testing happy paths",
          "Claiming your solution is optimal when you're unsure",
          "Showing disinterest during the 'ask questions' phase",
          "Being overconfident or arrogant about your solution",
          "Not managing time - spending too long on one section",
        ],
      },
    ],
  },

  companyQuestions: {
    title: "Company-Wise Questions",
    icon: "Building2",
    companies: [
      {
        name: "Google",
        logo: Google,
        color: "#4285f4",
        totalQuestions: 500,
        frequency: "Very High",
        categories: [
          {
            category: "Array & String",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "Very High",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "Very High",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "High",
                leetcodeNum: 42,
                topics: [
                  "Array",
                  "Two Pointers",
                  "Dynamic Programming",
                  "Stack",
                  "Monotonic Stack",
                ],
                acceptance: "65.1%",
              },
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "High",
                leetcodeNum: 4,
                topics: ["Array", "Binary Search", "Divide and Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 14,
                name: "Longest Common Prefix",
                difficulty: "Easy",
                frequency: "Very High",
                leetcodeNum: 14,
                topics: ["String", "Trie"],
                acceptance: "45.5%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 20,
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 128,
                name: "Longest Consecutive Sequence",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 128,
                topics: ["Array", "Hash Table", "Union Find"],
                acceptance: "47.0%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 48,
                name: "Rotate Image",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 48,
                topics: ["Array", "Math", "Matrix"],
                acceptance: "77.9%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
              {
                id: 28,
                name: "Find the Index of the First Occurrence in a String",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 28,
                topics: ["Two Pointers", "String", "String Matching"],
                acceptance: "45.0%",
              },
              {
                id: 242,
                name: "Valid Anagram",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 242,
                topics: ["Hash Table", "String", "Sorting"],
                acceptance: "66.7%",
              },
              {
                id: 41,
                name: "First Missing Positive",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 41,
                topics: ["Array", "Hash Table"],
                acceptance: "41.1%",
              },
              {
                id: 68,
                name: "Text Justification",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 68,
                topics: ["Array", "String", "Simulation"],
                acceptance: "48.1%",
              },
              {
                id: 57,
                name: "Insert Interval",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 57,
                topics: ["Array"],
                acceptance: "43.5%",
              },
              {
                id: 152,
                name: "Maximum Product Subarray",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 152,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "34.9%",
              },
              {
                id: 8,
                name: "String to Integer (atoi)",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 8,
                topics: ["String"],
                acceptance: "19.2%",
              },
            ],
          },
          {
            category: "Linked List",
            questions: [
              {
                id: 2,
                name: "Add Two Numbers",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 2,
                topics: ["Linked List", "Math", "Recursion"],
                acceptance: "46.2%",
              },
              {
                id: 206,
                name: "Reverse Linked List",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 206,
                topics: ["Linked List", "Recursion"],
                acceptance: "79.2%",
              },
              {
                id: 138,
                name: "Copy List with Random Pointer",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 138,
                topics: ["Hash Table", "Linked List"],
                acceptance: "60.5%",
              },
              {
                id: 114,
                name: "Flatten Binary Tree to Linked List",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 114,
                topics: [
                  "Linked List",
                  "Stack",
                  "Tree",
                  "Depth-First Search",
                  "Binary Tree",
                ],
                acceptance: "68.5%",
              },
            ],
          },
          {
            category: "Tree & Graph",
            questions: [
              {
                id: 100,
                name: "Same Tree",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 100,
                topics: [
                  "Tree",
                  "Depth-First Search",
                  "Breadth-First Search",
                  "Binary Tree",
                ],
                acceptance: "65.1%",
              },
              {
                id: 101,
                name: "Symmetric Tree",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 101,
                topics: [
                  "Tree",
                  "Depth-First Search",
                  "Breadth-First Search",
                  "Binary Tree",
                ],
                acceptance: "59.3%",
              },
              {
                id: 230,
                name: "Kth Smallest Element in a BST",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 230,
                topics: [
                  "Tree",
                  "Depth-First Search",
                  "Binary Search Tree",
                  "Binary Tree",
                ],
                acceptance: "75.3%",
              },
              {
                id: 133,
                name: "Clone Graph",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 133,
                topics: [
                  "Hash Table",
                  "Depth-First Search",
                  "Breadth-First Search",
                  "Graph",
                ],
                acceptance: "62.4%",
              },
              {
                id: 212,
                name: "Word Search II",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 212,
                topics: ["Array", "String", "Backtracking", "Trie", "Matrix"],
                acceptance: "37.3%",
              },
              {
                id: 199,
                name: "Binary Tree Right Side View",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 199,
                topics: [
                  "Tree",
                  "Depth-First Search",
                  "Breadth-First Search",
                  "Binary Tree",
                ],
                acceptance: "67.0%",
              },
              {
                id: 261,
                name: "Graph Valid Tree",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 261,
                topics: [
                  "Depth-First Search",
                  "Breadth-First Search",
                  "Union Find",
                  "Graph",
                ],
                acceptance: "49.3%",
              },
              {
                id: 286,
                name: "Walls and Gates",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 286,
                topics: ["Array", "Breadth-First Search", "Matrix"],
                acceptance: "63.0%",
              },
              {
                id: 257,
                name: "Binary Tree Paths",
                difficulty: "Easy",
                frequency: "Low",
                leetcodeNum: 257,
                topics: [
                  "String",
                  "Backtracking",
                  "Tree",
                  "Depth-First Search",
                  "Binary Tree",
                ],
                acceptance: "66.6%",
              },
            ],
          },
          {
            category: "Dynamic Programming",
            questions: [
              {
                id: 55,
                name: "Jump Game",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 55,
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "39.5%",
              },
              {
                id: 45,
                name: "Jump Game II",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 45,
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "41.5%",
              },
              {
                id: 64,
                name: "Minimum Path Sum",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 64,
                topics: ["Array", "Dynamic Programming", "Matrix"],
                acceptance: "66.5%",
              },
              {
                id: 118,
                name: "Pascal's Triangle",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 118,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "77.0%",
              },
              {
                id: 85,
                name: "Maximal Rectangle",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 85,
                topics: [
                  "Array",
                  "Dynamic Programming",
                  "Stack",
                  "Matrix",
                  "Monotonic Stack",
                ],
                acceptance: "53.7%",
              },
              {
                id: 309,
                name: "Best Time to Buy and Sell Stock with Cooldown",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 309,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "60.4%",
              },
              {
                id: 188,
                name: "Best Time to Buy and Sell Stock IV",
                difficulty: "Hard",
                frequency: "Low",
                leetcodeNum: 188,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "47.1%",
              },
              {
                id: 256,
                name: "Paint House",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 256,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "63.7%",
              },
              {
                id: 276,
                name: "Paint Fence",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 276,
                topics: ["Dynamic Programming"],
                acceptance: "47.7%",
              },
            ],
          },
          {
            category: "Backtracking & Recursion",
            questions: [
              {
                id: 90,
                name: "Subsets II",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 90,
                topics: ["Array", "Backtracking", "Bit Manipulation"],
                acceptance: "59.5%",
              },
              {
                id: 77,
                name: "Combinations",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 77,
                topics: ["Backtracking"],
                acceptance: "72.9%",
              },
              {
                id: 51,
                name: "N-Queens II",
                difficulty: "Hard",
                frequency: "Low",
                leetcodeNum: 52,
                topics: ["Backtracking"],
                acceptance: "76.7%",
              },
              {
                id: 87,
                name: "Scramble String",
                difficulty: "Hard",
                frequency: "Low",
                leetcodeNum: 87,
                topics: ["String", "Dynamic Programming"],
                acceptance: "42.2%",
              },
            ],
          },
          {
            category: "Design & Implementation",
            questions: [
              {
                id: 295,
                name: "Find Median from Data Stream",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 295,
                topics: [
                  "Two Pointers",
                  "Design",
                  "Sorting",
                  "Heap (Priority Queue)",
                  "Data Stream",
                ],
                acceptance: "53.3%",
              },
              {
                id: 225,
                name: "Implement Stack using Queues",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 225,
                topics: ["Stack", "Design", "Queue"],
                acceptance: "67.3%",
              },
              {
                id: 173,
                name: "Peeking Iterator",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 284,
                topics: ["Array", "Design", "Iterator"],
                acceptance: "60.3%",
              },
              {
                id: 281,
                name: "Zigzag Iterator",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 281,
                topics: ["Array", "Design", "Queue", "Iterator"],
                acceptance: "65.8%",
              },
              {
                id: 251,
                name: "Flatten 2D Vector",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 251,
                topics: ["Array", "Two Pointers", "Design", "Iterator"],
                acceptance: "50.1%",
              },
            ],
          },
          {
            category: "Math & Bit Manipulation",
            questions: [
              {
                id: 136,
                name: "Single Number",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 136,
                topics: ["Array", "Bit Manipulation"],
                acceptance: "76.0%",
              },
              {
                id: 67,
                name: "Add Binary",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 67,
                topics: ["Math", "String", "Bit Manipulation", "Simulation"],
                acceptance: "55.7%",
              },
              {
                id: 12,
                name: "Integer to Roman",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 12,
                topics: ["Hash Table", "Math", "String"],
                acceptance: "68.6%",
              },
              {
                id: 258,
                name: "Missing Number",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 268,
                topics: [
                  "Array",
                  "Hash Table",
                  "Math",
                  "Binary Search",
                  "Bit Manipulation",
                  "Sorting",
                ],
                acceptance: "70.1%",
              },
              {
                id: 172,
                name: "Factorial Trailing Zeroes",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 172,
                topics: ["Math"],
                acceptance: "44.9%",
              },
              {
                id: 201,
                name: "Bitwise AND of Numbers Range",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 201,
                topics: ["Bit Manipulation"],
                acceptance: "47.7%",
              },
              {
                id: 89,
                name: "Gray Code",
                difficulty: "Medium",
                frequency: "Low",
                leetcodeNum: 89,
                topics: ["Math", "Backtracking", "Bit Manipulation"],
                acceptance: "61.9%",
              },
            ],
          },
        ],
      },
      {
        name: "Amazon",
        logo: Amazon,
        color: "#ff9900",
        totalQuestions: 600,
        frequency: "Extremely High",
        categories: [
          {
            category: "Array & String",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "Very High",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.7%",
              },
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "Very High",
                leetcodeNum: 42,
                topics: ["Array", "Two Pointers", "Stack"],
                acceptance: "65.1%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 88,
                name: "Merge Sorted Array",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 88,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "52.9%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String"],
                acceptance: "70.9%",
              },
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 15,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 56,
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 11,
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 54,
                name: "Spiral Matrix",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 54,
                topics: ["Array", "Matrix", "Simulation"],
                acceptance: "53.9%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 41,
                name: "First Missing Positive",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 41,
                topics: ["Array", "Hash Table"],
                acceptance: "41.1%",
              },
            ],
          },
          {
            category: "Linked List",
            questions: [
              {
                id: 2,
                name: "Add Two Numbers",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 2,
                topics: ["Linked List", "Math", "Recursion"],
                acceptance: "46.2%",
              },
              {
                id: 21,
                name: "Merge Two Sorted Lists",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 21,
                topics: ["Linked List", "Recursion"],
                acceptance: "66.8%",
              },
              {
                id: 23,
                name: "Merge k Sorted Lists",
                difficulty: "Hard",
                frequency: "High",
                leetcodeNum: 23,
                topics: ["Linked List", "Heap (Priority Queue)"],
                acceptance: "56.8%",
              },
              {
                id: 25,
                name: "Reverse Nodes in k-Group",
                difficulty: "Hard",
                frequency: "Medium",
                leetcodeNum: 25,
                topics: ["Linked List", "Recursion"],
                acceptance: "63.0%",
              },
              {
                id: 19,
                name: "Remove Nth Node From End of List",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 19,
                topics: ["Linked List", "Two Pointers"],
                acceptance: "49.0%",
              },
            ],
          },
          {
            category: "Tree & Graph",
            questions: [
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 98,
                topics: ["Tree", "Depth-First Search", "Binary Search Tree"],
                acceptance: "34.4%",
              },
              {
                id: 100,
                name: "Same Tree",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 100,
                topics: ["Tree", "DFS", "BFS"],
                acceptance: "65.1%",
              },
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 79,
                topics: ["Array", "Backtracking", "Matrix"],
                acceptance: "45.3%",
              },
              {
                id: 94,
                name: "Binary Tree Inorder Traversal",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 94,
                topics: ["Stack", "Tree", "DFS"],
                acceptance: "78.6%",
              },
            ],
          },
          {
            category: "Dynamic Programming",
            questions: [
              {
                id: 70,
                name: "Climbing Stairs",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 70,
                topics: ["Math", "Dynamic Programming"],
                acceptance: "53.5%",
              },
              {
                id: 55,
                name: "Jump Game",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 55,
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "39.5%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 5,
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
              {
                id: 45,
                name: "Jump Game II",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 45,
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "41.5%",
              },
              {
                id: 72,
                name: "Edit Distance",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 72,
                topics: ["String", "Dynamic Programming"],
                acceptance: "58.8%",
              },
            ],
          },
          {
            category: "Backtracking & Recursion",
            questions: [
              {
                id: 22,
                name: "Generate Parentheses",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 22,
                topics: ["String", "Dynamic Programming", "Backtracking"],
                acceptance: "77.1%",
              },
              {
                id: 17,
                name: "Letter Combinations of a Phone Number",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 17,
                topics: ["Hash Table", "String", "Backtracking"],
                acceptance: "63.9%",
              },
              {
                id: 78,
                name: "Subsets",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 78,
                topics: ["Array", "Backtracking", "Bit Manipulation"],
                acceptance: "80.9%",
              },
              {
                id: 39,
                name: "Combination Sum",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 39,
                topics: ["Array", "Backtracking"],
                acceptance: "74.7%",
              },
              {
                id: 51,
                name: "N-Queens",
                difficulty: "Hard",
                frequency: "Low",
                leetcodeNum: 51,
                topics: ["Array", "Backtracking"],
                acceptance: "72.8%",
              },
            ],
          },
          {
            category: "Math & Bit Manipulation",
            questions: [
              {
                id: 9,
                name: "Palindrome Number",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 9,
                topics: ["Math"],
                acceptance: "59.2%",
              },
              {
                id: 13,
                name: "Roman to Integer",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 13,
                topics: ["Hash Table", "Math", "String"],
                acceptance: "64.9%",
              },
              {
                id: 7,
                name: "Reverse Integer",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 7,
                topics: ["Math"],
                acceptance: "30.3%",
              },
              {
                id: 50,
                name: "Pow(x, n)",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 50,
                topics: ["Math", "Recursion"],
                acceptance: "37.0%",
              },
              {
                id: 67,
                name: "Add Binary",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 67,
                topics: ["Math", "String", "Bit Manipulation"],
                acceptance: "55.7%",
              },
            ],
          },
        ],
      },
      {
        name: "Meta",
        logo: Meta,
        color: "#0668E1",
        totalQuestions: 550,
        frequency: "Extremely High",
        categories: [
          {
            category: "Array & String",
            questions: [
              {
                id: 88,
                name: "Merge Sorted Array",
                difficulty: "Easy",
                frequency: "88.1%",
                leetcodeNum: 88,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "52.9%",
              },
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "83.8%",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "84.8%",
                leetcodeNum: 56,
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 31,
                name: "Next Permutation",
                difficulty: "Medium",
                frequency: "78.3%",
                leetcodeNum: 31,
                topics: ["Array", "Two Pointers"],
                acceptance: "43.1%",
              },
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "76.7%",
                leetcodeNum: 15,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "71.1%",
                leetcodeNum: 20,
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 14,
                name: "Longest Common Prefix",
                difficulty: "Easy",
                frequency: "66.7%",
                leetcodeNum: 14,
                topics: ["String", "Trie"],
                acceptance: "45.5%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "59.0%",
                leetcodeNum: 5,
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
              {
                id: 26,
                name: "Remove Duplicates from Sorted Array",
                difficulty: "Easy",
                frequency: "59.0%",
                leetcodeNum: 26,
                topics: ["Array", "Two Pointers"],
                acceptance: "60.4%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "53.6%",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
            ],
          },
          {
            category: "Binary Search",
            questions: [
              {
                id: 34,
                name: "Find First and Last Position of Element in Sorted Array",
                difficulty: "Medium",
                frequency: "72.1%",
                leetcodeNum: 34,
                topics: ["Array", "Binary Search"],
                acceptance: "46.8%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "57.0%",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "58.2%",
                leetcodeNum: 4,
                topics: ["Array", "Binary Search", "Divide and Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 69,
                name: "Sqrt(x)",
                difficulty: "Easy",
                frequency: "44.3%",
                leetcodeNum: 69,
                topics: ["Math", "Binary Search"],
                acceptance: "40.4%",
              },
              {
                id: 74,
                name: "Search a 2D Matrix",
                difficulty: "Medium",
                frequency: "43.3%",
                leetcodeNum: 74,
                topics: ["Array", "Binary Search", "Matrix"],
                acceptance: "52.3%",
              },
            ],
          },
          {
            category: "Trees & BFS/DFS",
            questions: [
              {
                id: 103,
                name: "Binary Tree Zigzag Level Order Traversal",
                difficulty: "Medium",
                frequency: "43.3%",
                leetcodeNum: 103,
                topics: ["Tree", "Breadth-First Search", "Binary Tree"],
                acceptance: "61.7%",
              },
              {
                id: 102,
                name: "Binary Tree Level Order Traversal",
                difficulty: "Medium",
                frequency: "42.7%",
                leetcodeNum: 102,
                topics: ["Tree", "Breadth-First Search", "Binary Tree"],
                acceptance: "70.6%",
              },
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "41.0%",
                leetcodeNum: 98,
                topics: ["Tree", "Depth-First Search", "Binary Search Tree"],
                acceptance: "34.4%",
              },
              {
                id: 94,
                name: "Binary Tree Inorder Traversal",
                difficulty: "Easy",
                frequency: "34.5%",
                leetcodeNum: 94,
                topics: ["Stack", "Tree", "Depth-First Search"],
                acceptance: "78.6%",
              },
              {
                id: 101,
                name: "Symmetric Tree",
                difficulty: "Easy",
                frequency: "30.6%",
                leetcodeNum: 101,
                topics: ["Tree", "Depth-First Search", "Breadth-First Search"],
                acceptance: "59.3%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Backtracking",
            questions: [
              {
                id: 70,
                name: "Climbing Stairs",
                difficulty: "Easy",
                frequency: "52.8%",
                leetcodeNum: 70,
                topics: ["Math", "Dynamic Programming", "Memoization"],
                acceptance: "53.5%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "52.1%",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 22,
                name: "Generate Parentheses",
                difficulty: "Medium",
                frequency: "47.5%",
                leetcodeNum: 22,
                topics: ["String", "Dynamic Programming", "Backtracking"],
                acceptance: "77.1%",
              },
              {
                id: 78,
                name: "Subsets",
                difficulty: "Medium",
                frequency: "64.1%",
                leetcodeNum: 78,
                topics: ["Array", "Backtracking", "Bit Manipulation"],
                acceptance: "80.9%",
              },
              {
                id: 17,
                name: "Letter Combinations of a Phone Number",
                difficulty: "Medium",
                frequency: "64.1%",
                leetcodeNum: 17,
                topics: ["Hash Table", "String", "Backtracking"],
                acceptance: "63.9%",
              },
            ],
          },
          {
            category: "Design & Implementation",
            questions: [
              {
                id: 71,
                name: "Simplify Path",
                difficulty: "Medium",
                frequency: "85.9%",
                leetcodeNum: 71,
                topics: ["String", "Stack"],
                acceptance: "47.9%",
              },
              {
                id: 48,
                name: "Rotate Image",
                difficulty: "Medium",
                frequency: "47.1%",
                leetcodeNum: 48,
                topics: ["Array", "Math", "Matrix"],
                acceptance: "77.9%",
              },
              {
                id: 54,
                name: "Spiral Matrix",
                difficulty: "Medium",
                frequency: "46.6%",
                leetcodeNum: 54,
                topics: ["Array", "Matrix", "Simulation"],
                acceptance: "53.9%",
              },
            ],
          },
        ],
      },
      {
        name: "Microsoft",
        logo: Microsoft,
        color: "#00a4ef",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Array & String",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "100.0%",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 88,
                name: "Merge Sorted Array",
                difficulty: "Easy",
                frequency: "83.7%",
                leetcodeNum: 88,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "52.9%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "79.4%",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "75.8%",
                leetcodeNum: 5,
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "73.8%",
                leetcodeNum: 56,
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "72.9%",
                leetcodeNum: 20,
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "72.9%",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "72.9%",
                leetcodeNum: 15,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "69.8%",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "67.7%",
                leetcodeNum: 11,
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
            ],
          },
          {
            category: "Linked List",
            questions: [
              {
                id: 2,
                name: "Add Two Numbers",
                difficulty: "Medium",
                frequency: "78.3%",
                leetcodeNum: 2,
                topics: ["Linked List", "Math", "Recursion"],
                acceptance: "46.2%",
              },
              {
                id: 25,
                name: "Reverse Nodes in k-Group",
                difficulty: "Hard",
                frequency: "68.5%",
                leetcodeNum: 25,
                topics: ["Linked List", "Recursion"],
                acceptance: "63.0%",
              },
              {
                id: 21,
                name: "Merge Two Sorted Lists",
                difficulty: "Easy",
                frequency: "68.5%",
                leetcodeNum: 21,
                topics: ["Linked List", "Recursion"],
                acceptance: "66.8%",
              },
              {
                id: 23,
                name: "Merge k Sorted Lists",
                difficulty: "Hard",
                frequency: "67.5%",
                leetcodeNum: 23,
                topics: [
                  "Linked List",
                  "Divide and Conquer",
                  "Heap (Priority Queue)",
                ],
                acceptance: "56.8%",
              },
              {
                id: 24,
                name: "Swap Nodes in Pairs",
                difficulty: "Medium",
                frequency: "52.8%",
                leetcodeNum: 24,
                topics: ["Linked List", "Recursion"],
                acceptance: "67.2%",
              },
              {
                id: 19,
                name: "Remove Nth Node From End of List",
                difficulty: "Medium",
                frequency: "49.5%",
                leetcodeNum: 19,
                topics: ["Linked List", "Two Pointers"],
                acceptance: "49.0%",
              },
              {
                id: 92,
                name: "Reverse Linked List II",
                difficulty: "Medium",
                frequency: "43.6%",
                leetcodeNum: 92,
                topics: ["Linked List"],
                acceptance: "49.6%",
              },
            ],
          },
          {
            category: "Trees & BFS/DFS",
            questions: [
              {
                id: 103,
                name: "Binary Tree Zigzag Level Order Traversal",
                difficulty: "Medium",
                frequency: "57.6%",
                leetcodeNum: 103,
                topics: ["Tree", "Breadth-First Search", "Binary Tree"],
                acceptance: "61.7%",
              },
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "55.6%",
                leetcodeNum: 98,
                topics: ["Tree", "Depth-First Search", "Binary Search Tree"],
                acceptance: "34.4%",
              },
              {
                id: 102,
                name: "Binary Tree Level Order Traversal",
                difficulty: "Medium",
                frequency: "50.9%",
                leetcodeNum: 102,
                topics: ["Tree", "Breadth-First Search", "Binary Tree"],
                acceptance: "70.6%",
              },
              {
                id: 94,
                name: "Binary Tree Inorder Traversal",
                difficulty: "Easy",
                frequency: "49.5%",
                leetcodeNum: 94,
                topics: ["Stack", "Tree", "Depth-First Search", "Binary Tree"],
                acceptance: "78.6%",
              },
              {
                id: 101,
                name: "Symmetric Tree",
                difficulty: "Easy",
                frequency: "42.5%",
                leetcodeNum: 101,
                topics: [
                  "Tree",
                  "Depth-First Search",
                  "Breadth-First Search",
                  "Binary Tree",
                ],
                acceptance: "59.3%",
              },
            ],
          },
          {
            category: "Matrix & Implementation",
            questions: [
              {
                id: 54,
                name: "Spiral Matrix",
                difficulty: "Medium",
                frequency: "68.3%",
                leetcodeNum: 54,
                topics: ["Array", "Matrix", "Simulation"],
                acceptance: "53.9%",
              },
              {
                id: 48,
                name: "Rotate Image",
                difficulty: "Medium",
                frequency: "67.7%",
                leetcodeNum: 48,
                topics: ["Array", "Math", "Matrix"],
                acceptance: "77.9%",
              },
              {
                id: 73,
                name: "Set Matrix Zeroes",
                difficulty: "Medium",
                frequency: "62.9%",
                leetcodeNum: 73,
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "60.7%",
              },
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "60.6%",
                leetcodeNum: 79,
                topics: [
                  "Array",
                  "String",
                  "Backtracking",
                  "Depth-First Search",
                  "Matrix",
                ],
                acceptance: "45.3%",
              },
              {
                id: 36,
                name: "Valid Sudoku",
                difficulty: "Medium",
                frequency: "50.2%",
                leetcodeNum: 36,
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "62.3%",
              },
            ],
          },
          {
            category: "Hard / Specialized",
            questions: [
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "75.7%",
                leetcodeNum: 42,
                topics: [
                  "Array",
                  "Two Pointers",
                  "Dynamic Programming",
                  "Stack",
                ],
                acceptance: "65.1%",
              },
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "72.9%",
                leetcodeNum: 4,
                topics: ["Array", "Binary Search", "Divide and Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 84,
                name: "Largest Rectangle in Histogram",
                difficulty: "Hard",
                frequency: "54.6%",
                leetcodeNum: 84,
                topics: ["Array", "Stack", "Monotonic Stack"],
                acceptance: "47.4%",
              },
              {
                id: 41,
                name: "First Missing Positive",
                difficulty: "Hard",
                frequency: "54.6%",
                leetcodeNum: 41,
                topics: ["Array", "Hash Table"],
                acceptance: "41.1%",
              },
            ],
          },
        ],
      },
      {
        name: "Apple",
        logo: Apple,
        color: "#555555",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Array & String",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "100.0%",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "78.3%",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 14,
                name: "Longest Common Prefix",
                difficulty: "Easy",
                frequency: "77.6%",
                leetcodeNum: 14,
                topics: ["String", "Trie"],
                acceptance: "45.5%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "76.9%",
                leetcodeNum: 20,
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "74.2%",
                leetcodeNum: 56,
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "74.2%",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
              {
                id: 88,
                name: "Merge Sorted Array",
                difficulty: "Easy",
                frequency: "73.8%",
                leetcodeNum: 88,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "52.9%",
              },
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "73.0%",
                leetcodeNum: 42,
                topics: [
                  "Array",
                  "Two Pointers",
                  "Dynamic Programming",
                  "Stack",
                ],
                acceptance: "65.1%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "68.5%",
                leetcodeNum: 5,
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "60.4%",
                leetcodeNum: 11,
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
            ],
          },
          {
            category: "Binary Search & Math",
            questions: [
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "75.8%",
                leetcodeNum: 4,
                topics: ["Array", "Binary Search", "Divide and Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 7,
                name: "Reverse Integer",
                difficulty: "Medium",
                frequency: "71.6%",
                leetcodeNum: 7,
                topics: ["Math"],
                acceptance: "30.3%",
              },
              {
                id: 9,
                name: "Palindrome Number",
                difficulty: "Easy",
                frequency: "68.0%",
                leetcodeNum: 9,
                topics: ["Math"],
                acceptance: "59.2%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "64.2%",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 69,
                name: "Sqrt(x)",
                difficulty: "Easy",
                frequency: "63.5%",
                leetcodeNum: 69,
                topics: ["Math", "Binary Search"],
                acceptance: "40.4%",
              },
              {
                id: 34,
                name: "Find First and Last Position of Element in Sorted Array",
                difficulty: "Medium",
                frequency: "52.1%",
                leetcodeNum: 34,
                topics: ["Array", "Binary Search"],
                acceptance: "46.8%",
              },
              {
                id: 50,
                name: "Pow(x, n)",
                difficulty: "Medium",
                frequency: "44.3%",
                leetcodeNum: 50,
                topics: ["Math", "Recursion"],
                acceptance: "37.0%",
              },
            ],
          },
          {
            category: "Trees & DFS/BFS",
            questions: [
              {
                id: 104,
                name: "Maximum Depth of Binary Tree",
                difficulty: "Easy",
                frequency: "52.1%",
                leetcodeNum: 104,
                topics: ["Tree", "Depth-First Search", "Breadth-First Search"],
                acceptance: "77.1%",
              },
              {
                id: 102,
                name: "Binary Tree Level Order Traversal",
                difficulty: "Medium",
                frequency: "52.1%",
                leetcodeNum: 102,
                topics: ["Tree", "Breadth-First Search", "Binary Tree"],
                acceptance: "70.6%",
              },
              {
                id: 101,
                name: "Symmetric Tree",
                difficulty: "Easy",
                frequency: "46.1%",
                leetcodeNum: 101,
                topics: ["Tree", "Depth-First Search", "Breadth-First Search"],
                acceptance: "59.3%",
              },
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "46.1%",
                leetcodeNum: 98,
                topics: ["Tree", "Depth-First Search", "Binary Search Tree"],
                acceptance: "34.4%",
              },
              {
                id: 94,
                name: "Binary Tree Inorder Traversal",
                difficulty: "Easy",
                frequency: "42.2%",
                leetcodeNum: 94,
                topics: ["Stack", "Tree", "Depth-First Search"],
                acceptance: "78.6%",
              },
            ],
          },
          {
            category: "Linked List",
            questions: [
              {
                id: 2,
                name: "Add Two Numbers",
                difficulty: "Medium",
                frequency: "66.8%",
                leetcodeNum: 2,
                topics: ["Linked List", "Math", "Recursion"],
                acceptance: "46.2%",
              },
              {
                id: 21,
                name: "Merge Two Sorted Lists",
                difficulty: "Easy",
                frequency: "62.7%",
                leetcodeNum: 21,
                topics: ["Linked List", "Recursion"],
                acceptance: "66.8%",
              },
              {
                id: 23,
                name: "Merge k Sorted Lists",
                difficulty: "Hard",
                frequency: "61.2%",
                leetcodeNum: 23,
                topics: ["Linked List", "Divide and Conquer", "Heap"],
                acceptance: "56.8%",
              },
              {
                id: 19,
                name: "Remove Nth Node From End of List",
                difficulty: "Medium",
                frequency: "53.3%",
                leetcodeNum: 19,
                topics: ["Linked List", "Two Pointers"],
                acceptance: "49.0%",
              },
              {
                id: 92,
                name: "Reverse Linked List II",
                difficulty: "Medium",
                frequency: "42.2%",
                leetcodeNum: 92,
                topics: ["Linked List"],
                acceptance: "49.6%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Backtracking",
            questions: [
              {
                id: 70,
                name: "Climbing Stairs",
                difficulty: "Easy",
                frequency: "70.1%",
                leetcodeNum: 70,
                topics: ["Math", "Dynamic Programming"],
                acceptance: "53.5%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "63.5%",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 22,
                name: "Generate Parentheses",
                difficulty: "Medium",
                frequency: "61.2%",
                leetcodeNum: 22,
                topics: ["String", "Dynamic Programming", "Backtracking"],
                acceptance: "77.1%",
              },
              {
                id: 55,
                name: "Jump Game",
                difficulty: "Medium",
                frequency: "47.8%",
                leetcodeNum: 55,
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "39.5%",
              },
              {
                id: 78,
                name: "Subsets",
                difficulty: "Medium",
                frequency: "37.4%",
                leetcodeNum: 78,
                topics: ["Array", "Backtracking", "Bit Manipulation"],
                acceptance: "80.9%",
              },
            ],
          },
        ],
      },
      {
        name: "Netflix",
        logo: Netflix,
        color: "#E50914",
        totalQuestions: 38,
        frequency: "High",
        categories: [
          {
            category: "Arrays & Intervals",
            questions: [
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "Very High",
                leetcodeNum: 56,
                topics: ["Array", "Sorting", "Greedy"],
                acceptance: "49.4%",
              },
              {
                id: 253,
                name: "Meeting Rooms II",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 253,
                topics: ["Array", "Two Pointers", "Heap", "Prefix Sum"],
                acceptance: "52.1%",
              },
              {
                id: 41,
                name: "First Missing Positive",
                difficulty: "Hard",
                frequency: "High",
                leetcodeNum: 41,
                topics: ["Array", "Hash Table"],
                acceptance: "41.1%",
              },
              {
                id: 121,
                name: "Best Time to Buy and Sell Stock",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 121,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "55.3%",
              },
              {
                id: 228,
                name: "Summary Ranges",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 228,
                topics: ["Array"],
                acceptance: "53.0%",
              },
            ],
          },
          {
            category: "Design & Caching",
            questions: [
              {
                id: 146,
                name: "LRU Cache",
                difficulty: "Medium",
                frequency: "Very High",
                leetcodeNum: 146,
                topics: [
                  "Hash Table",
                  "Linked List",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "45.2%",
              },
              {
                id: 981,
                name: "Time Based Key-Value Store",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 981,
                topics: ["Hash Table", "Binary Search", "Design"],
                acceptance: "49.4%",
              },
              {
                id: 359,
                name: "Logger Rate Limiter",
                difficulty: "Easy",
                frequency: "High",
                leetcodeNum: 359,
                topics: ["Hash Table", "Design", "Data Stream"],
                acceptance: "76.6%",
              },
              {
                id: 380,
                name: "Insert Delete GetRandom O(1)",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 380,
                topics: ["Array", "Hash Table", "Design"],
                acceptance: "54.9%",
              },
              {
                id: 232,
                name: "Implement Queue using Stacks",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 232,
                topics: ["Stack", "Design", "Queue"],
                acceptance: "68.1%",
              },
            ],
          },
          {
            category: "Strings & Sliding Window",
            questions: [
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 1249,
                name: "Minimum Remove to Make Valid Parentheses",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 1249,
                topics: ["String", "Stack"],
                acceptance: "70.7%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "Medium",
                leetcodeNum: 20,
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 76,
                name: "Minimum Window Substring",
                difficulty: "Hard",
                frequency: "Low",
                leetcodeNum: 76,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "45.4%",
              },
            ],
          },
          {
            category: "Graphs & Search",
            questions: [
              {
                id: 332,
                name: "Reconstruct Itinerary",
                difficulty: "Hard",
                frequency: "High",
                leetcodeNum: 332,
                topics: ["DFS", "Graph", "Eulerian Circuit"],
                acceptance: "43.6%",
              },
              {
                id: 210,
                name: "Course Schedule II",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 210,
                topics: ["DFS", "BFS", "Graph", "Topological Sort"],
                acceptance: "53.4%",
              },
              {
                id: 743,
                name: "Network Delay Time",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 743,
                topics: ["Graph", "BFS", "DFS", "Shortest Path"],
                acceptance: "57.4%",
              },
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 79,
                topics: ["Backtracking", "Matrix", "DFS"],
                acceptance: "45.3%",
              },
            ],
          },
          {
            category: "Heaps & Sorting",
            questions: [
              {
                id: 347,
                name: "Top K Frequent Elements",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 347,
                topics: ["Array", "Heap", "Quickselect", "Bucket Sort"],
                acceptance: "64.6%",
              },
              {
                id: 692,
                name: "Top K Frequent Words",
                difficulty: "Medium",
                frequency: "Medium",
                leetcodeNum: 692,
                topics: ["Hash Table", "String", "Heap", "Trie"],
                acceptance: "59.3%",
              },
              {
                id: 215,
                name: "Kth Largest Element in an Array",
                difficulty: "Medium",
                frequency: "High",
                leetcodeNum: 215,
                topics: ["Array", "Divide and Conquer", "Heap", "Quickselect"],
                acceptance: "67.1%",
              },
            ],
          },
        ],
      },
      {
        name: "Adobe",
        logo: Adobe,
        color: "#FF0000",
        totalQuestions: 101,
        frequency: "Very High",
        categories: [
          {
            category: "Array & String",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "100.0%",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 14,
                name: "Longest Common Prefix",
                difficulty: "Easy",
                frequency: "79.0%",
                leetcodeNum: 14,
                topics: ["String", "Trie"],
                acceptance: "45.5%",
              },
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "78.6%",
                leetcodeNum: 15,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "75.7%",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "72.2%",
                leetcodeNum: 42,
                topics: [
                  "Array",
                  "Two Pointers",
                  "Dynamic Programming",
                  "Stack",
                ],
                acceptance: "65.1%",
              },
              {
                id: 88,
                name: "Merge Sorted Array",
                difficulty: "Easy",
                frequency: "71.6%",
                leetcodeNum: 88,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "52.9%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "68.6%",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "64.3%",
                leetcodeNum: 11,
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "64.3%",
                leetcodeNum: 5,
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
              {
                id: 31,
                name: "Next Permutation",
                difficulty: "Medium",
                frequency: "63.4%",
                leetcodeNum: 31,
                topics: ["Array", "Two Pointers"],
                acceptance: "43.1%",
              },
            ],
          },
          {
            category: "Linked List",
            questions: [
              {
                id: 2,
                name: "Add Two Numbers",
                difficulty: "Medium",
                frequency: "70.5%",
                leetcodeNum: 2,
                topics: ["Linked List", "Math", "Recursion"],
                acceptance: "46.2%",
              },
              {
                id: 21,
                name: "Merge Two Sorted Lists",
                difficulty: "Easy",
                frequency: "61.7%",
                leetcodeNum: 21,
                topics: ["Linked List", "Recursion"],
                acceptance: "66.8%",
              },
              {
                id: 23,
                name: "Merge k Sorted Lists",
                difficulty: "Hard",
                frequency: "49.0%",
                leetcodeNum: 23,
                topics: ["Linked List", "Divide and Conquer", "Heap"],
                acceptance: "56.8%",
              },
              {
                id: 25,
                name: "Reverse Nodes in k-Group",
                difficulty: "Hard",
                frequency: "42.9%",
                leetcodeNum: 25,
                topics: ["Linked List", "Recursion"],
                acceptance: "63.0%",
              },
              {
                id: 92,
                name: "Reverse Linked List II",
                difficulty: "Medium",
                frequency: "42.9%",
                leetcodeNum: 92,
                topics: ["Linked List"],
                acceptance: "49.6%",
              },
            ],
          },
          {
            category: "Trees & Binary Search",
            questions: [
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "77.0%",
                leetcodeNum: 4,
                topics: ["Array", "Binary Search", "Divide and Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 105,
                name: "Construct Binary Tree from Preorder and Inorder Traversal",
                difficulty: "Medium",
                frequency: "45.2%",
                leetcodeNum: 105,
                topics: ["Array", "Hash Table", "Tree"],
                acceptance: "66.8%",
              },
              {
                id: 103,
                name: "Binary Tree Zigzag Level Order Traversal",
                difficulty: "Medium",
                frequency: "42.9%",
                leetcodeNum: 103,
                topics: ["Tree", "Breadth-First Search", "Binary Tree"],
                acceptance: "61.7%",
              },
              {
                id: 94,
                name: "Binary Tree Inorder Traversal",
                difficulty: "Easy",
                frequency: "37.3%",
                leetcodeNum: 94,
                topics: ["Stack", "Tree", "DFS"],
                acceptance: "78.6%",
              },
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "29.7%",
                leetcodeNum: 98,
                topics: ["Tree", "DFS", "BST"],
                acceptance: "34.4%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Backtracking",
            questions: [
              {
                id: 70,
                name: "Climbing Stairs",
                difficulty: "Easy",
                frequency: "69.9%",
                leetcodeNum: 70,
                topics: ["Math", "DP", "Memoization"],
                acceptance: "53.5%",
              },
              {
                id: 55,
                name: "Jump Game",
                difficulty: "Medium",
                frequency: "57.5%",
                leetcodeNum: 55,
                topics: ["Array", "DP", "Greedy"],
                acceptance: "39.5%",
              },
              {
                id: 22,
                name: "Generate Parentheses",
                difficulty: "Medium",
                frequency: "56.3%",
                leetcodeNum: 22,
                topics: ["String", "DP", "Backtracking"],
                acceptance: "77.1%",
              },
              {
                id: 17,
                name: "Letter Combinations of a Phone Number",
                difficulty: "Medium",
                frequency: "55.1%",
                leetcodeNum: 17,
                topics: ["Hash Table", "String", "Backtracking"],
                acceptance: "63.9%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "49.0%",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "DP"],
                acceptance: "52.1%",
              },
            ],
          },
          {
            category: "Math & Matrix",
            questions: [
              {
                id: 7,
                name: "Reverse Integer",
                difficulty: "Medium",
                frequency: "74.3%",
                leetcodeNum: 7,
                topics: ["Math"],
                acceptance: "30.3%",
              },
              {
                id: 9,
                name: "Palindrome Number",
                difficulty: "Easy",
                frequency: "68.6%",
                leetcodeNum: 9,
                topics: ["Math"],
                acceptance: "59.2%",
              },
              {
                id: 54,
                name: "Spiral Matrix",
                difficulty: "Medium",
                frequency: "62.6%",
                leetcodeNum: 54,
                topics: ["Array", "Matrix", "Simulation"],
                acceptance: "53.9%",
              },
              {
                id: 48,
                name: "Rotate Image",
                difficulty: "Medium",
                frequency: "55.1%",
                leetcodeNum: 48,
                topics: ["Array", "Math", "Matrix"],
                acceptance: "77.9%",
              },
              {
                id: 73,
                name: "Set Matrix Zeroes",
                difficulty: "Medium",
                frequency: "59.7%",
                leetcodeNum: 73,
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "60.7%",
              },
            ],
          },
        ],
      },
      {
        name: "LinkedIn",
        logo: LinkedIn,
        color: "#0077B5",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Design & Data Structures",
            questions: [
              {
                id: 716,
                name: "Max Stack",
                difficulty: "Hard",
                frequency: "100.0%",
                leetcodeNum: 716,
                topics: [
                  "Linked List",
                  "Stack",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "45.5%",
              },
              {
                id: 432,
                name: "All O`one Data Structure",
                difficulty: "Hard",
                frequency: "94.4%",
                leetcodeNum: 432,
                topics: [
                  "Hash Table",
                  "Linked List",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "44.1%",
              },
              {
                id: 146,
                name: "LRU Cache",
                difficulty: "Medium",
                frequency: "56.3%",
                leetcodeNum: 146,
                topics: [
                  "Hash Table",
                  "Linked List",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "45.2%",
              },
              {
                id: 380,
                name: "Insert Delete GetRandom O(1)",
                difficulty: "Medium",
                frequency: "69.3%",
                leetcodeNum: 380,
                topics: ["Array", "Hash Table", "Math", "Design"],
                acceptance: "54.9%",
              },
              {
                id: 173,
                name: "Binary Search Tree Iterator",
                difficulty: "Medium",
                frequency: "50.0%",
                leetcodeNum: 173,
                topics: ["Stack", "Tree", "Design", "BST"],
                acceptance: "74.8%",
              },
              {
                id: 155,
                name: "Min Stack",
                difficulty: "Medium",
                frequency: "43.4%",
                leetcodeNum: 155,
                topics: ["Stack", "Design"],
                acceptance: "56.4%",
              },
            ],
          },
          {
            category: "Graph & BFS/DFS",
            questions: [
              {
                id: 127,
                name: "Word Ladder",
                difficulty: "Hard",
                frequency: "79.1%",
                leetcodeNum: 127,
                topics: ["Hash Table", "String", "BFS"],
                acceptance: "39.1%",
              },
              {
                id: 200,
                name: "Number of Islands",
                difficulty: "Medium",
                frequency: "75.6%",
                leetcodeNum: 200,
                topics: ["Array", "DFS", "BFS", "Union Find"],
                acceptance: "59.1%",
              },
              {
                id: 277,
                name: "Find the Celebrity",
                difficulty: "Medium",
                frequency: "79.1%",
                leetcodeNum: 277,
                topics: ["Two Pointers", "Graph", "Greedy"],
                acceptance: "48.4%",
              },
              {
                id: 261,
                name: "Graph Valid Tree",
                difficulty: "Medium",
                frequency: "45.8%",
                leetcodeNum: 261,
                topics: ["DFS", "BFS", "Union Find", "Graph"],
                acceptance: "49.3%",
              },
              {
                id: 323,
                name: "Number of Connected Components",
                difficulty: "Medium",
                frequency: "43.4%",
                leetcodeNum: 323,
                topics: ["DFS", "BFS", "Union Find", "Graph"],
                acceptance: "64.2%",
              },
            ],
          },
          {
            category: "Trees & Recursion",
            questions: [
              {
                id: 364,
                name: "Nested List Weight Sum II",
                difficulty: "Medium",
                frequency: "94.7%",
                leetcodeNum: 364,
                topics: ["Stack", "DFS", "BFS"],
                acceptance: "65.5%",
              },
              {
                id: 236,
                name: "Lowest Common Ancestor of a Binary Tree",
                difficulty: "Medium",
                frequency: "63.0%",
                leetcodeNum: 236,
                topics: ["Tree", "DFS", "Binary Tree"],
                acceptance: "62.4%",
              },
              {
                id: 297,
                name: "Serialize and Deserialize Binary Tree",
                difficulty: "Hard",
                frequency: "66.4%",
                leetcodeNum: 297,
                topics: ["String", "Tree", "BFS", "DFS", "Design"],
                acceptance: "58.9%",
              },
              {
                id: 104,
                name: "Maximum Depth of Binary Tree",
                difficulty: "Easy",
                frequency: "63.9%",
                leetcodeNum: 104,
                topics: ["Tree", "DFS", "BFS"],
                acceptance: "77.1%",
              },
              {
                id: 102,
                name: "Binary Tree Level Order Traversal",
                difficulty: "Medium",
                frequency: "56.3%",
                leetcodeNum: 102,
                topics: ["Tree", "BFS", "Binary Tree"],
                acceptance: "70.6%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Backtracking",
            questions: [
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "77.4%",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "DP"],
                acceptance: "52.1%",
              },
              {
                id: 256,
                name: "Paint House",
                difficulty: "Medium",
                frequency: "67.9%",
                leetcodeNum: 256,
                topics: ["Array", "DP"],
                acceptance: "63.7%",
              },
              {
                id: 198,
                name: "House Robber",
                difficulty: "Medium",
                frequency: "57.6%",
                leetcodeNum: 198,
                topics: ["Array", "DP"],
                acceptance: "52.3%",
              },
              {
                id: 17,
                name: "Letter Combinations of a Phone Number",
                difficulty: "Medium",
                frequency: "63.0%",
                leetcodeNum: 17,
                topics: ["Hash Table", "String", "Backtracking"],
                acceptance: "63.9%",
              },
              {
                id: 698,
                name: "Partition to K Equal Sum Subsets",
                difficulty: "Medium",
                frequency: "68.7%",
                leetcodeNum: 698,
                topics: ["Array", "DP", "Backtracking", "Bitmask"],
                acceptance: "38.1%",
              },
            ],
          },
          {
            category: "Binary Search & Two Pointers",
            questions: [
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "71.3%",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 76,
                name: "Minimum Window Substring",
                difficulty: "Hard",
                frequency: "64.8%",
                leetcodeNum: 76,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "45.4%",
              },
              {
                id: 34,
                name: "Find First and Last Position of Element in Sorted Array",
                difficulty: "Medium",
                frequency: "67.9%",
                leetcodeNum: 34,
                topics: ["Array", "Binary Search"],
                acceptance: "46.8%",
              },
              {
                id: 215,
                name: "Kth Largest Element in an Array",
                difficulty: "Medium",
                frequency: "63.0%",
                leetcodeNum: 215,
                topics: ["Array", "Divide and Conquer", "Heap", "Quickselect"],
                acceptance: "67.1%",
              },
            ],
          },
        ],
      },
      {
        name: "J.P. Morgan",
        logo: JP,
        color: "#2C2E2F",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Math & Bit Manipulation",
            questions: [
              {
                id: 1356,
                name: "Sort Integers by The Number of 1 Bits",
                difficulty: "Easy",
                frequency: "100.0%",
                leetcodeNum: 1356,
                topics: ["Array", "Bit Manipulation", "Sorting"],
                acceptance: "78.6%",
              },
              {
                id: 780,
                name: "Reaching Points",
                difficulty: "Hard",
                frequency: "96.5%",
                leetcodeNum: 780,
                topics: ["Math"],
                acceptance: "33.6%",
              },
              {
                id: 1015,
                name: "Numbers With Repeated Digits",
                difficulty: "Hard",
                frequency: "74.3%",
                leetcodeNum: 1015,
                topics: ["Math", "Dynamic Programming"],
                acceptance: "43.4%",
              },
              {
                id: 202,
                name: "Happy Number",
                difficulty: "Easy",
                frequency: "59.5%",
                leetcodeNum: 202,
                topics: ["Hash Table", "Math", "Two Pointers"],
                acceptance: "58.0%",
              },
              {
                id: 263,
                name: "Ugly Number",
                difficulty: "Easy",
                frequency: "36.3%",
                leetcodeNum: 263,
                topics: ["Math"],
                acceptance: "42.3%",
              },
              {
                id: 50,
                name: "Pow(x, n)",
                difficulty: "Medium",
                frequency: "70.6%",
                leetcodeNum: 50,
                topics: ["Math", "Recursion"],
                acceptance: "37.0%",
              },
            ],
          },
          {
            category: "Arrays & Greedy Strategy",
            questions: [
              {
                id: 1481,
                name: "Least Number of Unique Integers after K Removals",
                difficulty: "Medium",
                frequency: "93.1%",
                leetcodeNum: 1481,
                topics: ["Array", "Hash Table", "Greedy", "Sorting"],
                acceptance: "63.4%",
              },
              {
                id: 121,
                name: "Best Time to Buy and Sell Stock",
                difficulty: "Easy",
                frequency: "76.0%",
                leetcodeNum: 121,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "55.2%",
              },
              {
                id: 122,
                name: "Best Time to Buy and Sell Stock II",
                difficulty: "Medium",
                frequency: "50.9%",
                leetcodeNum: 122,
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "69.5%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "72.4%",
                leetcodeNum: 56,
                topics: ["Array", "Sorting"],
                acceptance: "49.3%",
              },
              {
                id: 1528,
                name: "Shuffle an Array",
                difficulty: "Medium",
                frequency: "40.5%",
                leetcodeNum: 384,
                topics: ["Array", "Math", "Design"],
                acceptance: "59.0%",
              },
              {
                id: 605,
                name: "Can Place Flowers",
                difficulty: "Easy",
                frequency: "72.4%",
                leetcodeNum: 605,
                topics: ["Array", "Greedy"],
                acceptance: "28.8%",
              },
            ],
          },
          {
            category: "Strings & Hashing",
            questions: [
              {
                id: 1545,
                name: "Minimum Suffix Flips",
                difficulty: "Medium",
                frequency: "90.8%",
                leetcodeNum: 1545,
                topics: ["String", "Greedy"],
                acceptance: "73.4%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "77.6%",
                leetcodeNum: 49,
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "59.5%",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 12,
                name: "Integer to Roman",
                difficulty: "Medium",
                frequency: "36.3%",
                leetcodeNum: 12,
                topics: ["Hash Table", "Math", "String"],
                acceptance: "68.6%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "65.6%",
                leetcodeNum: 5,
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Prefix Sum",
            questions: [
              {
                id: 70,
                name: "Climbing Stairs",
                difficulty: "Easy",
                frequency: "55.6%",
                leetcodeNum: 70,
                topics: ["Math", "Dynamic Programming"],
                acceptance: "53.5%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "50.9%",
                leetcodeNum: 53,
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.0%",
              },
              {
                id: 560,
                name: "Subarray Sum Equals K",
                difficulty: "Medium",
                frequency: "50.9%",
                leetcodeNum: 560,
                topics: ["Array", "Hash Table", "Prefix Sum"],
                acceptance: "45.4%",
              },
              {
                id: 322,
                name: "Coin Change",
                difficulty: "Medium",
                frequency: "44.8%",
                leetcodeNum: 322,
                topics: ["Array", "Dynamic Programming", "BFS"],
                acceptance: "46.4%",
              },
              {
                id: 118,
                name: "Pascal's Triangle",
                difficulty: "Easy",
                frequency: "44.8%",
                leetcodeNum: 118,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "77.0%",
              },
            ],
          },
          {
            category: "Trees & Binary Search",
            questions: [
              {
                id: 2415,
                name: "Reverse Odd Levels of Binary Tree",
                difficulty: "Medium",
                frequency: "70.4%",
                leetcodeNum: 2415,
                topics: ["Tree", "BFS", "Binary Tree"],
                acceptance: "86.6%",
              },
              {
                id: 199,
                name: "Binary Tree Right Side View",
                difficulty: "Medium",
                frequency: "44.8%",
                leetcodeNum: 199,
                topics: ["Tree", "BFS", "DFS"],
                acceptance: "67.0%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "71.3%",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 215,
                name: "Kth Largest Element in an Array",
                difficulty: "Medium",
                frequency: "36.3%",
                leetcodeNum: 215,
                topics: ["Array", "Divide and Conquer", "Heap"],
                acceptance: "67.9%",
              },
            ],
          },
        ],
      },
      {
        name: "Nvidia",
        logo: Nvidia,
        color: "#76B900",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Bit Manipulation & Math",
            questions: [
              {
                id: 190,
                name: "Reverse Bits",
                difficulty: "Easy",
                frequency: "75.8%",
                leetcodeNum: 190,
                topics: ["Divide and Conquer", "Bit Manipulation"],
                acceptance: "63.2%",
              },
              {
                id: 136,
                name: "Single Number",
                difficulty: "Easy",
                frequency: "38.4%",
                leetcodeNum: 136,
                topics: ["Array", "Bit Manipulation"],
                acceptance: "76.0%",
              },
              {
                id: 191,
                name: "Counting Bits",
                difficulty: "Easy",
                frequency: "47.7%",
                leetcodeNum: 338,
                topics: ["Dynamic Programming", "Bit Manipulation"],
                acceptance: "79.7%",
              },
              {
                id: 2601,
                name: "Find The Original Array of Prefix Xor",
                difficulty: "Medium",
                frequency: "47.7%",
                leetcodeNum: 2433,
                topics: ["Array", "Bit Manipulation"],
                acceptance: "88.1%",
              },
              {
                id: 2571,
                name: "Minimum Operations to Reduce an Integer to 0",
                difficulty: "Medium",
                frequency: "85.3%",
                leetcodeNum: 2571,
                topics: ["Greedy", "Bit Manipulation"],
                acceptance: "57.3%",
              },
              {
                id: 149,
                name: "Max Points on a Line",
                difficulty: "Hard",
                frequency: "47.7%",
                leetcodeNum: 149,
                topics: ["Hash Table", "Math", "Geometry"],
                acceptance: "29.0%",
              },
            ],
          },
          {
            category: "High-Performance Design",
            questions: [
              {
                id: 146,
                name: "LRU Cache",
                difficulty: "Medium",
                frequency: "89.6%",
                leetcodeNum: 146,
                topics: [
                  "Hash Table",
                  "Linked List",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "45.2%",
              },
              {
                id: 295,
                name: "Find Median from Data Stream",
                difficulty: "Hard",
                frequency: "63.8%",
                leetcodeNum: 295,
                topics: ["Heap (Priority Queue)", "Design", "Data Stream"],
                acceptance: "53.3%",
              },
              {
                id: 208,
                name: "Implement Trie (Prefix Tree)",
                difficulty: "Medium",
                frequency: "47.7%",
                leetcodeNum: 208,
                topics: ["Hash Table", "String", "Design", "Trie"],
                acceptance: "67.9%",
              },
              {
                id: 706,
                name: "Design HashMap",
                difficulty: "Easy",
                frequency: "47.7%",
                leetcodeNum: 706,
                topics: ["Array", "Hash Table", "Design"],
                acceptance: "65.9%",
              },
              {
                id: 1148,
                name: "Task Scheduler II",
                difficulty: "Medium",
                frequency: "67.4%",
                leetcodeNum: 2365,
                topics: ["Array", "Hash Table", "Simulation"],
                acceptance: "54.0%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Optimization",
            questions: [
              {
                id: 121,
                name: "Best Time to Buy and Sell Stock",
                difficulty: "Easy",
                frequency: "82.0%",
                leetcodeNum: 121,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "55.3%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "78.0%",
                leetcodeNum: 53,
                topics: ["Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 322,
                name: "Coin Change",
                difficulty: "Medium",
                frequency: "38.4%",
                leetcodeNum: 322,
                topics: ["Array", "Dynamic Programming", "BFS"],
                acceptance: "46.5%",
              },
              {
                id: 329,
                name: "Longest Increasing Path in a Matrix",
                difficulty: "Hard",
                frequency: "38.4%",
                leetcodeNum: 329,
                topics: ["Dynamic Programming", "DFS", "Graph", "Matrix"],
                acceptance: "55.3%",
              },
              {
                id: 221,
                name: "Maximal Square",
                difficulty: "Medium",
                frequency: "54.4%",
                leetcodeNum: 221,
                topics: ["Array", "Dynamic Programming", "Matrix"],
                acceptance: "48.8%",
              },
            ],
          },
          {
            category: "Trees, Graphs & Search",
            questions: [
              {
                id: 200,
                name: "Number of Islands",
                difficulty: "Medium",
                frequency: "78.0%",
                leetcodeNum: 200,
                topics: ["DFS", "BFS", "Union Find", "Matrix"],
                acceptance: "62.3%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "75.8%",
                leetcodeNum: 33,
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 210,
                name: "Course Schedule II",
                difficulty: "Medium",
                frequency: "47.7%",
                leetcodeNum: 210,
                topics: ["DFS", "BFS", "Graph", "Topological Sort"],
                acceptance: "53.4%",
              },
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "54.4%",
                leetcodeNum: 98,
                topics: ["Tree", "DFS", "BST"],
                acceptance: "34.4%",
              },
              {
                id: 124,
                name: "Binary Tree Maximum Path Sum",
                difficulty: "Hard",
                frequency: "54.4%",
                leetcodeNum: 124,
                topics: ["DP", "Tree", "DFS"],
                acceptance: "41.2%",
              },
            ],
          },
          {
            category: "Arrays & Two Pointers",
            questions: [
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "38.4%",
                leetcodeNum: 15,
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "80.1%",
                leetcodeNum: 1,
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "67.4%",
                leetcodeNum: 56,
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 48,
                name: "Rotate Image",
                difficulty: "Medium",
                frequency: "63.8%",
                leetcodeNum: 48,
                topics: ["Array", "Math", "Matrix"],
                acceptance: "77.9%",
              },
            ],
          },
        ],
      },
      {
        name: "Uber",
        logo: Uber,
        color: "#000000",
        totalQuestions: 101,
        frequency: "Very High",
        categories: [
          {
            category: "Grid & Matrix (Logistics)",
            questions: [
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "73.2%",
                leetcodeNum: 79,
                topics: ["Array", "Backtracking", "Matrix"],
                acceptance: "45.3%",
              },
              {
                id: 54,
                name: "Spiral Matrix",
                difficulty: "Medium",
                frequency: "69.5%",
                leetcodeNum: 54,
                topics: ["Array", "Matrix", "Simulation"],
                acceptance: "53.9%",
              },
              {
                id: 36,
                name: "Valid Sudoku",
                difficulty: "Medium",
                frequency: "68.7%",
                leetcodeNum: 36,
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "62.3%",
              },
              {
                id: 48,
                name: "Rotate Image",
                difficulty: "Medium",
                frequency: "59.3%",
                leetcodeNum: 48,
                topics: ["Array", "Math", "Matrix"],
                acceptance: "77.9%",
              },
              {
                id: 73,
                name: "Set Matrix Zeroes",
                difficulty: "Medium",
                frequency: "38.9%",
                leetcodeNum: 73,
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "60.7%",
              },
              {
                id: 64,
                name: "Minimum Path Sum",
                difficulty: "Medium",
                frequency: "38.9%",
                leetcodeNum: 64,
                topics: ["Array", "Dynamic Programming", "Matrix"],
                acceptance: "66.5%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Strings",
            questions: [
              {
                id: 121,
                name: "Best Time to Buy and Sell Stock",
                difficulty: "Easy",
                frequency: "76.3%",
                leetcodeNum: 121,
                topics: ["Array", "Dynamic Programming"],
                acceptance: "55.3%",
              },
              {
                id: 68,
                name: "Text Justification",
                difficulty: "Hard",
                frequency: "71.0%",
                leetcodeNum: 68,
                topics: ["Array", "String", "Simulation"],
                acceptance: "48.1%",
              },
              {
                id: 139,
                name: "Word Break",
                difficulty: "Medium",
                frequency: "60.6%",
                leetcodeNum: 139,
                topics: ["DP", "Trie", "String"],
                acceptance: "48.3%",
              },
              {
                id: 91,
                name: "Decode Ways",
                difficulty: "Medium",
                frequency: "57.9%",
                leetcodeNum: 91,
                topics: ["String", "Dynamic Programming"],
                acceptance: "36.5%",
              },
              {
                id: 10,
                name: "Regular Expression Matching",
                difficulty: "Hard",
                frequency: "54.9%",
                leetcodeNum: 10,
                topics: ["String", "Dynamic Programming", "Recursion"],
                acceptance: "29.3%",
              },
            ],
          },
          {
            category: "Graph & BFS/DFS",
            questions: [
              {
                id: 200,
                name: "Number of Islands",
                difficulty: "Medium",
                frequency: "78.0%",
                leetcodeNum: 200,
                topics: ["Array", "DFS", "BFS", "Union Find"],
                acceptance: "59.1%",
              },
              {
                id: 133,
                name: "Clone Graph",
                difficulty: "Medium",
                frequency: "49.5%",
                leetcodeNum: 133,
                topics: ["Hash Table", "BFS", "DFS", "Graph"],
                acceptance: "62.4%",
              },
              {
                id: 127,
                name: "Word Ladder",
                difficulty: "Hard",
                frequency: "44.8%",
                leetcodeNum: 127,
                topics: ["Hash Table", "String", "BFS"],
                acceptance: "39.1%",
              },
              {
                id: 210,
                name: "Course Schedule II",
                difficulty: "Medium",
                frequency: "47.7%",
                leetcodeNum: 210,
                topics: ["Graph", "Topological Sort", "BFS"],
                acceptance: "53.4%",
              },
            ],
          },
          {
            category: "Optimization (Two Pointers & Sliding Window)",
            questions: [
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "60.6%",
                leetcodeNum: 42,
                topics: ["Array", "Two Pointers", "Stack", "Monotonic Stack"],
                acceptance: "65.1%",
              },
              {
                id: 76,
                name: "Minimum Window Substring",
                difficulty: "Hard",
                frequency: "62.9%",
                leetcodeNum: 76,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "45.4%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "49.5%",
                leetcodeNum: 3,
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "49.5%",
                leetcodeNum: 11,
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
            ],
          },
          {
            category: "Hard Core Design & Logic",
            questions: [
              {
                id: 84,
                name: "Largest Rectangle in Histogram",
                difficulty: "Hard",
                frequency: "56.5%",
                leetcodeNum: 84,
                topics: ["Array", "Stack", "Monotonic Stack"],
                acceptance: "47.4%",
              },
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "56.5%",
                leetcodeNum: 4,
                topics: ["Array", "Binary Search", "Divide and Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 23,
                name: "Merge k Sorted Lists",
                difficulty: "Hard",
                frequency: "54.9%",
                leetcodeNum: 23,
                topics: ["Linked List", "Heap (Priority Queue)"],
                acceptance: "56.8%",
              },
              {
                id: 138,
                name: "Copy List with Random Pointer",
                difficulty: "Medium",
                frequency: "49.5%",
                leetcodeNum: 138,
                topics: ["Hash Table", "Linked List"],
                acceptance: "60.5%",
              },
            ],
          },
        ],
      },
      {
        name: "Flipkart",
        logo: Flipkart,
        color: "#2874F0",
        totalQuestions: 101,
        frequency: "Very High",
        categories: [
          {
            category: "Advanced Grid & Graph",
            questions: [
              {
                id: 863,
                name: "Shortest Bridge",
                difficulty: "Medium",
                frequency: "97.5%",
                leetcodeNum: 934,
                topics: ["Array", "DFS", "BFS", "Matrix"],
                acceptance: "58.6%",
              },
              {
                id: 200,
                name: "Number of Islands",
                difficulty: "Medium",
                frequency: "64.2%",
                leetcodeNum: 200,
                topics: ["DFS", "BFS", "Union Find"],
                acceptance: "62.3%",
              },
              {
                id: 994,
                name: "Rotting Oranges",
                difficulty: "Medium",
                frequency: "64.2%",
                leetcodeNum: 994,
                topics: ["BFS", "Matrix"],
                acceptance: "56.6%",
              },
              {
                id: 207,
                name: "Course Schedule",
                difficulty: "Medium",
                frequency: "64.2%",
                leetcodeNum: 207,
                topics: ["Graph", "Topological Sort"],
                acceptance: "49.2%",
              },
              {
                id: 2097,
                name: "Minimum Cost to Reach City With Discounts",
                difficulty: "Medium",
                frequency: "81.5%",
                leetcodeNum: 2097,
                topics: ["Graph", "Shortest Path", "Dijkstra"],
                acceptance: "59.9%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Optimization",
            questions: [
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "94.7%",
                leetcodeNum: 42,
                topics: ["Two Pointers", "DP", "Stack"],
                acceptance: "65.1%",
              },
              {
                id: 1235,
                name: "Maximum Profit in Job Scheduling",
                difficulty: "Hard",
                frequency: "64.2%",
                leetcodeNum: 1235,
                topics: ["DP", "Binary Search", "Sorting"],
                acceptance: "54.4%",
              },
              {
                id: 1463,
                name: "Cherry Pickup II",
                difficulty: "Hard",
                frequency: "72.6%",
                leetcodeNum: 1463,
                topics: ["DP", "Matrix"],
                acceptance: "71.9%",
              },
              {
                id: 124,
                name: "Binary Tree Maximum Path Sum",
                difficulty: "Hard",
                frequency: "75.9%",
                leetcodeNum: 124,
                topics: ["Tree", "DFS", "DP"],
                acceptance: "41.2%",
              },
              {
                id: 198,
                name: "House Robber",
                difficulty: "Medium",
                frequency: "51.6%",
                leetcodeNum: 198,
                topics: ["DP"],
                acceptance: "52.3%",
              },
            ],
          },
          {
            category: "Sliding Window & Binary Search",
            questions: [
              {
                id: 632,
                name: "Smallest Range Covering Elements from K Lists",
                difficulty: "Hard",
                frequency: "100.0%",
                leetcodeNum: 632,
                topics: ["Heap", "Sliding Window", "Greedy"],
                acceptance: "69.7%",
              },
              {
                id: 1423,
                name: "Maximum Points You Can Obtain from Cards",
                difficulty: "Medium",
                frequency: "93.2%",
                leetcodeNum: 1423,
                topics: ["Sliding Window", "Prefix Sum"],
                acceptance: "55.6%",
              },
              {
                id: 1011,
                name: "Capacity To Ship Packages Within D Days",
                difficulty: "Medium",
                frequency: "81.5%",
                leetcodeNum: 1011,
                topics: ["Binary Search"],
                acceptance: "72.1%",
              },
              {
                id: 875,
                name: "Koko Eating Bananas",
                difficulty: "Medium",
                frequency: "75.9%",
                leetcodeNum: 875,
                topics: ["Binary Search"],
                acceptance: "49.1%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "75.9%",
                leetcodeNum: 3,
                topics: ["Sliding Window", "Hash Table"],
                acceptance: "36.9%",
              },
            ],
          },
          {
            category: "Greedy & Heaps",
            questions: [
              {
                id: 134,
                name: "Gas Station",
                difficulty: "Medium",
                frequency: "64.2%",
                leetcodeNum: 134,
                topics: ["Greedy"],
                acceptance: "46.4%",
              },
              {
                id: 135,
                name: "Candy",
                difficulty: "Hard",
                frequency: "51.6%",
                leetcodeNum: 135,
                topics: ["Greedy"],
                acceptance: "46.7%",
              },
              {
                id: 1383,
                name: "Maximum Performance of a Team",
                difficulty: "Hard",
                frequency: "51.6%",
                leetcodeNum: 1383,
                topics: ["Heap", "Sorting", "Greedy"],
                acceptance: "47.5%",
              },
              {
                id: 630,
                name: "Course Schedule III",
                difficulty: "Hard",
                frequency: "41.7%",
                leetcodeNum: 630,
                topics: ["Greedy", "Heap"],
                acceptance: "40.7%",
              },
            ],
          },
        ],
      },
      {
        name: "PayPal",
        logo: PayPal,
        color: "#003087",
        totalQuestions: 101,
        frequency: "Very High",
        categories: [
          {
            category: "Design & Data Structures",
            questions: [
              {
                id: 6,
                name: "Zigzag Conversion",
                difficulty: "Medium",
                frequency: "100.0%",
                topics: ["String"],
                acceptance: "51.6%",
              },
              {
                id: 146,
                name: "LRU Cache",
                difficulty: "Medium",
                frequency: "79.2%",
                topics: [
                  "Hash Table",
                  "Linked List",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "45.2%",
              },
              {
                id: 460,
                name: "LFU Cache",
                difficulty: "Hard",
                frequency: "46.7%",
                topics: ["Hash Table", "Linked List", "Design"],
                acceptance: "46.6%",
              },
              {
                id: 295,
                name: "Find Median from Data Stream",
                difficulty: "Hard",
                frequency: "37.9%",
                topics: ["Two Pointers", "Design", "Heap"],
                acceptance: "53.3%",
              },
              {
                id: 155,
                name: "Min Stack",
                difficulty: "Medium",
                frequency: "37.9%",
                topics: ["Stack", "Design"],
                acceptance: "56.4%",
              },
              {
                id: 355,
                name: "Design Twitter",
                difficulty: "Medium",
                frequency: "37.9%",
                topics: ["Hash Table", "Linked List", "Design", "Heap"],
                acceptance: "42.6%",
              },
              {
                id: 895,
                name: "Maximum Frequency Stack",
                difficulty: "Hard",
                frequency: "37.9%",
                topics: ["Hash Table", "Stack", "Design"],
                acceptance: "66.2%",
              },
              {
                id: 706,
                name: "Design HashMap",
                difficulty: "Easy",
                frequency: "47.7%",
                topics: ["Array", "Hash Table", "Design"],
                acceptance: "65.9%",
              },
            ],
          },
          {
            category: "Arrays & Two Pointers",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "71.0%",
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "46.7%",
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "37.9%",
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
              {
                id: 88,
                name: "Merge Sorted Array",
                difficulty: "Easy",
                frequency: "62.0%",
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "52.9%",
              },
              {
                id: 75,
                name: "Sort Colors",
                difficulty: "Medium",
                frequency: "53.0%",
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "67.6%",
              },
              {
                id: 283,
                name: "Move Zeroes",
                difficulty: "Easy",
                frequency: "67.4%",
                topics: ["Array", "Two Pointers"],
                acceptance: "62.8%",
              },
              {
                id: 977,
                name: "Squares of a Sorted Array",
                difficulty: "Easy",
                frequency: "37.9%",
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "73.2%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Greedy",
            questions: [
              {
                id: 121,
                name: "Best Time to Buy and Sell Stock",
                difficulty: "Easy",
                frequency: "73.4%",
                topics: ["Array", "Dynamic Programming"],
                acceptance: "55.3%",
              },
              {
                id: 322,
                name: "Coin Change",
                difficulty: "Medium",
                frequency: "68.4%",
                topics: ["Array", "Dynamic Programming", "BFS"],
                acceptance: "46.5%",
              },
              {
                id: 198,
                name: "House Robber",
                difficulty: "Medium",
                frequency: "53.0%",
                topics: ["Array", "Dynamic Programming"],
                acceptance: "52.3%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "46.7%",
                topics: ["Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 55,
                name: "Jump Game",
                difficulty: "Medium",
                frequency: "46.7%",
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "39.5%",
              },
              {
                id: 300,
                name: "Longest Increasing Subsequence",
                difficulty: "Medium",
                frequency: "71.0%",
                topics: ["Array", "Binary Search", "DP"],
                acceptance: "57.8%",
              },
              {
                id: 221,
                name: "Maximal Square",
                difficulty: "Medium",
                frequency: "71.0%",
                topics: ["Array", "DP", "Matrix"],
                acceptance: "48.8%",
              },
            ],
          },
          {
            category: "Binary Search & Math",
            questions: [
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "46.7%",
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 4,
                name: "Median of Two Sorted Arrays",
                difficulty: "Hard",
                frequency: "46.7%",
                topics: ["Array", "Binary Search", "Divide & Conquer"],
                acceptance: "43.8%",
              },
              {
                id: 875,
                name: "Koko Eating Bananas",
                difficulty: "Medium",
                frequency: "46.7%",
                topics: ["Array", "Binary Search"],
                acceptance: "49.1%",
              },
              {
                id: 162,
                name: "Find Peak Element",
                difficulty: "Medium",
                frequency: "37.9%",
                topics: ["Array", "Binary Search"],
                acceptance: "46.5%",
              },
              {
                id: 7,
                name: "Reverse Integer",
                difficulty: "Medium",
                frequency: "47.7%",
                topics: ["Math"],
                acceptance: "30.3%",
              },
              {
                id: 202,
                name: "Happy Number",
                difficulty: "Easy",
                frequency: "46.7%",
                topics: ["Hash Table", "Math", "Two Pointers"],
                acceptance: "58.1%",
              },
            ],
          },
          {
            category: "Graphs & BFS/DFS",
            questions: [
              {
                id: 200,
                name: "Number of Islands",
                difficulty: "Medium",
                frequency: "75.5%",
                topics: ["Array", "DFS", "BFS", "Union Find"],
                acceptance: "62.3%",
              },
              {
                id: 207,
                name: "Course Schedule",
                difficulty: "Medium",
                frequency: "37.9%",
                topics: ["Graph", "Topological Sort"],
                acceptance: "49.2%",
              },
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "75.5%",
                topics: ["Array", "Backtracking", "DFS"],
                acceptance: "45.3%",
              },
              {
                id: 841,
                name: "Keys and Rooms",
                difficulty: "Medium",
                frequency: "47.7%",
                topics: ["DFS", "BFS", "Graph"],
                acceptance: "74.7%",
              },
            ],
          },
        ],
      },
      {
        name: "Paytm",
        logo: Paytm,
        color: "#002E6E",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Strings & Stack (Message Processing)",
            questions: [
              {
                id: 1047,
                name: "Remove All Adjacent Duplicates In String",
                difficulty: "Easy",
                frequency: "100.0%",
                topics: ["String", "Stack"],
                acceptance: "71.6%",
              },
              {
                id: 155,
                name: "Min Stack",
                difficulty: "Medium",
                frequency: "94.9%",
                topics: ["Stack", "Design"],
                acceptance: "56.4%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "74.6%",
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 151,
                name: "Reverse Words in a String",
                difficulty: "Medium",
                frequency: "67.3%",
                topics: ["Two Pointers", "String"],
                acceptance: "51.9%",
              },
              {
                id: 443,
                name: "String Compression",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Two Pointers", "String"],
                acceptance: "58.1%",
              },
              {
                id: 907,
                name: "Sum of Subarray Minimums",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: [
                  "Array",
                  "Dynamic Programming",
                  "Stack",
                  "Monotonic Stack",
                ],
                acceptance: "37.6%",
              },
            ],
          },
          {
            category: "Financial Algorithms (Greedy & DP)",
            questions: [
              {
                id: 2439,
                name: "Minimize Maximum of Array",
                difficulty: "Medium",
                frequency: "97.6%",
                topics: [
                  "Array",
                  "Binary Search",
                  "Dynamic Programming",
                  "Greedy",
                ],
                acceptance: "46.4%",
              },
              {
                id: 122,
                name: "Best Time to Buy and Sell Stock II",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "69.5%",
              },
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "57.2%",
                topics: ["Array", "Two Pointers", "DP", "Stack"],
                acceptance: "65.1%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "67.3%",
                topics: ["Two Pointers", "String", "DP"],
                acceptance: "35.8%",
              },
              {
                id: 179,
                name: "Largest Number",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "String", "Greedy", "Sorting"],
                acceptance: "41.3%",
              },
            ],
          },
          {
            category: "Pointers & Arrays",
            questions: [
              {
                id: 11,
                name: "Container With Most Water",
                difficulty: "Medium",
                frequency: "74.6%",
                topics: ["Array", "Two Pointers", "Greedy"],
                acceptance: "57.8%",
              },
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "67.3%",
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 15,
                name: "3Sum",
                difficulty: "Medium",
                frequency: "67.3%",
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "37.1%",
              },
              {
                id: 31,
                name: "Next Permutation",
                difficulty: "Medium",
                frequency: "67.3%",
                topics: ["Array", "Two Pointers"],
                acceptance: "43.1%",
              },
              {
                id: 238,
                name: "Product of Array Except Self",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "Prefix Sum"],
                acceptance: "67.8%",
              },
              {
                id: 2908,
                name: "Find Indices With Index and Value Difference I",
                difficulty: "Easy",
                frequency: "97.6%",
                topics: ["Array", "Two Pointers"],
                acceptance: "60.3%",
              },
            ],
          },
          {
            category: "Sliding Window & Hashing",
            questions: [
              {
                id: 2401,
                name: "Longest Nice Subarray",
                difficulty: "Medium",
                frequency: "97.6%",
                topics: ["Array", "Bit Manipulation", "Sliding Window"],
                acceptance: "64.8%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "67.3%",
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 128,
                name: "Longest Consecutive Sequence",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "Hash Table", "Union Find"],
                acceptance: "47.0%",
              },
            ],
          },
          {
            category: "Linked Lists & Graphs",
            questions: [
              {
                id: 142,
                name: "Linked List Cycle II",
                difficulty: "Medium",
                frequency: "80.2%",
                topics: ["Hash Table", "Linked List", "Two Pointers"],
                acceptance: "54.9%",
              },
              {
                id: 206,
                name: "Reverse Linked List",
                difficulty: "Easy",
                frequency: "57.2%",
                topics: ["Linked List", "Recursion"],
                acceptance: "79.2%",
              },
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "Backtracking", "DFS"],
                acceptance: "45.3%",
              },
            ],
          },
          {
            category: "Binary Search",
            questions: [
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 153,
                name: "Find Minimum in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "57.2%",
                topics: ["Array", "Binary Search"],
                acceptance: "52.6%",
              },
            ],
          },
        ],
      },
      {
        name: "Juspay",
        logo: Juspay,
        color: "#94C11F",
        totalQuestions: 101,
        frequency: "Extremely High",
        categories: [
          {
            category: "Advanced Graph Theory (The Core)",
            questions: [
              {
                id: 2360,
                name: "Longest Cycle in a Graph",
                difficulty: "Hard",
                frequency: "100.0%",
                topics: ["DFS", "BFS", "Graph", "Topological Sort"],
                acceptance: "49.8%",
              },
              {
                id: 2359,
                name: "Find Closest Node to Given Two Nodes",
                difficulty: "Medium",
                frequency: "95.8%",
                topics: ["DFS", "BFS", "Graph"],
                acceptance: "52.8%",
              },
              {
                id: 2374,
                name: "Node With Highest Edge Score",
                difficulty: "Medium",
                frequency: "90.6%",
                topics: ["Hash Table", "Graph"],
                acceptance: "48.4%",
              },
              {
                id: 1857,
                name: "Largest Color Value in a Directed Graph",
                difficulty: "Hard",
                frequency: "62.6%",
                topics: ["Hash Table", "DP", "Graph", "Topological Sort"],
                acceptance: "57.7%",
              },
              {
                id: 127,
                name: "Word Ladder",
                difficulty: "Hard",
                frequency: "53.9%",
                topics: ["Hash Table", "String", "BFS"],
                acceptance: "42.8%",
              },
            ],
          },
          {
            category: "Tree Operations & Traversals",
            questions: [
              {
                id: 1993,
                name: "Operations on Tree",
                difficulty: "Medium",
                frequency: "100.0%",
                topics: ["Array", "Tree", "DFS", "Design"],
                acceptance: "43.4%",
              },
              {
                id: 101,
                name: "Shortest Path in a Weighted Tree",
                difficulty: "Hard",
                frequency: "62.6%",
                topics: ["Array", "Tree", "DFS", "Segment Tree"],
                acceptance: "32.2%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Matrix",
            questions: [
              {
                id: 3202,
                name: "Longest Subsequence With Decreasing Adjacent Difference",
                difficulty: "Medium",
                frequency: "88.5%",
                topics: ["Array", "Dynamic Programming"],
                acceptance: "14.7%",
              },
              {
                id: 73,
                name: "Set Matrix Zeroes",
                difficulty: "Medium",
                frequency: "68.8%",
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "60.7%",
              },
            ],
          },
          {
            category: "Sliding Window & Hashing",
            questions: [
              {
                id: 239,
                name: "Sliding Window Maximum",
                difficulty: "Hard",
                frequency: "62.6%",
                topics: ["Queue", "Sliding Window", "Monotonic Queue"],
                acceptance: "47.6%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "62.6%",
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
            ],
          },
        ],
        note: "Juspay interviews are heavily centered on Graph algorithms. Expect multiple rounds of 'Find the reachability' and 'Cycle detection' logic.",
      },
      {
        name: "IBM",
        logo: IBM,
        color: "#006699",
        totalQuestions: 101,
        frequency: "Very High",
        categories: [
          {
            category: "Data Processing & Stacks",
            questions: [
              {
                id: 636,
                name: "Exclusive Time of Functions",
                difficulty: "Medium",
                frequency: "100.0%",
                topics: ["Array", "Stack"],
                acceptance: "64.8%",
              },
              {
                id: 20,
                name: "Valid Parentheses",
                difficulty: "Easy",
                frequency: "66.5%",
                topics: ["String", "Stack"],
                acceptance: "42.3%",
              },
              {
                id: 2390,
                name: "Removing Stars From a String",
                difficulty: "Medium",
                frequency: "35.0%",
                topics: ["String", "Stack", "Simulation"],
                acceptance: "78.0%",
              },
              {
                id: 456,
                name: "132 Pattern",
                difficulty: "Medium",
                frequency: "35.0%",
                topics: ["Array", "Binary Search", "Stack", "Monotonic Stack"],
                acceptance: "34.1%",
              },
              {
                id: 844,
                name: "Backspace String Compare",
                difficulty: "Easy",
                frequency: "44.1%",
                topics: ["Two Pointers", "String", "Stack"],
                acceptance: "49.5%",
              },
            ],
          },
          {
            category: "Mathematical Logic & Simulation",
            questions: [
              {
                id: 12,
                name: "Integer to Roman",
                difficulty: "Medium",
                frequency: "92.9%",
                topics: ["Hash Table", "Math", "String"],
                acceptance: "68.6%",
              },
              {
                id: 412,
                name: "Fizz Buzz",
                difficulty: "Easy",
                frequency: "89.9%",
                topics: ["Math", "String", "Simulation"],
                acceptance: "74.4%",
              },
              {
                id: 13,
                name: "Roman to Integer",
                difficulty: "Easy",
                frequency: "88.8%",
                topics: ["Hash Table", "Math", "String"],
                acceptance: "64.9%",
              },
              {
                id: 2520,
                name: "Count the Digits That Divide a Number",
                difficulty: "Easy",
                frequency: "71.7%",
                topics: ["Math"],
                acceptance: "84.3%",
              },
              {
                id: 258,
                name: "The kth Factor of n",
                difficulty: "Medium",
                frequency: "73.9%",
                topics: ["Math", "Number Theory"],
                acceptance: "69.6%",
              },
            ],
          },
          {
            category: "Array Optimization (Prefix Sum & Greedy)",
            questions: [
              {
                id: 2574,
                name: "Minimum Operations to Make All Array Elements Equal",
                difficulty: "Medium",
                frequency: "81.1%",
                topics: ["Array", "Binary Search", "Sorting", "Prefix Sum"],
                acceptance: "36.9%",
              },
              {
                id: 2559,
                name: "Count Vowel Strings in Ranges",
                difficulty: "Medium",
                frequency: "71.7%",
                topics: ["Array", "String", "Prefix Sum"],
                acceptance: "67.9%",
              },
              {
                id: 2433,
                name: "Find The Original Array of Prefix Xor",
                difficulty: "Medium",
                frequency: "44.1%",
                topics: ["Array", "Bit Manipulation"],
                acceptance: "88.1%",
              },
              {
                id: 605,
                name: "Maximum Units on a Truck",
                difficulty: "Easy",
                frequency: "69.3%",
                topics: ["Array", "Greedy", "Sorting"],
                acceptance: "74.3%",
              },
              {
                id: 2571,
                name: "Minimum Operations to Reduce an Integer to 0",
                difficulty: "Medium",
                frequency: "35.0%",
                topics: ["DP", "Greedy", "Bit Manipulation"],
                acceptance: "57.3%",
              },
            ],
          },
          {
            category: "Sorting & Intervals",
            questions: [
              {
                id: 2580,
                name: "Count Ways to Group Overlapping Ranges",
                difficulty: "Medium",
                frequency: "85.3%",
                topics: ["Array", "Sorting"],
                acceptance: "38.1%",
              },
              {
                id: 253,
                name: "Meeting Rooms II",
                difficulty: "Medium",
                frequency: "82.6%",
                topics: ["Array", "Two Pointers", "Greedy", "Sorting", "Heap"],
                acceptance: "52.1%",
              },
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "71.7%",
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 2512,
                name: "Sort the Students by Their Kth Score",
                difficulty: "Medium",
                frequency: "71.7%",
                topics: ["Array", "Sorting", "Matrix"],
                acceptance: "85.6%",
              },
            ],
          },
          {
            category: "Advanced Search & DP",
            questions: [
              {
                id: 146,
                name: "LRU Cache",
                difficulty: "Medium",
                frequency: "35.0%",
                topics: ["Hash Table", "Linked List", "Design"],
                acceptance: "45.2%",
              },
              {
                id: 300,
                name: "Longest Increasing Subsequence",
                difficulty: "Medium",
                frequency: "44.1%",
                topics: ["Array", "Binary Search", "DP"],
                acceptance: "57.8%",
              },
              {
                id: 98,
                name: "Validate Binary Search Tree",
                difficulty: "Medium",
                frequency: "50.6%",
                topics: ["Tree", "DFS", "BST"],
                acceptance: "34.4%",
              },
              {
                id: 70,
                name: "Climbing Stairs",
                difficulty: "Easy",
                frequency: "50.6%",
                topics: ["Math", "DP"],
                acceptance: "53.5%",
              },
            ],
          },
        ],
      },
      {
        name: "Oracle",
        logo: Oracle,
        color: "#F80000",
        totalQuestions: 101,
        frequency: "High",
        categories: [
          {
            category: "Data Management & Design",
            questions: [
              {
                id: 146,
                name: "LRU Cache",
                difficulty: "Medium",
                frequency: "100.0%",
                topics: [
                  "Hash Table",
                  "Linked List",
                  "Design",
                  "Doubly-Linked List",
                ],
                acceptance: "45.2%",
              },
              {
                id: 155,
                name: "Min Stack",
                difficulty: "Medium",
                frequency: "54.7%",
                topics: ["Stack", "Design"],
                acceptance: "56.4%",
              },
              {
                id: 295,
                name: "Find Median from Data Stream",
                difficulty: "Hard",
                frequency: "37.9%",
                topics: ["Two Pointers", "Design", "Heap", "Data Stream"],
                acceptance: "53.3%",
              },
              {
                id: 232,
                name: "Implement Queue using Stacks",
                difficulty: "Easy",
                frequency: "33.6%",
                topics: ["Stack", "Design", "Queue"],
                acceptance: "68.1%",
              },
              {
                id: 37,
                name: "Sudoku Solver",
                difficulty: "Hard",
                frequency: "39.7%",
                topics: ["Array", "Hash Table", "Backtracking", "Matrix"],
                acceptance: "63.9%",
              },
            ],
          },
          {
            category: "Grid & Graph (Cloud Infrastructure)",
            questions: [
              {
                id: 200,
                name: "Number of Islands",
                difficulty: "Medium",
                frequency: "89.5%",
                topics: ["Array", "DFS", "BFS", "Union Find", "Matrix"],
                acceptance: "62.3%",
              },
              {
                id: 207,
                name: "Course Schedule",
                difficulty: "Medium",
                frequency: "61.6%",
                topics: ["DFS", "BFS", "Graph", "Topological Sort"],
                acceptance: "49.2%",
              },
              {
                id: 210,
                name: "Course Schedule II",
                difficulty: "Medium",
                frequency: "59.6%",
                topics: ["DFS", "BFS", "Graph", "Topological Sort"],
                acceptance: "53.4%",
              },
              {
                id: 54,
                name: "Spiral Matrix",
                difficulty: "Medium",
                frequency: "66.9%",
                topics: ["Array", "Matrix", "Simulation"],
                acceptance: "53.9%",
              },
              {
                id: 73,
                name: "Set Matrix Zeroes",
                difficulty: "Medium",
                frequency: "54.7%",
                topics: ["Array", "Hash Table", "Matrix"],
                acceptance: "60.7%",
              },
              {
                id: 79,
                name: "Word Search",
                difficulty: "Medium",
                frequency: "57.3%",
                topics: ["Array", "String", "Backtracking", "Matrix"],
                acceptance: "45.3%",
              },
            ],
          },
          {
            category: "Searching & Sorting",
            questions: [
              {
                id: 56,
                name: "Merge Intervals",
                difficulty: "Medium",
                frequency: "85.1%",
                topics: ["Array", "Sorting"],
                acceptance: "49.4%",
              },
              {
                id: 33,
                name: "Search in Rotated Sorted Array",
                difficulty: "Medium",
                frequency: "71.2%",
                topics: ["Array", "Binary Search"],
                acceptance: "42.8%",
              },
              {
                id: 240,
                name: "Search a 2D Matrix II",
                difficulty: "Medium",
                frequency: "59.6%",
                topics: [
                  "Array",
                  "Binary Search",
                  "Divide and Conquer",
                  "Matrix",
                ],
                acceptance: "55.2%",
              },
              {
                id: 215,
                name: "Kth Largest Element in an Array",
                difficulty: "Medium",
                frequency: "54.7%",
                topics: [
                  "Array",
                  "Divide and Conquer",
                  "Sorting",
                  "Heap",
                  "Quickselect",
                ],
                acceptance: "68.0%",
              },
              {
                id: 75,
                name: "Sort Colors",
                difficulty: "Medium",
                frequency: "51.8%",
                topics: ["Array", "Two Pointers", "Sorting"],
                acceptance: "67.6%",
              },
              {
                id: 34,
                name: "Find First and Last Position of Element in Sorted Array",
                difficulty: "Medium",
                frequency: "51.8%",
                topics: ["Array", "Binary Search"],
                acceptance: "46.8%",
              },
            ],
          },
          {
            category: "Dynamic Programming & Optimization",
            questions: [
              {
                id: 42,
                name: "Trapping Rain Water",
                difficulty: "Hard",
                frequency: "66.9%",
                topics: [
                  "Array",
                  "Two Pointers",
                  "Dynamic Programming",
                  "Stack",
                ],
                acceptance: "65.1%",
              },
              {
                id: 55,
                name: "Jump Game",
                difficulty: "Medium",
                frequency: "61.6%",
                topics: ["Array", "Dynamic Programming", "Greedy"],
                acceptance: "39.5%",
              },
              {
                id: 121,
                name: "Best Time to Buy and Sell Stock",
                difficulty: "Easy",
                frequency: "61.6%",
                topics: ["Array", "Dynamic Programming"],
                acceptance: "55.3%",
              },
              {
                id: 53,
                name: "Maximum Subarray",
                difficulty: "Medium",
                frequency: "57.3%",
                topics: ["Array", "Divide and Conquer", "Dynamic Programming"],
                acceptance: "52.1%",
              },
              {
                id: 221,
                name: "Maximal Square",
                difficulty: "Medium",
                frequency: "51.8%",
                topics: ["Array", "Dynamic Programming", "Matrix"],
                acceptance: "48.8%",
              },
            ],
          },
          {
            category: "Strings & Hashing",
            questions: [
              {
                id: 1,
                name: "Two Sum",
                difficulty: "Easy",
                frequency: "88.3%",
                topics: ["Array", "Hash Table"],
                acceptance: "55.8%",
              },
              {
                id: 3,
                name: "Longest Substring Without Repeating Characters",
                difficulty: "Medium",
                frequency: "85.8%",
                topics: ["Hash Table", "String", "Sliding Window"],
                acceptance: "36.9%",
              },
              {
                id: 49,
                name: "Group Anagrams",
                difficulty: "Medium",
                frequency: "82.2%",
                topics: ["Array", "Hash Table", "String", "Sorting"],
                acceptance: "70.9%",
              },
              {
                id: 5,
                name: "Longest Palindromic Substring",
                difficulty: "Medium",
                frequency: "77.8%",
                topics: ["Two Pointers", "String", "Dynamic Programming"],
                acceptance: "35.8%",
              },
            ],
          },
        ],
      },
      {
        name: "D.E. Shaw",
        logo: DE,
        color: "#9E1B32",
        totalQuestions: 101,
        frequency: "Extremely High",
        categories: [
          {
            category: "Advanced Tree & Graph Optimization",
            questions: [
              {
                id: 968,
                name: "Binary Tree Cameras",
                difficulty: "Hard",
                frequency: "100.0%",
                topics: ["Dynamic Programming", "Tree", "DFS"],
                acceptance: "47.2%",
              },
              {
                id: 2071,
                name: "Minimum Runes to Add to Cast Spell",
                difficulty: "Hard",
                frequency: "77.4%",
                topics: ["Graph", "Topological Sort", "Union Find"],
                acceptance: "42.6%",
              },
              {
                id: 2940,
                name: "Find Products of Elements of Big Array",
                difficulty: "Hard",
                frequency: "71.7%",
                topics: ["Binary Search", "Bit Manipulation"],
                acceptance: "21.8%",
              },
              {
                id: 124,
                name: "Binary Tree Maximum Path Sum",
                difficulty: "Hard",
                frequency: "30.7%",
                topics: ["DP", "Tree", "DFS"],
                acceptance: "41.2%",
              },
              {
                id: 863,
                name: "All Nodes Distance K in Binary Tree",
                difficulty: "Medium",
                frequency: "39.6%",
                topics: ["BFS", "Tree", "Hash Table"],
                acceptance: "66.4%",
              },
            ],
          },
          {
            category: "High-Complexity Dynamic Programming",
            questions: [
              {
                id: 2920,
                name: "Maximum Points After Collecting Coins From All Nodes",
                difficulty: "Hard",
                frequency: "79.6%",
                topics: ["DP", "Tree", "Memoization"],
                acceptance: "35.8%",
              },
              {
                id: 3082,
                name: "Find the Sum of the Power of All Subsequences",
                difficulty: "Hard",
                frequency: "77.4%",
                topics: ["Array", "Dynamic Programming"],
                acceptance: "36.6%",
              },
              {
                id: 1473,
                name: "Paint House III",
                difficulty: "Hard",
                frequency: "73.4%",
                topics: ["Array", "Dynamic Programming"],
                acceptance: "61.0%",
              },
              {
                id: 221,
                name: "Maximal Square",
                difficulty: "Medium",
                frequency: "55.7%",
                topics: ["Array", "DP", "Matrix"],
                acceptance: "48.8%",
              },
              {
                id: 329,
                name: "Longest Increasing Path in a Matrix",
                difficulty: "Hard",
                frequency: "39.6%",
                topics: ["DP", "DFS", "Graph"],
                acceptance: "55.3%",
              },
            ],
          },
          {
            category: "Greedy & Efficient Searching",
            questions: [
              {
                id: 2542,
                name: "Maximum Subsequence Score",
                difficulty: "Medium",
                frequency: "81.7%",
                topics: ["Greedy", "Sorting", "Heap"],
                acceptance: "54.3%",
              },
              {
                id: 2101,
                name: "Minimum Cost Walk in Weighted Graph",
                difficulty: "Hard",
                frequency: "79.6%",
                topics: ["Graph", "Union Find", "Bit Manipulation"],
                acceptance: "68.5%",
              },
              {
                id: 871,
                name: "Minimum Number of Refueling Stops",
                difficulty: "Hard",
                frequency: "65.3%",
                topics: ["Greedy", "Heap", "DP"],
                acceptance: "40.6%",
              },
              {
                id: 1383,
                name: "Maximum Performance of a Team",
                difficulty: "Hard",
                frequency: "49.0%",
                topics: ["Greedy", "Heap", "Sorting"],
                acceptance: "47.5%",
              },
              {
                id: 1011,
                name: "Capacity To Ship Packages Within D Days",
                difficulty: "Medium",
                frequency: "49.0%",
                topics: ["Binary Search"],
                acceptance: "72.1%",
              },
            ],
          },
          {
            category: "Mathematical Logic & Simulation",
            questions: [
              {
                id: 1359,
                name: "Count All Valid Pickup and Delivery Options",
                difficulty: "Hard",
                frequency: "79.6%",
                topics: ["Math", "Dynamic Programming", "Combinatorics"],
                acceptance: "63.9%",
              },
              {
                id: 2045,
                name: "Second Minimum Time to Reach Destination",
                difficulty: "Hard",
                frequency: "79.6%",
                topics: ["BFS", "Graph", "Shortest Path"],
                acceptance: "34.1%",
              },
              {
                id: 12,
                name: "Integer to Roman",
                difficulty: "Medium",
                frequency: "92.9%",
                topics: ["Math", "String"],
                acceptance: "68.6%",
              },
              {
                id: 224,
                name: "Basic Calculator",
                difficulty: "Hard",
                frequency: "39.6%",
                topics: ["Math", "Stack", "Recursion"],
                acceptance: "45.6%",
              },
            ],
          },
          {
            category: "Advanced Matrix & Subarray",
            questions: [
              {
                id: 239,
                name: "Sliding Window Maximum",
                difficulty: "Hard",
                frequency: "61.0%",
                topics: ["Sliding Window", "Monotonic Queue"],
                acceptance: "47.6%",
              },
              {
                id: 410,
                name: "Split Array Largest Sum",
                difficulty: "Hard",
                frequency: "49.0%",
                topics: ["Binary Search", "Greedy"],
                acceptance: "58.1%",
              },
              {
                id: 84,
                name: "Largest Rectangle in Histogram",
                difficulty: "Hard",
                frequency: "39.6%",
                topics: ["Array", "Stack", "Monotonic Stack"],
                acceptance: "47.4%",
              },
              {
                id: 85,
                name: "Maximal Rectangle",
                difficulty: "Hard",
                frequency: "39.6%",
                topics: ["DP", "Stack", "Matrix"],
                acceptance: "53.7%",
              },
            ],
          },
        ],
      },
    ],
  },
};
