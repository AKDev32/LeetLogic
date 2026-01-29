export const cheatsheetData = {
  bigO: {
    title: "Big O Complexity",
    icon: "Activity",
    sections: [
      {
        title: "Time Complexity",
        items: [
          { name: "O(1)", description: "Constant", examples: ["Array access", "Hash table lookup"], color: "#10b981" },
          { name: "O(log n)", description: "Logarithmic", examples: ["Binary search", "Balanced BST operations"], color: "#3b82f6" },
          { name: "O(n)", description: "Linear", examples: ["Array traversal", "Single loop"], color: "#8b5cf6" },
          { name: "O(n log n)", description: "Linearithmic", examples: ["Merge sort", "Quick sort", "Heap sort"], color: "#f59e0b" },
          { name: "O(n²)", description: "Quadratic", examples: ["Bubble sort", "Nested loops"], color: "#ef4444" },
          { name: "O(2ⁿ)", description: "Exponential", examples: ["Recursive fibonacci", "Subset generation"], color: "#dc2626" },
          { name: "O(n!)", description: "Factorial", examples: ["Permutations", "Traveling salesman"], color: "#991b1b" }
        ]
      },
      {
        title: "Space Complexity",
        items: [
          { name: "O(1)", description: "Constant space", examples: ["Variables only"] },
          { name: "O(n)", description: "Linear space", examples: ["Array of size n", "Hash map"] },
          { name: "O(n²)", description: "Quadratic space", examples: ["2D matrix"] }
        ]
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
          deletion: "O(n)"
        },
        techniques: ["Two pointers", "Sliding window", "Prefix sum", "Kadane's algorithm"],
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
}`
        }
      },
      {
        title: "Linked Lists",
        complexity: {
          access: "O(n)",
          search: "O(n)",
          insertion: "O(1)",
          deletion: "O(1)"
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
    return prev`
        }
      },
      {
        title: "Stacks & Queues",
        complexity: {
          push: "O(1)",
          pop: "O(1)",
          peek: "O(1)"
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
};`
        }
      },
      {
        title: "Hash Tables",
        complexity: {
          search: "O(1) avg",
          insert: "O(1) avg",
          delete: "O(1) avg"
        },
        techniques: ["Frequency counting", "Two sum pattern", "Anagram detection"],
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
    return []`
        }
      },
      {
        title: "Trees & Graphs",
        complexity: {
          search: "O(log n) - O(n)",
          insert: "O(log n) - O(n)",
          delete: "O(log n) - O(n)"
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
};`
        }
      },
      {
        title: "Heaps (Priority Queue)",
        complexity: {
          insert: "O(log n)",
          deleteMin: "O(log n)",
          peek: "O(1)"
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
}`
        }
      },
      {
        title: "Tries",
        complexity: {
          search: "O(m)",
          insert: "O(m)",
          delete: "O(m)"
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
}`
        }
      }
    ]
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
};`
            }
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
};`
            }
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
};`
            }
          }
        ]
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
};`
            }
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
};`
            }
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
};`
            }
          }
        ]
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
};`
            }
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
};`
            }
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
};`
            }
          }
        ]
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
};`
            }
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
};`
            }
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
};`
            }
          }
        ]
      }
    ]
  },
  
  patterns: {
    title: "Common Patterns",
    icon: "Lightbulb",
    sections: [
      {
        title: "Two Pointers",
        description: "Use two pointers to traverse array/string from different positions",
        useCases: ["Sorted arrays", "Palindromes", "Pair finding", "Removing duplicates"],
        examples: ["Two Sum II", "Container With Most Water", "3Sum", "Remove Duplicates"]
      },
      {
        title: "Sliding Window",
        description: "Maintain a window that slides through array/string",
        useCases: ["Subarray problems", "Substring problems", "Finding patterns"],
        examples: ["Longest Substring Without Repeating", "Maximum Sum Subarray", "Minimum Window Substring"]
      },
      {
        title: "Fast & Slow Pointers",
        description: "Two pointers moving at different speeds",
        useCases: ["Cycle detection", "Finding middle", "Palindrome linked list"],
        examples: ["Linked List Cycle", "Happy Number", "Find Duplicate Number"]
      },
      {
        title: "Binary Search",
        description: "Divide search space in half repeatedly",
        useCases: ["Sorted arrays", "Search space reduction", "Finding boundaries"],
        examples: ["Search Insert Position", "Find First and Last Position", "Search in Rotated Array"]
      },
      {
        title: "Top K Elements",
        description: "Use heap to track K largest/smallest elements",
        useCases: ["Finding K largest", "K closest points", "Frequency-based problems"],
        examples: ["Kth Largest Element", "Top K Frequent Elements", "K Closest Points"]
      },
      {
        title: "Merge Intervals",
        description: "Sort intervals and merge overlapping ones",
        useCases: ["Overlapping intervals", "Scheduling problems", "Range queries"],
        examples: ["Merge Intervals", "Insert Interval", "Meeting Rooms"]
      },
      {
        title: "Modified Binary Search",
        description: "Binary search on rotated/modified arrays",
        useCases: ["Rotated arrays", "Peak finding", "Mountain arrays"],
        examples: ["Search Rotated Array", "Find Peak Element", "Find in Mountain Array"]
      },
      {
        title: "Monotonic Stack",
        description: "Stack maintaining monotonic order",
        useCases: ["Next greater element", "Temperature problems", "Histogram problems"],
        examples: ["Daily Temperatures", "Next Greater Element", "Largest Rectangle in Histogram"]
      }
    ]
  },
  
  tips: {
    title: "Problem-Solving Tips",
    icon: "Brain",
    sections: [
      {
        title: "Before Coding",
        items: [
          "Clarify the problem - ask questions",
          "Think about edge cases",
          "Consider input constraints",
          "Discuss approach before coding",
          "Start with brute force, then optimize"
        ]
      },
      {
        title: "During Coding",
        items: [
          "Use meaningful variable names",
          "Write clean, readable code",
          "Test with examples as you go",
          "Handle edge cases explicitly",
          "Comment complex logic"
        ]
      },
      {
        title: "Optimization",
        items: [
          "Identify bottlenecks in your solution",
          "Can you use a hash table for O(1) lookup?",
          "Can you sort the input first?",
          "Can you use two pointers instead of nested loops?",
          "Is there redundant computation to cache?"
        ]
      },
      {
        title: "Common Edge Cases",
        items: [
          "Empty input ([], \"\", null)",
          "Single element",
          "All elements same",
          "Sorted vs unsorted",
          "Negative numbers",
          "Integer overflow",
          "Duplicate elements"
        ]
      }
    ]
  }
};