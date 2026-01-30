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
    title: "Big O Complexity",
    icon: "Activity",
    sections: [
      {
        title: "Time Complexity",
        items: [
          {
            name: "O(1)",
            description: "Constant",
            examples: ["Array access", "Hash table lookup"],
            color: "#10b981",
          },
          {
            name: "O(log n)",
            description: "Logarithmic",
            examples: ["Binary search", "Balanced BST operations"],
            color: "#3b82f6",
          },
          {
            name: "O(n)",
            description: "Linear",
            examples: ["Array traversal", "Single loop"],
            color: "#8b5cf6",
          },
          {
            name: "O(n log n)",
            description: "Linearithmic",
            examples: ["Merge sort", "Quick sort", "Heap sort"],
            color: "#f59e0b",
          },
          {
            name: "O(n²)",
            description: "Quadratic",
            examples: ["Bubble sort", "Nested loops"],
            color: "#ef4444",
          },
          {
            name: "O(2ⁿ)",
            description: "Exponential",
            examples: ["Recursive fibonacci", "Subset generation"],
            color: "#dc2626",
          },
          {
            name: "O(n!)",
            description: "Factorial",
            examples: ["Permutations", "Traveling salesman"],
            color: "#991b1b",
          },
        ],
      },
      {
        title: "Space Complexity",
        items: [
          {
            name: "O(1)",
            description: "Constant space",
            examples: ["Variables only"],
          },
          {
            name: "O(n)",
            description: "Linear space",
            examples: ["Array of size n", "Hash map"],
          },
          {
            name: "O(n²)",
            description: "Quadratic space",
            examples: ["2D matrix"],
          },
        ],
      },
    ],
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
    title: "Common Patterns",
    icon: "Lightbulb",
    sections: [
      {
        title: "Two Pointers",
        description:
          "Use two pointers to traverse array/string from different positions",
        useCases: [
          "Sorted arrays",
          "Palindromes",
          "Pair finding",
          "Removing duplicates",
        ],
        examples: [
          "Two Sum II",
          "Container With Most Water",
          "3Sum",
          "Remove Duplicates",
        ],
      },
      {
        title: "Sliding Window",
        description: "Maintain a window that slides through array/string",
        useCases: [
          "Subarray problems",
          "Substring problems",
          "Finding patterns",
        ],
        examples: [
          "Longest Substring Without Repeating",
          "Maximum Sum Subarray",
          "Minimum Window Substring",
        ],
      },
      {
        title: "Fast & Slow Pointers",
        description: "Two pointers moving at different speeds",
        useCases: [
          "Cycle detection",
          "Finding middle",
          "Palindrome linked list",
        ],
        examples: [
          "Linked List Cycle",
          "Happy Number",
          "Find Duplicate Number",
        ],
      },
      {
        title: "Binary Search",
        description: "Divide search space in half repeatedly",
        useCases: [
          "Sorted arrays",
          "Search space reduction",
          "Finding boundaries",
        ],
        examples: [
          "Search Insert Position",
          "Find First and Last Position",
          "Search in Rotated Array",
        ],
      },
      {
        title: "Top K Elements",
        description: "Use heap to track K largest/smallest elements",
        useCases: [
          "Finding K largest",
          "K closest points",
          "Frequency-based problems",
        ],
        examples: [
          "Kth Largest Element",
          "Top K Frequent Elements",
          "K Closest Points",
        ],
      },
      {
        title: "Merge Intervals",
        description: "Sort intervals and merge overlapping ones",
        useCases: [
          "Overlapping intervals",
          "Scheduling problems",
          "Range queries",
        ],
        examples: ["Merge Intervals", "Insert Interval", "Meeting Rooms"],
      },
      {
        title: "Modified Binary Search",
        description: "Binary search on rotated/modified arrays",
        useCases: ["Rotated arrays", "Peak finding", "Mountain arrays"],
        examples: [
          "Search Rotated Array",
          "Find Peak Element",
          "Find in Mountain Array",
        ],
      },
      {
        title: "Monotonic Stack",
        description: "Stack maintaining monotonic order",
        useCases: [
          "Next greater element",
          "Temperature problems",
          "Histogram problems",
        ],
        examples: [
          "Daily Temperatures",
          "Next Greater Element",
          "Largest Rectangle in Histogram",
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
