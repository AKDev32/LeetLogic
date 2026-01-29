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
          "Start with brute force, then optimize",
        ],
      },
      {
        title: "During Coding",
        items: [
          "Use meaningful variable names",
          "Write clean, readable code",
          "Test with examples as you go",
          "Handle edge cases explicitly",
          "Comment complex logic",
        ],
      },
      {
        title: "Optimization",
        items: [
          "Identify bottlenecks in your solution",
          "Can you use a hash table for O(1) lookup?",
          "Can you sort the input first?",
          "Can you use two pointers instead of nested loops?",
          "Is there redundant computation to cache?",
        ],
      },
      {
        title: "Common Edge Cases",
        items: [
          'Empty input ([], "", null)',
          "Single element",
          "All elements same",
          "Sorted vs unsorted",
          "Negative numbers",
          "Integer overflow",
          "Duplicate elements",
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
        logo: "🔍",
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
        logo: "📦",
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
        logo: "♾️",
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
        logo: "🪟",
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
        logo: "🍎",
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
        logo: "🎬",
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
    ],
  },
};
