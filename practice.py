## Leetcode 75 with Python Solutions 
## Blind 75 
## Start with easy questions in each section then move to medium in each section then move to hard in each section 


 # ARRAYS & HASHING (0/8)


# 1. Contains Duplicate (easy)

# Question Link: https://leetcode.com/problems/contains-duplicate/
# Video Solution: https://www.youtube.com/watch?v=3OamzN90kPg
# Python Solution: 

# class Solution:
#     def containsDuplicate(self, nums: List[int]) -> bool:
#         hashset = set()

#         for n in nums:
#             if n in hashset:
#                 return True
#             hashset.add(n)
#         return False





# 2. Valid Anagram (easy)

# Question Link https://leetcode.com/problems/valid-anagram/
# Video Solution: https://www.youtube.com/watch?v=9UtInBqnCgA&feature=emb_imp_woyt
# Python Solution:

# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         if len(s) != len(t):
#             return False

#         countS, countT = {}, {}

#         for i in range(len(s)):
#             countS[s[i]] = 1 + countS.get(s[i], 0)
#             countT[t[i]] = 1 + countT.get(t[i], 0)
#         return countS == countT



# 3. Two Sum (easy)

# Question Link https://leetcode.com/problems/two-sum/
# Video Solution: https://www.youtube.com/watch?v=KLlXCFG5TnA
# Python Solution: 

# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         prevMap = {}  # val -> index

#         for i, n in enumerate(nums):
#             diff = target - n
#             if diff in prevMap:
#                 return [prevMap[diff], i]
#             prevMap[n] = i

from curses.ascii import isalnum
from email.policy import default


def twoSum(self, nums: List[int], target: int) -> List[int]:
    prevMap = {} # val : index

    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i
    return



# 4. Group Anagrams (medium)

# Question Link https://leetcode.com/problems/group-anagrams/
# Video Solution: https://www.youtube.com/watch?v=vzdNOK2oB2E
# Python Solution:

# class Solution:
#     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
#         ans = collections.defaultdict(list)

#         for s in strs:
#             count = [0] * 26
#             for c in s:
#                 count[ord(c) - ord('a')] += 1
#             ans[tuple(count)].append(s)
#         return ans.values()



# 5. Top K Frequent Elements (medium)
# Question Link https://leetcode.com/problems/top-k-frequent-elements/
# Video Solution: https://www.youtube.com/watch?v=YPTqKIgVk-k
# Python Solution: 

# class Solution:
#     def topKFrequent(self, nums: List[int], k: int) -> List[int]:
#         count = {}
#         freq = [[] for i in range(len(nums) + 1)]

#         for n in nums:
#             count[n] = 1 + count.get(n, 0)
#         for n, c in count.items():
#             freq[c].append(n)

#         res = []
#         for i in range(len(freq) - 1, 0, -1):
#             for n in freq[i]:
#                 res.append(n)
#                 if len(res) == k:
#                     return res

        # O(n)


# 6. Product of Array Except Self (medium)
# Question Link https://leetcode.com/problems/product-of-array-except-self/
# Video Solution: https://www.youtube.com/watch?v=bNvIQI2wAjk
# Python Solution:

# class Solution:
#     def productExceptSelf(self, nums: List[int]) -> List[int]:
#         res = [1] * (len(nums))

#         prefix = 1
#         for i in range(len(nums)):
#             res[i] = prefix
#             prefix *= nums[i]
#         postfix = 1
#         for i in range(len(nums) - 1, -1, -1):
#             res[i] *= postfix
#             postfix *= nums[i]
#         return res



# 7. Encode and Decode Strings (medium)
# Question Link https://leetcode.com/problems/encode-and-decode-strings/
# Video Solution: https://www.youtube.com/watch?v=B1k_sxOSgv8
# Python Solution: 

# class Solution:
#     """
#     @param: strs: a list of strings
#     @return: encodes a list of strings to a single string.
#     """

#     def encode(self, strs):
#         res = ""
#         for s in strs:
#             res += str(len(s)) + "#" + s
#         return res

#     """
#     @param: str: A string
#     @return: dcodes a single string to a list of strings
#     """

#     def decode(self, str):
#         res, i = [], 0

#         while i < len(str):
#             j = i
#             while str[j] != "#":
#                 j += 1
#             length = int(str[i:j])
#             res.append(str[j + 1 : j + 1 + length])
#             i = j + 1 + length
#         return res



# 8. Longest Consecutive Sequence (medium)
# Question Link https://leetcode.com/problems/longest-consecutive-sequence/
# Video Solution: https://www.youtube.com/watch?v=P6RZZMu_maU
# Python Solution:

# class Solution:
#     def longestConsecutive(self, nums: List[int]) -> int:
#         numSet = set(nums)
#         longest = 0

#         for n in nums:
#             # check if its the start of a sequence
#             if (n - 1) not in numSet:
#                 length = 1
#                 while (n + length) in numSet:
#                     length += 1
#                 longest = max(length, longest)
#         return longest


# TWO POINTERS(0/3)



# 9. Valid Palindrome (easy)
# Question Link https://leetcode.com/problems/valid-palindrome/
# Video Solution: https://www.youtube.com/watch?v=jJXJ16kPFWg
# Python Solution:

class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        
        while l < r:
            while l < r and not self.alphanum(s[l]):
                l += 1
            while l < r and not self.alphanum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

    # Could write own alpha-numeric function
    def alphanum(self, c):
        return (
            ord("A") <= ord(c) <= ord("Z")
            or ord("a") <= ord(c) <= ord("z")
            or ord("0") <= ord(c) <= ord("9")
        )

# 10. 3Sum (Medium)
# Question Link https://leetcode.com/problems/3sum/
# Video Solution: https://www.youtube.com/watch?v=jzZsG8n2R9A
# Python Solution:

# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         res = []
#         nums.sort()

#         for i, a in enumerate(nums):
#             if i > 0 and a == nums[i - 1]:
#                 continue

#             l, r = i + 1, len(nums) - 1
#             while l < r:
#                 threeSum = a + nums[l] + nums[r]
#                 if threeSum > 0:
#                     r -= 1
#                 elif threeSum < 0:
#                     l += 1
#                 else:
#                     res.append([a, nums[l], nums[r]])
#                     l += 1
#                     while nums[l] == nums[l - 1] and l < r:
#                         l += 1
#         return res


# 11. Container with Most Water (Medium)
# Question Link https://leetcode.com/problems/container-with-most-water/
# Video Solution: https://www.youtube.com/watch?v=UuiTKBwPgAo
# Python Solution:

# class Solution:
#     def maxArea(self, height: List[int]) -> int:
#         l, r = 0, len(height) - 1
#         res = 0

#         while l < r:
#             res = max(res, min(height[l], height[r]) * (r - l))
#             if height[l] < height[r]:
#                 l += 1
#             elif height[r] <= height[l]:
#                 r -= 1
#         return res


                                                                  # SLIDING WINDOW (0/4)

# 12. Best Time to Buy & Sell Stock (easy)
# Question Link https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# Video Solution: https://www.youtube.com/watch?v=1pkOgXD63yU
# Python Solution: 

# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         res = 0

#         l = 0
#         for r in range(1, len(prices)):
#             if prices[r] < prices[l]:
#                 l = r
#             res = max(res, prices[r] - prices[l])
#         return res




# 13. Longest Substring Without Repeating Characters (Medium)
# Question Link https://leetcode.com/problems/longest-substring-without-repeating-characters/
# Video Solution: https://www.youtube.com/watch?v=wiGpQwVHdE0
# Python Solution:

# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         charSet = set()
#         l = 0
#         res = 0

#         for r in range(len(s)):
#             while s[r] in charSet:
#                 charSet.remove(s[l])
#                 l += 1
#             charSet.add(s[r])
#             res = max(res, r - l + 1)
#         return res


# 14. Longest Repeating Character Replacement (Medium)
# Question Link https://leetcode.com/problems/longest-repeating-character-replacement/
# Video Solution: https://www.youtube.com/watch?v=gqXU1UyA8pk
# Python Solution: 

# class Solution:
#     def characterReplacement(self, s: str, k: int) -> int:
#         count = {}
#         res = 0

#         l = 0
#         maxf = 0
#         for r in range(len(s)):
#             count[s[r]] = 1 + count.get(s[r], 0)
#             maxf = max(maxf, count[s[r]])

#             if (r - l + 1) - maxf > k:
#                 count[s[l]] -= 1
#                 l += 1

#             res = max(res, r - l + 1)
#         return res

# 15. Minimum Window Substring (Hard)
# Question Link https://leetcode.com/problems/contains-duplicate/
# Video Solution: https://leetcode.com/problems/minimum-window-substring/
# Python Solution: 

# class Solution:
#     def minWindow(self, s: str, t: str) -> str:
#         if t == "":
#             return ""

#         countT, window = {}, {}
#         for c in t:
#             countT[c] = 1 + countT.get(c, 0)

#         have, need = 0, len(countT)
#         res, resLen = [-1, -1], float("infinity")
#         l = 0
#         for r in range(len(s)):
#             c = s[r]
#             window[c] = 1 + window.get(c, 0)

#             if c in countT and window[c] == countT[c]:
#                 have += 1

#             while have == need:
#                 # update our result
#                 if (r - l + 1) < resLen:
#                     res = [l, r]
#                     resLen = r - l + 1
#                 # pop from the left of our window
#                 window[s[l]] -= 1
#                 if s[l] in countT and window[s[l]] < countT[s[l]]:
#                     have -= 1
#                 l += 1


                                                                    # STACK (0/1)

# 16. Valid Parentheses (Easy)
# Question Link https://leetcode.com/problems/valid-parentheses/
# Video Solution: https://www.youtube.com/watch?v=WTzjTskDFMg
# Python Solution: 

# class Solution:
    # def isValid(self, s: str) -> bool:
    #     Map = {")": "(", "]": "[", "}": "{"}
    #     stack = []

    #     for c in s:
    #         if c not in Map:
    #             stack.append(c)
    #             continue
    #         if not stack or stack[-1] != Map[c]:
    #             return False
    #         stack.pop()

    #     return not stack


                                                              # BINARY SEARCH (0/2)

# 17. Search Rotated Sorted Array (Medium)
# Question Link https://leetcode.com/problems/search-in-rotated-sorted-array/
# Video Solution: https://www.youtube.com/watch?v=U8XENwh8Oy8
# Python Solution: 

# class Solution:
    # def search(self, nums: List[int], target: int) -> int:
    #     l, r = 0, len(nums) - 1

    #     while l <= r:
    #         mid = (l + r) // 2
    #         if target == nums[mid]:
    #             return mid

    #         # left sorted portion
    #         if nums[l] <= nums[mid]:
    #             if target > nums[mid] or target < nums[l]:
    #                 l = mid + 1
    #             else:
    #                 r = mid - 1
    #         # right sorted portion
    #         else:
    #             if target < nums[mid] or target > nums[r]:
    #                 r = mid - 1
    #             else:
    #                 l = mid + 1
    #     return -1

# 18. Find Minimum in Rotated Sorted Array (Medium)
# Question Link https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
# Video Solution: https://www.youtube.com/watch?v=U8XENwh8Oy8
# Python Solution: 

# class Solution:
#     def findMin(self, nums: List[int]) -> int:
#         res = nums[0]
#         l, r = 0, len(nums) - 1

#         while l <= r:
#             if nums[l] < nums[r]:
#                 res = min(res, nums[l])
#                 break
#             m = (l + r) // 2
#             res = min(res, nums[m])
#             if nums[m] >= nums[l]:
#                 l = m + 1
#             else:
#                 r = m - 1
#         return res

                                                                    # LINKED LIST (0/6)

# 19. Reverse Linked List (Easy)
# Question Link https://leetcode.com/problems/reverse-linked-list/
# Video Solution: https://www.youtube.com/watch?v=G0_I-ZF0S38
# Python Solution: 

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def reverseList(self, head: ListNode) -> ListNode:
#         prev, curr = None, head

#         while curr:
#             temp = curr.next
#             curr.next = prev
#             prev = curr
#             curr = temp
#         return prev

# 20. Merge Two Linked Lists (Easy)
# Question Link https://leetcode.com/problems/merge-two-sorted-lists/
# Video Solution: https://www.youtube.com/watch?v=G0_I-ZF0S38
# Python Solution:

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# class Solution:
#     def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
#         dummy = ListNode()
#         tail = dummy

#         while list1 and list2:
#             if list1.val < list2.val:
#                 tail.next = list1
#                 list1 = list1.next
#             else:
#                 tail.next = list2
#                 list2 = list2.next
#             tail = tail.next

#         if list1:
#             tail.next = list1
#         elif list2:
#             tail.next = list2

#         return dummy.next



# 21. Reorder List (Medium)
# Question Link https://leetcode.com/problems/reorder-list/
# Video Solution: https://www.youtube.com/watch?v=S5bfdUTrKLM
# Python Solution: 

# class Solution:
#     def reorderList(self, head: ListNode) -> None:
#         # find middle
#         slow, fast = head, head.next
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next

#         # reverse second half
#         second = slow.next
#         prev = slow.next = None
#         while second:
#             tmp = second.next
#             second.next = prev
#             prev = second
#             second = tmp

#         # merge two halfs
#         first, second = head, prev
#         while second:
#             tmp1, tmp2 = first.next, second.next
#             first.next = second
#             second.next = tmp1
#             first, second = tmp1, tmp2


# 22. Remove Nth Node from End of List (Medium)
# Question Link https://leetcode.com/problems/remove-nth-node-from-end-of-list/
# Video Solution: https://www.youtube.com/watch?v=XVuQxVej6y8
# Python Solution: 

# class Solution:
#     def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
#         dummy = ListNode(0, head)
#         left = dummy
#         right = head

#         while n > 0:
#             right = right.next
#             n -= 1

#         while right:
#             left = left.next
#             right = right.next

#         # delete
#         left.next = left.next.next
#         return dummy.next


# 23. Linked List Cycle (Easy)
# Question Link https://leetcode.com/problems/linked-list-cycle/
# Video Solution: https://www.youtube.com/watch?v=gBTe7lFR3vc
# Python Solution: 

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


# class Solution:
#     def hasCycle(self, head: ListNode) -> bool:
#         slow, fast = head, head

#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next
#             if slow == fast:
#                 return True
#         return False


# 24. Merge K Sorted Lists (Hard)
# Question Link https://leetcode.com/problems/merge-k-sorted-lists/
# Video Solution: https://www.youtube.com/watch?v=q5a5OiGbT6Q
# Python Solution: 

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# class Solution:
#     def mergeKLists(self, lists: List[ListNode]) -> ListNode:
#         if not lists or len(lists) == 0:
#             return None

#         while len(lists) > 1:
#             mergedLists = []
#             for i in range(0, len(lists), 2):
#                 l1 = lists[i]
#                 l2 = lists[i + 1] if (i + 1) < len(lists) else None
#                 mergedLists.append(self.mergeList(l1, l2))
#             lists = mergedLists
#         return lists[0]

#     def mergeList(self, l1, l2):
#         dummy = ListNode()
#         tail = dummy

#         while l1 and l2:
#             if l1.val < l2.val:
#                 tail.next = l1
#                 l1 = l1.next
#             else:
#                 tail.next = l2

                                                                      # TREES (0-11)

# 25. Invert Binary Tree (Easy)
# Question Link https://leetcode.com/problems/invert-binary-tree/
# Video Solution: https://www.youtube.com/watch?v=OnSn2XEQ4MY
# Python Solution: 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def invertTree(self, root: TreeNode) -> TreeNode:
#         if not root:
#             return None

#         # swap the children
#         tmp = root.left
#         root.left = root.right
#         root.right = tmp

#         self.invertTree(root.left)
#         self.invertTree(root.right)
#         return root


# 26. Maximum Depth of Binary Tree (Easy)
# Question Link https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Video Solution: https://www.youtube.com/watch?v=hTM3phVI6YQ
# Python Solution:

# RECURSIVE DFS
# class Solution:
#     def maxDepth(self, root: TreeNode) -> int:
#         if not root:
#             return 0

#         return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


# ITERATIVE DFS
# class Solution:
#     def maxDepth(self, root: TreeNode) -> int:
#         stack = [[root, 1]]
#         res = 0

#         while stack:
#             node, depth = stack.pop()

#             if node:
#                 res = max(res, depth)
#                 stack.append([node.left, depth + 1])
#                 stack.append([node.right, depth + 1])
#         return res


# BFS
# class Solution:
#     def maxDepth(self, root: TreeNode) -> int:
#         q = deque()
#         if root:
#             q.append(root)

#         level = 0
        
#         while q:

#             for i in range(len(q)):
#                 node = q.popleft()
#                 if node.left:
#                     q.append(node.left)
#                 if node.right:
#                     q.append(node.right)
#             level += 1
#         return level



# 27. Same Tree (Easy)
# Question Link https://leetcode.com/problems/same-tree/
# Video Solution: https://www.youtube.com/watch?v=vRbbcKXCxOw
# Python Solution: 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


# class Solution:
#     def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
#         if not p and not q:
#             return True
#         if p and q and p.val == q.val:
#             return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
#         else:
#             return False



# 28. Subtree of Another Tree (Easy)
# Question Link https://leetcode.com/problems/subtree-of-another-tree/
# Video Solution: https://www.youtube.com/watch?v=E36O5SWp-LE
# Python Solution: 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
#         if not t:
#             return True
#         if not s:
#             return False

#         if self.sameTree(s, t):
#             return True
#         return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

#     def sameTree(self, s, t):
#         if not s and not t:
#             return True
#         if s and t and s.val == t.val:
#             return self.sameTree(s.left, t.left) and self.sameTree(s.right, t.right)
#         return False


# 29. Lowest Common Ancestor of a BST (Easy)
# Question Link https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
# Video Solution: https://www.youtube.com/watch?v=gs2LMfuOR9k
# Python Solution:

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         cur = root
#         while cur:
#             if p.val > cur.val and q.val > cur.val:
#                 cur = cur.right
#             elif p.val < cur.val and q.val < cur.val:
#                 cur = cur.left
#             else:
#                 return cur



# 30. Binary Tree Level Order Traversal (Medium)
# Question Link https://leetcode.com/problems/binary-tree-level-order-traversal/
# Video Solution: https://www.youtube.com/watch?v=6ZnyEApgFYg
# Python Solution: 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Solution:
#     def levelOrder(self, root: TreeNode) -> List[List[int]]:
#         res = []
#         q = collections.deque()
#         if root:
#             q.append(root)

#         while q:
#             val = []

#             for i in range(len(q)):
#                 node = q.popleft()
#                 val.append(node.val)
#                 if node.left:
#                     q.append(node.left)
#                 if node.right:
#                     q.append(node.right)
#             res.append(val)
#         return res


# 31. Validate Binary Search Tree (Medium)
# Question Link https://leetcode.com/problems/validate-binary-search-tree/
# Video Solution: https://www.youtube.com/watch?v=s6ATEkipzow
# Python Solution: 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def isValidBST(self, root: TreeNode) -> bool:
#         def valid(node, left, right):
#             if not node:
#                 return True
#             if not (node.val < right and node.val > left):
#                 return False

#             return valid(node.left, left, node.val) and valid(
#                 node.right, node.val, right
#             )

#         return valid(root, float("-inf"), float("inf"))


# 32. Kth Smallest Element in a BST (Medium)
# Question Link https://leetcode.com/problems/kth-smallest-element-in-a-bst/
# Video Solution: https://www.youtube.com/watch?v=5LUXSvjmGCw
# Python Solution:

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Solution:
#     def kthSmallest(self, root: TreeNode, k: int) -> int:
#         stack = []
#         curr = root

#         while stack or curr:
#             while curr:
#                 stack.append(curr)
#                 curr = curr.left
#             curr = stack.pop()
#             k -= 1
#             if k == 0:
#                 return curr.val
#             curr = curr.right



# 33. Construct Tree from Preorder and Inorder Traversal (Medium)
# Question Link https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
# Video Solution: https://www.youtube.com/watch?v=ihj4IQGZ2zc
# Python Solution:
# class Solution:
    # def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    #     if not preorder or not inorder:
    #         return None

    #     root = TreeNode(preorder[0])
    #     mid = inorder.index(preorder[0])
    #     root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
    #     root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])
    #     return root


# 34. Binary Tree Max Path Sum (Hard)
# Question Link https://leetcode.com/problems/binary-tree-maximum-path-sum/
# Video Solution: https://www.youtube.com/watch?v=Hr5cWUld4vU
# Python Solution:

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def maxPathSum(self, root: TreeNode) -> int:
#         res = [root.val]

#         # return max path sum without split
#         def dfs(root):
#             if not root:
#                 return 0

#             leftMax = dfs(root.left)
#             rightMax = dfs(root.right)
#             leftMax = max(leftMax, 0)
#             rightMax = max(rightMax, 0)

#             # compute max path sum WITH split
#             res[0] = max(res[0], root.val + leftMax + rightMax)
#             return root.val + max(leftMax, rightMax)

#         dfs(root)
#         return res[0]


# 35. Serialize and Deserialize Binary Tree	(Hard)
# Question Link https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
# Video Solution: https://www.youtube.com/watch?v=u4JAi2JJhI8
# Python Solution: 

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Codec:
#     def serialize(self, root):
#         res = []

#         def dfs(node):
#             if not node:
#                 res.append("N")
#                 return
#             res.append(str(node.val))
#             dfs(node.left)
#             dfs(node.right)

#         dfs(root)
#         return ",".join(res)

#     def deserialize(self, data):
#         vals = data.split(",")
#         self.i = 0

#         def dfs():
#             if vals[self.i] == "N":
#                 self.i += 1
#                 return None
#             node = TreeNode(int(vals[self.i]))
#             self.i += 1
#             node.left = dfs()
#             node.right = dfs()
#             return node

#         return dfs()

                                                                      # TRIES(0/3)

# 36. Implement Trie (Medium)
# Question Link https://leetcode.com/problems/implement-trie-prefix-tree/
# Video Solution: https://www.youtube.com/watch?v=oobqoCJlHA0
# Python Solution: 
# class TrieNode:
#     def __init__(self):
#         self.children = [None] * 26
#         self.end = False


# class Trie:
#     def __init__(self):
#         """
#         Initialize your data structure here.
#         """
#         self.root = TrieNode()

#     def insert(self, word: str) -> None:
#         """
#         Inserts a word into the trie.
#         """
#         curr = self.root
#         for c in word:
#             i = ord(c) - ord("a")
#             if curr.children[i] == None:
#                 curr.children[i] = TrieNode()
#             curr = curr.children[i]
#         curr.end = True

#     def search(self, word: str) -> bool:
#         """
#         Returns if the word is in the trie.
#         """
#         curr = self.root
#         for c in word:
#             i = ord(c) - ord("a")
#             if curr.children[i] == None:
#                 return False
#             curr = curr.children[i]
#         return curr.end

#     def startsWith(self, prefix: str) -> bool:
#         """
#         Returns if there is any word in the trie that starts with the given prefix.
#         """
#         curr = self.root
#         for c in prefix:
#             i = ord(c) - ord("a")
#             if curr.children[i] == None:
#                 return False
#             curr = curr.children[i]
#         return True



# 37. Design Add and Search Word Data Structure	(Medium)
# Question Link https://leetcode.com/problems/design-add-and-search-words-data-structure/
# Video Solution: https://www.youtube.com/watch?v=BTf05gs_8iU
# Python Solution: 

# class TrieNode:
#     def __init__(self):
#         self.children = {}  # a : TrieNode
#         self.word = False


# class WordDictionary:
#     def __init__(self):
#         self.root = TrieNode()

#     def addWord(self, word: str) -> None:
#         cur = self.root
#         for c in word:
#             if c not in cur.children:
#                 cur.children[c] = TrieNode()
#             cur = cur.children[c]
#         cur.word = True

#     def search(self, word: str) -> bool:
#         def dfs(j, root):
#             cur = root

#             for i in range(j, len(word)):
#                 c = word[i]
#                 if c == ".":
#                     for child in cur.children.values():
#                         if dfs(i + 1, child):
#                             return True
#                     return False
#                 else:
#                     if c not in cur.children:
#                         return False
#                     cur = cur.children[c]
#             return cur.word

#         return dfs(0, self.root)


# 38. Word Search II (Hard)	
# Question Link https://leetcode.com/problems/word-search-ii/
# Video Solution: https://www.youtube.com/watch?v=asbcE9mZz_U
# Python Solution: 

# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.isWord = False
#         self.refs = 0

#     def addWord(self, word):
#         cur = self
#         cur.refs += 1
#         for c in word:
#             if c not in cur.children:
#                 cur.children[c] = TrieNode()
#             cur = cur.children[c]
#             cur.refs += 1
#         cur.isWord = True

#     def removeWord(self, word):
#         cur = self
#         cur.refs -= 1
#         for c in word:
#             if c in cur.children:
#                 cur = cur.children[c]
#                 cur.refs -= 1


# class Solution:
#     def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
#         root = TrieNode()
#         for w in words:
#             root.addWord(w)

#         ROWS, COLS = len(board), len(board[0])
#         res, visit = set(), set()

#         def dfs(r, c, node, word):
#             if (
#                 r < 0
#                 or c < 0
#                 or r == ROWS
#                 or c == COLS
#                 or board[r][c] not in node.children
#                 or node.children[board[r][c]].refs < 1
#                 or (r, c) in visit
#             ):
#                 return

#             visit.add((r, c))
#             node = node.children[board[r][c]]
#             word += board[r][c]
#             if node.isWord:
#                 node.isWord = False
#                 res.add(word)
#                 root.removeWord(word)

#             dfs(r + 1, c, node, word)
#             dfs(r - 1, c, node, word)
#             dfs(r, c + 1, node, word)
#             dfs(r, c - 1, node, word)
#             visit.remove((r, c))

#         for r in range(ROWS):
#             for c in range(COLS):
#                 dfs(r, c, root, "")

#         return list(res)

                                                              # HEAP/PRIORITY QUEUE (0-1)

# 39. Find Median from Data Stream (Hard)
# Question Link https://leetcode.com/problems/find-median-from-data-stream/
# Video Solution: https://www.youtube.com/watch?v=itmhHWaHupI
# Python Solution: 

# class MedianFinder:
#     def __init__(self):
#         """
#         initialize your data structure here.
#         """
#         # two heaps, large, small, minheap, maxheap
#         # heaps should be equal size
#         self.small, self.large = [], []  # maxHeap, minHeap (python default)

#     def addNum(self, num: int) -> None:
#         heapq.heappush(self.small, -1 * num)

#         if self.small and self.large and (-1 * self.small[0]) > self.large[0]:
#             val = -1 * heapq.heappop(self.small)
#             heapq.heappush(self.large, val)

#         if len(self.small) > len(self.large) + 1:
#             val = -1 * heapq.heappop(self.small)
#             heapq.heappush(self.large, val)
#         if len(self.large) > len(self.small) + 1:
#             val = heapq.heappop(self.large)
#             heapq.heappush(self.small, -1 * val)

#     def findMedian(self) -> float:
#         if len(self.small) > len(self.large):
#             return -1 * self.small[0]
#         elif len(self.large) > len(self.small):
#             return self.large[0]
#         return (-1 * self.small[0] + self.large[0]) / 2

                                                                # BACKTRACKING (0/2)


# 40. Combination Sum	(Medium)
# Question Link https://leetcode.com/problems/combination-sum/
# Video Solution: https://www.youtube.com/watch?v=GBKI9VSKdGg
# Python Solution: 

# class Solution:
    # def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    #     res = []

    #     def dfs(i, cur, total):
    #         if total == target:
    #             res.append(cur.copy())
    #             return
    #         if i >= len(candidates) or total > target:
    #             return

    #         cur.append(candidates[i])
    #         dfs(i, cur, total + candidates[i])
    #         cur.pop()
    #         dfs(i + 1, cur, total)

    #     dfs(0, [], 0)
    #     return res


# 41. Word Search (Medium)
# Question Link https://leetcode.com/problems/word-search/
# Video Solution: https://www.youtube.com/watch?v=pfiQ_PS1g8E
# Python Solution: 

# class Solution:
#     def exist(self, board: List[List[str]], word: str) -> bool:
#         ROWS, COLS = len(board), len(board[0])
#         path = set()

#         def dfs(r, c, i):
#             if i == len(word):
#                 return True
#             if (
#                 min(r, c) < 0
#                 or r >= ROWS
#                 or c >= COLS
#                 or word[i] != board[r][c]
#                 or (r, c) in path
#             ):
#                 return False
#             path.add((r, c))
#             res = (
#                 dfs(r + 1, c, i + 1)
#                 or dfs(r - 1, c, i + 1)
#                 or dfs(r, c + 1, i + 1)
#                 or dfs(r, c - 1, i + 1)
#             )
#             path.remove((r, c))
#             return res

#         for r in range(ROWS):
#             for c in range(COLS):
#                 if dfs(r, c, 0):
#                     return True
#         return False

    # O(n * m * 4^n)

                                                                      #GRAPHS (0/6)

# 42. Number of Islands	(Medium)
# Question Link https://leetcode.com/problems/number-of-islands/
# Video Solution: https://www.youtube.com/watch?v=pV2kpPD66nE
# Python Solution: 

# class Solution:
#     def numIslands(self, grid: List[List[str]]) -> int:
#         if not grid or not grid[0]:
#             return 0

#         islands = 0
#         visit = set()
#         rows, cols = len(grid), len(grid[0])

#         def dfs(r, c):
#             if (
#                 r not in range(rows)
#                 or c not in range(cols)
#                 or grid[r][c] == "0"
#                 or (r, c) in visit
#             ):
#                 return

#             visit.add((r, c))
#             directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
#             for dr, dc in directions:
#                 dfs(r + dr, c + dc)

#         for r in range(rows):
#             for c in range(cols):
#                 if grid[r][c] == "1" and (r, c) not in visit:
#                     islands += 1
#                     dfs(r, c)
#         return islands


# BFS Version From Video
# class SolutionBFS:
#     def numIslands(self, grid: List[List[str]]) -> int:
#         if not grid:
#             return 0

#         rows, cols = len(grid), len(grid[0])
#         visited=set()
#         islands=0

#          def bfs(r,c):
#              q = deque()
#              visited.add((r,c))
#              q.append((r,c))
           
#              while q:
#                  row,col = q.popleft()
#                  directions= [[1,0],[-1,0],[0,1],[0,-1]]
               
#                  for dr,dc in directions:
#                      r,c = row + dr, col + dc
#                      if (r) in range(rows) and (c) in range(cols) and grid[r][c] == '1' and (r ,c) not in visited:
                       
#                          q.append((r , c ))
#                          visited.add((r, c ))

#          for r in range(rows):
#              for c in range(cols):
               
#                  if grid[r][c] == "1" and (r,c) not in visited:
#                      bfs(r,c)
#                      islands +=1 

#          return islands



# 43. Clone Graph	(Medium)
# Question Link https://leetcode.com/problems/clone-graph/
# Video Solution: https://www.youtube.com/watch?v=mQeF6bN8hMk
# Python Solution: 

# class Solution:
    # def cloneGraph(self, node: "Node") -> "Node":
    #     oldToNew = {}

    #     def dfs(node):
    #         if node in oldToNew:
    #             return oldToNew[node]

    #         copy = Node(node.val)
    #         oldToNew[node] = copy
    #         for nei in node.neighbors:
    #             copy.neighbors.append(dfs(nei))
    #         return copy

    #     return dfs(node) if node else None


# 44. Pacific Atlantic Waterflow (Medium)
# Question Link https://leetcode.com/problems/pacific-atlantic-water-flow/
# Video Solution: https://www.youtube.com/watch?v=s-VkcjHqkGI
# Python Solution: 

# class Solution:
#     def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
#         ROWS, COLS = len(heights), len(heights[0])
#         pac, atl = set(), set()

#         def dfs(r, c, visit, prevHeight):
#             if (
#                 (r, c) in visit
#                 or r < 0
#                 or c < 0
#                 or r == ROWS
#                 or c == COLS
#                 or heights[r][c] < prevHeight
#             ):
#                 return
#             visit.add((r, c))
#             dfs(r + 1, c, visit, heights[r][c])
#             dfs(r - 1, c, visit, heights[r][c])
#             dfs(r, c + 1, visit, heights[r][c])
#             dfs(r, c - 1, visit, heights[r][c])

#         for c in range(COLS):
#             dfs(0, c, pac, heights[0][c])
#             dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])

#         for r in range(ROWS):
#             dfs(r, 0, pac, heights[r][0])
#             dfs(r, COLS - 1, atl, heights[r][COLS - 1])

#         res = []
#         for r in range(ROWS):
#             for c in range(COLS):
#                 if (r, c) in pac and (r, c) in atl:
#                     res.append([r, c])
#         return res


# 45. Course Schedule	(Medium)
# Question Link https://leetcode.com/problems/course-schedule/
# Video Solution: https://www.youtube.com/watch?v=EgI5nU9etnU
# Python Solution: 

# class Solution:
#     def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
#         # dfs
#         preMap = {i: [] for i in range(numCourses)}

#         # map each course to : prereq list
#         for crs, pre in prerequisites:
#             preMap[crs].append(pre)

#         visiting = set()

#         def dfs(crs):
#             if crs in visiting:
#                 return False
#             if preMap[crs] == []:
#                 return True

#             visiting.add(crs)
#             for pre in preMap[crs]:
#                 if not dfs(pre):
#                     return False
#             visiting.remove(crs)
#             preMap[crs] = []
#             return True

#         for c in range(numCourses):
#             if not dfs(c):
#                 return False
#         return True



# 46. Number of Connected Components in Graph	(Medium)
# Question Link https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
# Video Solution: https://www.youtube.com/watch?v=8f1XPm4WOUc
# Python Solution: 

# class UnionFind:
#     def __init__(self):
#         self.f = {}

#     def findParent(self, x):
#         y = self.f.get(x, x)
#         if x != y:
#             y = self.f[x] = self.findParent(y)
#         return y

#     def union(self, x, y):
#         self.f[self.findParent(x)] = self.findParent(y)


# class Solution:
#     def countComponents(self, n: int, edges: List[List[int]]) -> int:
#         dsu = UnionFind()
#         for a, b in edges:
#             dsu.union(a, b)
#         return len(set(dsu.findParent(x) for x in range(n)))



# 47. Graph Valid Tree (Medium)
# Question Link https://leetcode.com/problems/graph-valid-tree/
# Video Solution: https://www.youtube.com/watch?v=bXsUuownnoQ
# Python Solution: 

# Problem is free on Lintcode
# class Solution:
#     """
#     @param n: An integer
#     @param edges: a list of undirected edges
#     @return: true if it's a valid tree, or false
#     """

#     def validTree(self, n, edges):
#         if not n:
#             return True
#         adj = {i: [] for i in range(n)}
#         for n1, n2 in edges:
#             adj[n1].append(n2)
#             adj[n2].append(n1)

#         visit = set()

#         def dfs(i, prev):
#             if i in visit:
#                 return False

#             visit.add(i)
#             for j in adj[i]:
#                 if j == prev:
#                     continue
#                 if not dfs(j, i):
#                     return False
#             return True

#         return dfs(0, -1) and n == len(visit)


                                                      # ADVANCED GRAPHS (0-1)


# 48. Alien Dictionary (Hard)
# Question Link https://leetcode.com/problems/alien-dictionary/
# Video Solution: https://www.youtube.com/watch?v=6kTZYvNNyps
# Python Solution: 

# class Solution:
    # def alienOrder(self, words: List[str]) -> str:
    #     adj = {char: set() for word in words for char in word}

    #     for i in range(len(words) - 1):
    #         w1, w2 = words[i], words[i + 1]
    #         minLen = min(len(w1), len(w2))
    #         if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
    #             return ""
    #         for j in range(minLen):
    #             if w1[j] != w2[j]:
    #                 print(w1[j], w2[j])
    #                 adj[w1[j]].add(w2[j])
    #                 break

    #     visited = {}  # {char: bool} False visited, True current path
    #     res = []

    #     def dfs(char):
    #         if char in visited:
    #             return visited[char]

    #         visited[char] = True

    #         for neighChar in adj[char]:
    #             if dfs(neighChar):
    #                 return True

    #         visited[char] = False
    #         res.append(char)

    #     for char in adj:
    #         if dfs(char):
    #             return ""

    #     res.reverse()
    #     return "".join(res)


                                                              # 1-D DYNAMIC PROGRAMMING (0-10) 


# 49. Climbing Stairs	(easy)
# Question Link https://leetcode.com/problems/climbing-stairs/
# Video Solution: https://www.youtube.com/watch?v=Y0lT9Fck7qI
# Python Solution: 

# class Solution:
#     def climbStairs(self, n: int) -> int:
#         if n <= 3:
#             return n
#         n1, n2 = 2, 3

#         for i in range(4, n + 1):
#             temp = n1 + n2
#             n1 = n2
#             n2 = temp
#         return n2


# 50. House Robber (Medium)
# Question Link https://leetcode.com/problems/house-robber/
# Video Solution: https://www.youtube.com/watch?v=73r3KWiEvyk
# Python Solution: 

# class Solution:
    # def rob(self, nums: List[int]) -> int:
    #     rob1, rob2 = 0, 0

    #     for n in nums:
    #         temp = max(n + rob1, rob2)
    #         rob1 = rob2
    #         rob2 = temp
    #     return rob2


# 51. House Robber II	(Medium)
# Question Link https://leetcode.com/problems/house-robber-ii/
# Video Solution: https://www.youtube.com/watch?v=rWAJCfYYOvM
# Python Solution: 

# class Solution:
    # def rob(self, nums: List[int]) -> int:
    #     return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))

    # def helper(self, nums):
    #     rob1, rob2 = 0, 0

    #     for n in nums:
    #         newRob = max(rob1 + n, rob2)
    #         rob1 = rob2
    #         rob2 = newRob
    #     return rob2



# 52. Longest Palindromic Substring	(Medium)
# Question Link https://leetcode.com/problems/longest-palindromic-substring/
# Video Solution: https://www.youtube.com/watch?v=XYQecbcd6_c
# Python Solution:

# class Solution:
#     def longestPalindrome(self, s: str) -> str:
#         res = ""
#         resLen = 0

#         for i in range(len(s)):
#             # odd length
#             l, r = i, i
#             while l >= 0 and r < len(s) and s[l] == s[r]:
#                 if (r - l + 1) > resLen:
#                     res = s[l : r + 1]
#                     resLen = r - l + 1
#                 l -= 1
#                 r += 1

#             # even length
#             l, r = i, i + 1
#             while l >= 0 and r < len(s) and s[l] == s[r]:
#                 if (r - l + 1) > resLen:
#                     res = s[l : r + 1]
#                     resLen = r - l + 1
#                 l -= 1
#                 r += 1

#         return res


# 53. Palindromic Substrings (Medium)
# Question Link https://leetcode.com/problems/palindromic-substrings/
# Video Solution: https://www.youtube.com/watch?v=4RACzI5-du8
# Python Solution:

# class Solution:
#     def countSubstrings(self, s: str) -> int:
#         res = 0

#         for i in range(len(s)):
#             res += self.countPali(s, i, i)
#             res += self.countPali(s, i, i + 1)
#         return res

#     def countPali(self, s, l, r):
#         res = 0
#         while l >= 0 and r < len(s) and s[l] == s[r]:
#             res += 1
#             l -= 1
#             r += 1
#         return res



# 54. Decode Ways	(Medium)
# Question Link https://leetcode.com/problems/decode-ways/
# Video Solution: https://www.youtube.com/watch?v=6aEyTjOwlJU
# Python Solution: 

# class Solution:
#     def numDecodings(self, s: str) -> int:
#         # Memoization
#         dp = {len(s): 1}

#         def dfs(i):
#             if i in dp:
#                 return dp[i]
#             if s[i] == "0":
#                 return 0

#             res = dfs(i + 1)
#             if i + 1 < len(s) and (
#                 s[i] == "1" or s[i] == "2" and s[i + 1] in "0123456"
#             ):
#                 res += dfs(i + 2)
#             dp[i] = res
#             return res

#         return dfs(0)

#         # Dynamic Programming
#         dp = {len(s): 1}
#         for i in range(len(s) - 1, -1, -1):
#             if s[i] == "0":
#                 dp[i] = 0
#             else:
#                 dp[i] = dp[i + 1]



# 55. Coin Change	(Medium)
# Question Link https://leetcode.com/problems/coin-change/
# Video Solution: https://www.youtube.com/watch?v=H9bfqozjoqs
# Python Solution: 

# class Solution:
    # def coinChange(self, coins: List[int], amount: int) -> int:
    #     dp = [amount + 1] * (amount + 1)
    #     dp[0] = 0

    #     for a in range(1, amount + 1):
    #         for c in coins:
    #             if a - c >= 0:
    #                 dp[a] = min(dp[a], 1 + dp[a - c])
    #     return dp[amount] if dp[amount] != amount + 1 else -1


# 56. Maximum Product Subarray (Medium)
# Question Link https://leetcode.com/problems/maximum-product-subarray/
# Video Solution: https://www.youtube.com/watch?v=lXVy6YWFcRM
# Python Solution: 

# class Solution:
#     def maxProduct(self, nums: List[int]) -> int:
#         # O(n)/O(1) : Time/Memory
#         res = nums[0]
#         curMin, curMax = 1, 1

#         for n in nums:

#             tmp = curMax * n
#             curMax = max(n * curMax, n * curMin, n)
#             curMin = min(tmp, n * curMin, n)
#             res = max(res, curMax)
#         return res

# 57. Word Break (Medium)
# Question Link https://leetcode.com/problems/word-break/
# Video Solution: https://www.youtube.com/watch?v=Sx9NNgInc3A
# Python Solution: 

# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:

#         dp = [False] * (len(s) + 1)
#         dp[len(s)] = True

#         for i in range(len(s) - 1, -1, -1):
#             for w in wordDict:
#                 if (i + len(w)) <= len(s) and s[i : i + len(w)] == w:
#                     dp[i] = dp[i + len(w)]
#                 if dp[i]:
#                     break

#         return dp[0]

# 58. Longest Increasing Subsequence (Medium)
# Question Link https://leetcode.com/problems/longest-increasing-subsequence/
# Video Solution: https://www.youtube.com/watch?v=cjWnW0hdF1Y
# Python Solution: 

# class Solution:
#     def lengthOfLIS(self, nums: List[int]) -> int:
#         LIS = [1] * len(nums)

#         for i in range(len(nums) - 1, -1, -1):
#             for j in range(i + 1, len(nums)):
#                 if nums[i] < nums[j]:
#                     LIS[i] = max(LIS[i], 1 + LIS[j])
#         return max(LIS)


                                                  # 2-D DYNAMIC PROGRAMMING (0/2)


# 59. Unique Paths (Medium)
# Question Link https://leetcode.com/problems/unique-paths/
# Video Solution: https://www.youtube.com/watch?v=IlEsdxuD4lY
# Python Solution: 

# class Solution:
    # def uniquePaths(self, m: int, n: int) -> int:
    #     row = [1] * n

    #     for i in range(m - 1):
    #         newRow = [1] * n
    #         for j in range(n - 2, -1, -1):
    #             newRow[j] = newRow[j + 1] + row[j]
    #         row = newRow
    #     return row[0]

        # O(n * m) O(n)


# 60. Longest Common Subsequence (Medium)
# Question Link https://leetcode.com/problems/longest-common-subsequence/
# Video Solution: https://www.youtube.com/watch?v=Ua0GhsJSlWM
# Python Solution: 

# class Solution:
    # def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    #     dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]

    #     for i in range(len(text1) - 1, -1, -1):
    #         for j in range(len(text2) - 1, -1, -1):
    #             if text1[i] == text2[j]:
    #                 dp[i][j] = 1 + dp[i + 1][j + 1]
    #             else:
    #                 dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])

    #     return dp[0][0]


                                                            # GREEDY (0/2)

# 61. Maximum Subarray (Medium)
# Question Link https://leetcode.com/problems/maximum-subarray/
# Video Solution: https://www.youtube.com/watch?v=5WZl3MMT0Eg
# Python Solution: 

# class Solution:
#     def maxSubArray(self, nums: List[int]) -> int:
#         res = nums[0]

#         total = 0
#         for n in nums:
#             total += n
#             res = max(res, total)
#             if total < 0:
#                 total = 0
#         return res


# 62. Jump Game	(Medium)
# Question Link https://leetcode.com/problems/jump-game/
# Video Solution: https://www.youtube.com/watch?v=Yan0cv2cLy8
# Python Solution: 

# class Solution:
    # def canJump(self, nums: List[int]) -> bool:
    #     goal = len(nums) - 1

    #     for i in range(len(nums) - 2, -1, -1):
    #         if i + nums[i] >= goal:
    #             goal = i
    #     return goal == 0


                                                                  #INTERVALS (0/5)


# 63. Insert Interval (Medium)
# Question Link https://leetcode.com/problems/insert-interval/
# Video Solution: https://www.youtube.com/watch?v=A8NUOmlwOlM
# Python Solution:

# class Solution:
#     def insert(
#         self, intervals: List[List[int]], newInterval: List[int]
#     ) -> List[List[int]]:
#         res = []

#         for i in range(len(intervals)):
#             if newInterval[1] < intervals[i][0]:
#                 res.append(newInterval)
#                 return res + intervals[i:]
#             elif newInterval[0] > intervals[i][1]:
#                 res.append(intervals[i])
#             else:
#                 newInterval = [
#                     min(newInterval[0], intervals[i][0]),
#                     max(newInterval[1], intervals[i][1]),
#                 ]
#         res.append(newInterval)
#         return res


# 64. Merge Intervals	(Medium)
# Question Link https://leetcode.com/problems/merge-intervals/
# Video Solution: https://www.youtube.com/watch?v=44H3cEC2fFM
# Python Solution: 

# class Solution:
#     def merge(self, intervals: List[List[int]]) -> List[List[int]]:
#         intervals.sort(key=lambda pair: pair[0])
#         output = [intervals[0]]

#         for start, end in intervals:
#             lastEnd = output[-1][1]

#             if start <= lastEnd:
#                 # merge
#                 output[-1][1] = max(lastEnd, end)
#             else:
#                 output.append([start, end])
#         return output


# 65. Non-Overlapping Intervals	(Medium)
# Question Link https://leetcode.com/problems/non-overlapping-intervals/
# Video Solution: https://www.youtube.com/watch?v=nONCGxWoUfM
# Python Solution: 

# class Solution:
#     def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
#         intervals.sort()
#         res = 0
#         prevEnd = intervals[0][1]
#         for start, end in intervals[1:]:
#             if start >= prevEnd:
#                 prevEnd = end
#             else:
#                 res += 1
#                 prevEnd = min(end, prevEnd)
#         return res



# 66. Meeting Rooms	(Easy)
# Question Link https://leetcode.com/problems/meeting-rooms/
# Video Solution: https://www.youtube.com/watch?v=PaJxqZVPhbg
# Python Solution:

# class Solution:
#     """
#     @param intervals: an array of meeting time intervals
#     @return: if a person could attend all meetings
#     """

#     def canAttendMeetings(self, intervals):
#         intervals.sort(key=lambda i: i[0])

#         for i in range(1, len(intervals)):
#             i1 = intervals[i - 1]
#             i2 = intervals[i]

#             if i1[1] > i2[0]:
#                 return False
#         return True



# 67. Meeting Rooms II (Medium)
# Question Link https://leetcode.com/problems/meeting-rooms-ii/
# Video Solution: https://www.youtube.com/watch?v=FdzJmTCVyJU
# Python Solution:

# class Solution:
#     """
#     @param intervals: an array of meeting time intervals
#     @return: the minimum number of conference rooms required
#     """

#     def minMeetingRooms(self, intervals):
#         start = sorted([i.start for i in intervals])
#         end = sorted([i.end for i in intervals])

#         res, count = 0, 0
#         s, e = 0, 0
#         while s < len(intervals):
#             if start[s] < end[e]:
#                 s += 1
#                 count += 1
#             else:
#                 e += 1
#                 count -= 1
#             res = max(res, count)
#         return res



                                                                # MATH & GEOMETRY (0/3)


# 68. Rotate Image (Medium)
# Question Link https://leetcode.com/problems/rotate-image/
# Video Solution: https://www.youtube.com/watch?v=fMSJSS7eO1w
# Python Solution: 

# class Solution:
    # def rotate(self, matrix: List[List[int]]) -> None:
    #     """
    #     Do not return anything, modify matrix in-place instead.
    #     """
    #     l, r = 0, len(matrix) - 1
    #     while l < r:
    #         for i in range(r - l):
    #             top, bottom = l, r

    #             # save the topleft
    #             topLeft = matrix[top][l + i]

    #             # move bottom left into top left
    #             matrix[top][l + i] = matrix[bottom - i][l]

    #             # move bottom right into bottom left
    #             matrix[bottom - i][l] = matrix[bottom][r - i]

    #             # move top right into bottom right
    #             matrix[bottom][r - i] = matrix[top + i][r]

    #             # move top left into top right
    #             matrix[top + i][r] = topLeft
    #         r -= 1
    #         l += 1

# 69. Spiral Matrix	(Medium)	
# Question Link https://leetcode.com/problems/spiral-matrix/
# Video Solution: https://www.youtube.com/watch?v=BJnMZNwUk1M
# Python Solution: 

# class Solution:
#     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
#         res = []
#         left, right = 0, len(matrix[0])
#         top, bottom = 0, len(matrix)

#         while left < right and top < bottom:
#             # get every i in the top row
#             for i in range(left, right):
#                 res.append(matrix[top][i])
#             top += 1
#             # get every i in the right col
#             for i in range(top, bottom):
#                 res.append(matrix[i][right - 1])
#             right -= 1
#             if not (left < right and top < bottom):
#                 break
#             # get every i in the bottom row
#             for i in range(right - 1, left - 1, -1):
#                 res.append(matrix[bottom - 1][i])
#             bottom -= 1
#             # get every i in the left col
#             for i in range(bottom - 1, top - 1, -1):
#                 res.append(matrix[i][left])
#             left += 1

#         return res


# 70. Set Matrix Zeroes	(Medium)
# Question Link https://leetcode.com/problems/set-matrix-zeroes/
# Video Solution: https://www.youtube.com/watch?v=T41rL0L3Pnw
# Python Solution:

# class Solution:
#     def setZeroes(self, matrix: List[List[int]]) -> None:
#         # O(1)
#         ROWS, COLS = len(matrix), len(matrix[0])
#         rowZero = False

#         # determine which rows/cols need to be zero
#         for r in range(ROWS):
#             for c in range(COLS):
#                 if matrix[r][c] == 0:
#                     matrix[0][c] = 0
#                     if r > 0:
#                         matrix[r][0] = 0
#                     else:
#                         rowZero = True

#         for r in range(1, ROWS):
#             for c in range(1, COLS):
#                 if matrix[0][c] == 0 or matrix[r][0] == 0:
#                     matrix[r][c] = 0

#         if matrix[0][0] == 0:
#             for r in range(ROWS):
#                 matrix[r][0] = 0

#         if rowZero:
#             for c in range(COLS):
#                 matrix[0][c] = 0

# BIT MANIPULATION (0/5)

# 71. Number of 1 Bits (Easy)
# Question Link https://leetcode.com/problems/number-of-1-bits/
# Video Solution: https://www.youtube.com/watch?v=5Km3utixwZs
# Python Solution:
 
# class Solution:
    # def hammingWeight(self, n: int) -> int:
    #     res = 0
    #     while n:
    #         n &= n - 1
    #         res += 1
    #     return res


# 72. Counting Bits	(Easy)
# Question Link https://leetcode.com/problems/counting-bits/
# Video Solution: https://www.youtube.com/watch?v=RyBM56RIWrM
# Python Solution: 

# class Solution:
#     def countBits(self, n: int) -> List[int]:
#         dp = [0] * (n + 1)
#         offset = 1

#         for i in range(1, n + 1):
#             if offset * 2 == i:
#                 offset = i
#             dp[i] = 1 + dp[i - offset]
#         return dp


# 73. Reverse Bits (Easy)
# Question Link https://leetcode.com/problems/reverse-bits/
# Video Solution: https://www.youtube.com/watch?v=UcoN6UjAI64
# Python Solution:

# class Solution:
#     def reverseBits(self, n: int) -> int:
#         res = 0
#         for i in range(32):
#             bit = (n >> i) & 1
#             res = res | (bit << (31 - i))
#         return res


# 74. Missing Number (Easy)
# Question Link https://leetcode.com/problems/missing-number/
# Video Solution: https://www.youtube.com/watch?v=WnPLSRLSANE
# Python Solution:

# class Solution:
#     def missingNumber(self, nums: List[int]) -> int:
#         res = len(nums)

#         for i in range(len(nums)):
#             res += i - nums[i]
#         return res


# 75. Sum of Two Integers	(Medium)
# Question Link https://leetcode.com/problems/sum-of-two-integers/
# Video Solution: https://www.youtube.com/watch?v=gVUrDV4tZfY
# Python Solution:

# class Solution:
#     def getSum(self, a: int, b: int) -> int:
#         def add(a, b):
#             if not a or not b:
#                 return a or b
#             return add(a ^ b, (a & b) << 1)

#         if a * b < 0:  # assume a < 0, b > 0
#             if a > 0:
#                 return self.getSum(b, a)
#             if add(~a, 1) == b:  # -a == b
#                 return 0
#             if add(~a, 1) < b:  # -a < b
#                 return add(~add(add(~a, 1), add(~b, 1)), 1)  # -add(-a, -b)

#         return add(a, b)  # a*b >= 0 or (-a) > b > 0












































