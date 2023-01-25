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
#         hashset = set() # initalize a hashset

#         for n in nums: # iterate over nums array
#             if n in hashset: # if n is in my hashset
#                 return True # there is duplicate
#             hashset.add(n) # else we add n to our hashset
#         return False # no duplicate found 





# 2. Valid Anagram (easy)

# Question Link https://leetcode.com/problems/valid-anagram/
# Video Solution: https://www.youtube.com/watch?v=9UtInBqnCgA&feature=emb_imp_woyt
# Python Solution:

# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         if len(s) != len(t): # if the length of s does not equal t we can immediatelty return false
#             return False

#         countS, countT = {}, {} # initalize two empty hashmaps to keep track of the # of each char

#         for i in range(len(s)): # we know the length of s and t are equal so we iterate over the len of s
#             countS[s[i]] = 1 + countS.get(s[i], 0) # build our maps [s[i]] character is the key 
#             countT[t[i]] = 1 + countT.get(t[i], 0) # .get special python function if the key doesnt exist we return a default val of 0  
#         if countS == countT: # if the hashmaps have the same # of each char they are anagrams
#           return True 



# 3. Two Sum (easy)

# Question Link https://leetcode.com/problems/two-sum/
# Video Solution: https://www.youtube.com/watch?v=KLlXCFG5TnA
# Python Solution: 

# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         prevMap = {}  # val -> index # initialize empty hashmap to keep track of val and index

#         for i, n in enumerate(nums): # i is the index, n is the val 
#             diff = target - n # diff variable, 9 - 2, n is the val we are currently at
#             if diff in prevMap: # if diff exists in our map 
#                 return [prevMap[diff], i] # we can return diffs index and i the index we are at
#             prevMap[n] = i # if diff is not in our map, we update our map, val of n with index i

from curses.ascii import isalnum
from email.policy import default

# 4. Group Anagrams (medium)

# Question Link https://leetcode.com/problems/group-anagrams/
# Video Solution: https://www.youtube.com/watch?v=vzdNOK2oB2E
# Python Solution:

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = defaultdict(list) # we are mapping the charCount of each string : list of anagrams
        
        for s in strs: # go through every string in the input
            count = [0] * 26 # array of 26 zeros, 1 for each char in the alphabet a ... z

            for c in s: # go through every single character in each string 
                count[ord(c) - ord("a")] += 1 # we want to map a to 0 and z to 25... 
            
            result[tuple(count)].append(s) # lists can't be keys in python 
        
        return result.values() # we just want the values (the lists of anagrams) not the keys

        # O(m * n) where m is the number of strings we are given
        # and n is avg length of each string (how many characters are in each string)



# 5. Top K Frequent Elements (medium)
# Question Link https://leetcode.com/problems/top-k-frequent-elements/
# Video Solution: https://www.youtube.com/watch?v=YPTqKIgVk-k
# Python Solution: 

# class Solution:
#     def topKFrequent(self, nums: List[int], k: int) -> List[int]:
#         count = {} # use a hashmap to count the occurances of each character    
# # special array, about the same size as input array (index is gonna be the count of the element, value is gonna be the list of vals that occur that many times)   
#         frequency = [[] for i in range(len(nums) + 1)] # initialize empty array, and the number of empty arrays inside will be the length of nums array + 1
#         for n in nums: # go through every val in the input array and see how many times it occurs 
#             count[n] = 1 + count.get(n, 0) # default val of zero if n doesn't exist in our hashmap
#         for n, c in count.items():  # Go through each val we counted. This will return each key val pair that we added to our dictionary # n is number, c is count 
#             frequency[c].append(n) # this value n, occurs c many times 

#         result = []  
#         for i in range(len(frequency) - 1, 0, -1): # looking for top k elements so we go in descending order 
#             for n in frequency[i]: # go through every value 
#                 result.append(n)
#                 if len(result) == k:
#                     return result



# 6. Product of Array Except Self (medium)
# Question Link https://leetcode.com/problems/product-of-array-except-self/
# Video Solution: https://www.youtube.com/watch?v=bNvIQI2wAjk
# Python Solution:

# class Solution:
#     def productExceptSelf(self, nums: List[int]) -> List[int]:
#         result = [0] * (len(nums)) # creating an output array, with a length of the input array. value [0] is arbitrary 

#         prefix = 1 # default val of 1 everything to left of the first val in the input array 
#         for i in range(len(nums)): # iterate over the nums array 
#             result[i] = prefix # we are assigning the output array "result" w the prefix values 
#             prefix *= nums[i]
#         postfix = 1
#         for i in range(len(nums) - 1, -1, -1): # start at the end of the input array and go up to the beginning 
#             result[i] *= postfix # essentially multiplying the prefix values currently in the output array by the postfix
#             postfix *= nums[i]
#         return result




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
#         numSet = set(nums) # create set, pass in nums array as the constructor 
#         longest = 0 # keep track of longest sequence, 

#         for n in nums: # iterate through every number in array
#             # check if a number is the start of a sequence
#             if (n - 1) not in numSet: # doesn't have left neighbor means its the start of a sequence 
#                 length = 0 # we want to find the length of the sequence
#                 while (n + length) in numSet: # n + length checks current number
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
        l, r = 0, len(s) - 1 # left and right pters, 1st index, last index
        
        while l < r: # while left pter is less than right pter 
            while l < r and not self.alphanum(s[l]): # not self.alphanum(s[l]) i.e space char
                l += 1 # we shift left pter to the right by 1
            while l < r and not self.alphanum(s[r]): 
                r -= 1 # we shift right pter to the left by 1
            if s[l].lower() != s[r].lower(): # if a char at index r is diff from a char at index l
                return False 
            l += 1 # still need to update pters on each iteration (move left)
            r -= 1 # move right by 1
        return True

    # Could write own alpha-numeric function
    def alphanum(self, c): # helper function 
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
#         result = [] # return the result as a list of lists
#         nums.sort() 

#         for i, a in enumerate(nums): # i is the index, a is the value 
#             if i > 0 and a == nums[i - 1]: # we don't want to reuse the value in the same position twice (i > 0 means it's not the first val in the index array)
#             # a == nums[i - 1] means it's the same value as before 
#                 continue
            
#             # two pter solution to basically solve two sum 
#             l, r = i + 1, len(nums) - 1 # len(nums) -1 will be the end of the list
#             while l < r: # left and right can't be equal 
#                 threeSum = a + nums[l] + nums[r] 
#                 if threeSum > 0:  
#                     r -= 1
#                 elif threeSum < 0:
#                     l += 1
#                 else:
#                     result.append([a, nums[l], nums[r]])
#                     l += 1
#                     while nums[l] == nums[l - 1] and l < r:
#                         l += 1
#         return result 


                    # [-2, -2, 0, 0, 2, 2]



# 11. Container with Most Water (Medium)
# Question Link https://leetcode.com/problems/container-with-most-water/
# Video Solution: https://www.youtube.com/watch?v=UuiTKBwPgAo
# Python Solution:

# class Solution:
#     def maxArea(self, height: List[int]) -> int:
        # # Brute Force Solution
        # result = 0

        # for l in range(len(height)): # left pter will be at every pos atleast once
        #     for r in range(l + 1, len(height)):# right pter one pos to the right of left pter
        #         area = (r - l) * min(height[l], height[r]) # width is (r - l), we need the bottleneck height
        #         result = max(result, area)
        # return result 

        # Linear Time Solution O(n)

        # result = 0 # area 
        # l, r = 0, len(height) - 1

        # while l < r:
        #     area = (r - l) * min(height[l], height[r])
        #     result = max(result, area)

        #     if height[l] < height[r]:
        #         l += 1 # we want to maximize both of the heights 
        #     else: # if they're equal or left > right pter 
        #         r -= 1
        # return result



# SLIDING WINDOW (0/4)

# 12. Best Time to Buy & Sell Stock (easy)
# Question Link https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# Video Solution: https://www.youtube.com/watch?v=1pkOgXD63yU
# Python Solution: 

# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         res = 0 # inital max is 0

#         l = 0 # left pter at 0 
#         for r in range(1, len(prices)): # right pter start at 1
#             if prices[r] < prices[l]: # buy high sell low
#                 l = r # shift left pter to right 
#             res = max(res, prices[r] - prices[l]) # high - low = maxProf
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
#     def isValid(self, s: str) -> bool:
#         stack = []
#         closeToOpen = {
#             ")" : "(", "}" : "{", "]" : "[" 
#         }

#         for c in s: # let's build our stack, then popping from it 
#             if c in closeToOpen: # it means it's a closing parens (b/c every key in our map is a closing parens)
#                 if stack and stack[-1] == closeToOpen[c]: # make sure stack is not empty and the closing parens mathches the opening
#                     stack.pop()
#                 else:
#                     return False
#             else:
#                 stack.append(c)

#         return True if not stack else False



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
#          prev, curr = None, head # Initialize our two pters 

#         while curr: # we want to keep iterating until we reach the end of the list (while curr is not null)
#             nxt = curr.next # temporary nxt variable pter 
#             curr.next = prev # This reverses the pointers (the direction of the links)
#             prev = curr # previous becomes the new head essentially 
#             curr = nxt 
#         return prev # result is stored in prev when this loop stops lo

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
        # dummy = ListNode() # to handle the edge case of an empty list, common technique of creating a dummy node with no value
        # tail = dummy

        # while list1 and list2: # while list1 and list2 are non null
        #     if list1.val < list2.val:
        #         tail.next = list1 # the next value in the tail will therefore be list1 
        #         list1 = list1.next # update our list1 pter
        #     else: # if list1.val > list2.val
        #         tail.next = list2
        #         list2 = list2.next
        #     tail = tail.next # still need to update the tail pter regardless of which list node is added

        # if list1: # if list1 is non null will take the remaining portion and add it to the tail of the list
        #     tail.next = list1
        # elif list2: # if list2 is non null will take the remaining portion and add it to the tail 
        #     tail.next = list2

        # return dummy.next # return the merged list 



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
#         slow, fast = head, head # two pters start same pos

#         while fast and fast.next: #
#             slow = slow.next # one shift
#             fast = fast.next.next # two shifts
#             if slow == fast: # if pters meet
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
#     def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         # DFS
     
#         if not root: # If the root is null
#             return None 
        
#         # If the root is not null, we're gonna swap the roots children
#         tmp = root.left # tmp variable for root.left children
#         root.left = root.right # Replace root.left w root.right to invert the children 
#         root.right = tmp
        
#         self.invertTree(root.left) # To invert all the proceeding subtrees children, we make a recursive call
#         self.invertTree(root.right) # To invert all the procreeding subtrees children, we make a recursive call
        
#         return root


# 26. Maximum Depth of Binary Tree (Easy)
# Question Link https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Video Solution: https://www.youtube.com/watch?v=hTM3phVI6YQ
# Python Solution:
# class Solution:
#     def maxDepth(self, root: Optional[TreeNode]) -> int:

        # 3 main ways to solve 1.Recursive DFS (best solution)  2.Iterative BFS 3.Iterative DFS
        
        
        # 1. Recursive DFS (in-order DFS)
        if not root: # If the root is empty 
            return 0 # Max depth will be zero
        
        else: 
  # we are making a recursive call to see what the max depth is of root.left and root.right
  # and it's plus b/c we know the root is not null so has to be a value of 1
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        

        # 2. Iterative BFS (basically level order traversal) (requires a queue)
        
        if not root: # same base case if root is null
            return 0 
        
        level = 0 
        
        queue = deque([root]) # root value is the only val initially in the queue b/c root in non null
        
        while queue: # will keep iterating untill the queue is empty
            
            for i in range(len(queue)): # traverse the whole level
                node = queue.popleft() # we are gonna pop a node
                if node.left: # if node.left is not null
                    queue.append(node.left) #  we will add node.left to the queue   
                if node.right: # if node.right is not null
                    queue.append(node.right) # we will add node.right to the queue
                    
                    
            level += 1 # incrementing the number of levels
        return level
    
    
        # 3. Iterative DFS (DFS w/o recursion) (requires a stack) (pre-order DFS easiest one iteratively)
         
        if not root: # same base case 
            return 0
        
        stack = [[root, 1]] # keeping track of root value, and the depth (has a val of 1 b/c root is not null)
        result = 1
        
        while stack: # we will loop through while stack is not empty 
            node, depth = stack.pop() # we're popping the node and it's depth 
            
            if node: # if node is not null
                result = max(result, depth)
                stack.append([node.left, depth + 1]) # we are adding the children of this node
                stack.append([node.right, depth + 1]) # we are adding the children of this node
        
        return Result   



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
#     def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
#         if not p and not q: # if both p and q are null they are the same 
#             return True
#         if not p or not q: # if one of the tree's is null and the other is not
#             return False
#         if p.val != q.val: # if the value of the root nodes are not eqaul to each other 
#             return False

        
#         return (self.isSameTree(p.left, q.left) and # recursive call for the left and right subtrees 
#                 self.isSameTree(p.right, q.right))



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
#     def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
#         if not subRoot: # If the subroot is null it is a subtree of root
#             return True
#         if not root: # if the root tree is null (no nodes in it) then we return False
#             return False

#         if self.sameTree(root, subRoot): # if the trees are the exactly the same, we can return True 
#             return True

#         return (self.isSubtree(root.left, subRoot) or # this checks if a subtree of the left or right is present 
#                 self.isSubtree(root.right, subRoot))

    
#     def sameTree(self, root, subRoot): 
#         if not root and not subRoot: # if both the root tree and subroot tree are empty, we can return True 
#             return True
        
#         if root and subRoot and root.val == subRoot.val: # if they are both non empty and their values equal each other 
#             return (self.sameTree(root.left, subRoot.left) and # we check to see if the values of the left and right tree vals equal each other
#                     self.sameTree(root.right, subRoot.right))
        
#         return False # this means the vals of the subtree does not equal the root tree vals for both the left and right 



# 29. Lowest Common Ancestor of a BST (Medium)
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












































